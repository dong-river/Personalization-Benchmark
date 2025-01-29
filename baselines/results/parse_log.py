import re
import os
import math
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from statistics import mean
import csv
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

@dataclass
class UserMetrics:
    accuracy: Optional[float] = None
    chat: Optional[float] = None
    chat_hard: Optional[float] = None
    safety: Optional[float] = None
    reasoning: Optional[float] = None

class LogParser:
    def __init__(self):
        self.users = defaultdict(UserMetrics)
        self.dataset_metrics = {}
        self.current_file = None

    def has_any_metrics(self) -> bool:
        """Check if any metrics were successfully parsed."""
        for user_id, metrics in self.users.items():
            if user_id == 0:  # Skip the special average user
                continue
            if any(getattr(metrics, field) is not None 
                   for field in ['accuracy', 'chat', 'chat_hard', 'safety', 'reasoning']):
                return True
        return False

    def calculate_averages(self):
        """Calculate averages for all metrics across users."""
        metrics = defaultdict(list)
        
        # Collect all non-None values for each metric
        for user_id, user_metrics in self.users.items():
            if user_id == 0:  # Skip the special average user
                continue
            for field in ['accuracy', 'chat', 'chat_hard', 'safety', 'reasoning']:
                value = getattr(user_metrics, field)
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    metrics[field].append(value)
        
        # Calculate averages
        avg_metrics = UserMetrics()
        for field, values in metrics.items():
            if values:  # Only calculate average if we have values
                setattr(avg_metrics, field, mean(values))
        
        self.users[0] = avg_metrics

    def parse_training_metrics(self, content: str):
        """Parse fine-tuning metrics including chat, safety, etc."""
        lines = content.split('\n')
        current_user = None
        
        for line in lines:
            # Track current user
            user_start = re.search(r'Training and Evaluating for user (\d+)', line)
            if user_start:
                current_user = int(user_start.group(1))
                continue
                
            # Parse initial metrics
            metric_match = re.search(r'Metrics for User (\d+): ([\d.]+)', line)
            if metric_match:
                user_id = int(metric_match.group(1))
                accuracy = float(metric_match.group(2))
                self.users[user_id].accuracy = accuracy
                
            # Parse dataset metrics if we have a current user
            if current_user is not None:
                dataset_match = re.search(r"Metrics for dataset \['([^']+)'\]: ([\d.]+)", line)
                if dataset_match:
                    dataset, score = dataset_match.group(1), float(dataset_match.group(2))
                    if dataset == 'chat':
                        self.users[current_user].chat = score
                    elif dataset == 'chat hard':
                        self.users[current_user].chat_hard = score
                    elif dataset == 'safety':
                        self.users[current_user].safety = score
                    elif dataset == 'reasoning':
                        self.users[current_user].reasoning = score

    def parse_rag_metrics(self, content: str):
        """Parse RAG-specific metrics and named metrics in accuracy dictionary."""
        rag_match = re.search(r'User accuracy: {([^}]+)} for model .+', content)
        if rag_match:
            metrics_str = rag_match.group(1)
            
            # Handle both quoted string keys and numeric keys
            metrics_pattern = re.finditer(r"(?:'([^']+)'|(\d+)): ([\d.]+|nan)", metrics_str)
            
            # Create data structure to store named metrics
            named_metrics = {}
            
            for match in metrics_pattern:
                string_key = match.group(1)  # 'chat', 'chat hard', etc.
                numeric_key = match.group(2)  # '1', '2', etc.
                value_str = match.group(3)
                
                value = float('nan') if value_str == 'nan' else float(value_str)
                
                if string_key:
                    # Store named metrics temporarily
                    if string_key == 'chat':
                        named_metrics['chat'] = value
                    elif string_key == 'chat hard':
                        named_metrics['chat_hard'] = value
                    elif string_key == 'safety':
                        named_metrics['safety'] = value
                    elif string_key == 'reasoning':
                        named_metrics['reasoning'] = value
                elif numeric_key:
                    # Handle user accuracies
                    user_id = int(numeric_key)
                    user_metrics = self.users[user_id]
                    user_metrics.accuracy = value
                    # Apply named metrics if we have them
                    if 'chat' in named_metrics:
                        user_metrics.chat = named_metrics['chat']
                    if 'chat_hard' in named_metrics:
                        user_metrics.chat_hard = named_metrics['chat_hard']
                    if 'safety' in named_metrics:
                        user_metrics.safety = named_metrics['safety']
                    if 'reasoning' in named_metrics:
                        user_metrics.reasoning = named_metrics['reasoning']

    def parse_id_metrics(self, content: str):
        """Parse ID-specific metrics."""
        lines = content.split('\n')
        for line in lines:
            # Parse numbered dataset metrics (treat these as user accuracies)
            dataset_num_match = re.search(r"Metrics for dataset \[(\d+)\]: ([\d.]+)", line)
            if dataset_num_match:
                user_id = int(dataset_num_match.group(1))
                accuracy = float(dataset_num_match.group(2))
                self.users[user_id].accuracy = accuracy
                
            # Parse named dataset metrics
            dataset_match = re.search(r"Metrics for dataset \['([^']+)'\]: ([\d.]+)", line)
            if dataset_match:
                dataset, score = dataset_match.group(1), float(dataset_match.group(2))
                # Apply to all users we've seen
                for user_metrics in self.users.values():
                    if dataset == 'chat':
                        user_metrics.chat = score
                    elif dataset == 'chat hard':
                        user_metrics.chat_hard = score
                    elif dataset == 'safety':
                        user_metrics.safety = score
                    elif dataset == 'reasoning':
                        user_metrics.reasoning = score

    def parse_prm_metrics(self, content: str):
        """Parse PRM-specific metrics."""
        lines = content.split('\n')
        for line in lines:
            user_match = re.search(r'eval_rewards/user_(\d+)_accuracies: ([\d.]+)', line)
            if user_match:
                user_id = int(user_match.group(1))
                accuracy = float(user_match.group(2))
                self.users[user_id].accuracy = accuracy

    def parse_filename(self, filepath: str) -> dict:
        """Parse configuration from filename."""
        filename = os.path.basename(filepath)
        config = {
            'method': 'NA',
            'model': 'NA', 
            'dataset': 'NA',
            'train_size': 'NA',
            'learning_rate': 'NA',
            'epoch': 'NA'
        }
        
        # Define valid options
        method_list = ['ft_rm_general', 'ft_rm', 'rag', 'prm', 'id_rm', 'vae', 'gpo', 'pretrained']
        model_list = ['gemma', 'llama', 'starling', 'rm-mistral']
        data_list = ['psoups', 'summarization', 'personal_llm', 'ultrafeedback']
        train_size_list = ['200000', '100000', '30000', '10000', '3000', '1000', '300']
        lr_list = ['3e-6', '5e-5', '0.0003', '1e-5', '0.0001', '3e-5', '1e-06', '5e-06', '1e-4', '3e-4', '1e-3']
        epoch_list = ['_1_', '_2_', '_3_', '_5_', '_10_', '_20_', '_30_', '_50_' '_100_']
        
        # Parse method
        for method in method_list:
            if method in filename.lower():
                config['method'] = method
                break
                
        # Parse model
        for model in model_list:
            if model in filename.lower():
                config['model'] = model
                break
                
        # Parse dataset
        for dataset in data_list:
            if dataset in filename.lower():
                config['dataset'] = dataset
                break
                
        # Parse training size
        for size in train_size_list:
            if size in filename:
                config['train_size'] = size
                break
                
        # Parse learning rate
        for lr in lr_list:
            if lr in filename:
                config['learning_rate'] = lr
                break
        
        # Parse epoch
        for epoch in epoch_list:
            if epoch in filename:
                config['epoch'] = epoch
                break
        return config

    def parse_file(self, filepath: str):
        """Parse a log file based on its filename pattern."""
        self.current_file = filepath
        with open(filepath, 'r') as f:
            content = f.read()

        if "rag" in filepath.lower():
            self.parse_rag_metrics(content)
        elif "ft_rm" in filepath.lower():
            try:
                self.parse_training_metrics(content)
                if not self.has_any_metrics():
                    self.parse_id_metrics(content)
            except:
                self.parse_id_metrics(content)
        elif "prm" in filepath.lower():
            self.parse_prm_metrics(content)
        elif "id_rm" in filepath.lower():
            self.parse_id_metrics(content)
        else:
            print(f"Warning: Unrecognized log format for file {filepath}")
            self.parse_training_metrics(content)
            self.parse_rag_metrics(content)
            self.parse_prm_metrics(content)
            self.parse_id_metrics(content)
            
        self.calculate_averages()

    def get_extended_table_data(self) -> tuple[List[str], List[List]]:
        """Convert metrics and configuration to extended table format."""
        # Get configuration from filename
        config = self.parse_filename(self.current_file)
        
        # Find all users excluding 0 (average)
        users = sorted([user_id for user_id in self.users.keys() if user_id != 0])
        
        # Create headers
        base_headers = ["Filename", "Method", "Model", "Dataset", "Train Size", "Learning Rate"]
        metric_headers = []
        for user_id in users:
            metric_headers.extend([
                f"User{user_id}_Acc",
                f"User{user_id}_Chat",
                f"User{user_id}_ChatH",
                f"User{user_id}_Safe",
                f"User{user_id}_Reas"
            ])
        
        headers = base_headers + ["Avg_Acc", "Avg_Chat", "Avg_ChatH", "Avg_Safe", "Avg_Reas"] + metric_headers
        
        # Create row
        row = [
            os.path.basename(self.current_file),
            config['method'],
            config['model'],
            config['dataset'],
            config['train_size'],
            config['learning_rate'],
            config['epoch']
        ]
        
        # Add averages
        avg_metrics = self.users[0]
        row.extend([
            "nan" if isinstance(avg_metrics.accuracy, float) and math.isnan(avg_metrics.accuracy) else f"{avg_metrics.accuracy:.3f}" if avg_metrics.accuracy is not None else "-",
            f"{avg_metrics.chat:.3f}" if avg_metrics.chat is not None else "-",
            f"{avg_metrics.chat_hard:.3f}" if avg_metrics.chat_hard is not None else "-",
            f"{avg_metrics.safety:.3f}" if avg_metrics.safety is not None else "-",
            f"{avg_metrics.reasoning:.3f}" if avg_metrics.reasoning is not None else "-"
        ])
        
        # Add per-user metrics
        for user_id in users:
            metrics = self.users[user_id]
            row.extend([
                "nan" if isinstance(metrics.accuracy, float) and math.isnan(metrics.accuracy) else f"{metrics.accuracy:.3f}" if metrics.accuracy is not None else "-",
                f"{metrics.chat:.3f}" if metrics.chat is not None else "-",
                f"{metrics.chat_hard:.3f}" if metrics.chat_hard is not None else "-",
                f"{metrics.safety:.3f}" if metrics.safety is not None else "-",
                f"{metrics.reasoning:.3f}" if metrics.reasoning is not None else "-"
            ])
        
        return headers, [row]

    def get_formatted_table(self) -> str:
        """Generate a formatted table of metrics."""
        if not self.has_any_metrics():
            return None
                
        headers, rows = self.get_extended_table_data()
        
        if HAS_TABULATE:
            return tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".3f")
        else:
            # Custom table formatting if tabulate is not available
            col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
            header_line = " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths))
            separator = "-|-".join("-" * width for width in col_widths)
            formatted_rows = [
                " | ".join(f"{cell:<{width}}" for cell, width in zip(row, col_widths))
                for row in rows
            ]
            return "\n".join([header_line, separator] + formatted_rows)

class GlobalStats:
    def __init__(self):
        self.user_metrics = defaultdict(lambda: defaultdict(list))  # user_id -> metric_type -> list of values
        
    def add_metrics(self, user_id: int, metrics: UserMetrics):
        """Add metrics for a user, excluding None and nan values."""
        for field in ['accuracy', 'chat', 'chat_hard', 'safety', 'reasoning']:
            value = getattr(metrics, field)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                self.user_metrics[user_id][field].append(value)
    
    def get_averages(self) -> dict:
        """Calculate averages for each user and metric."""
        averages = defaultdict(dict)
        for user_id, metrics in self.user_metrics.items():
            for metric_type, values in metrics.items():
                if values:  # Only calculate if we have values
                    averages[user_id][metric_type] = mean(values)
        return averages

def parse_logs(log_directory: str, output_file: str = None):
    """Parse all log files in the directory and track global statistics."""
    import glob
    import csv
    import os
    
    # Get all .log files
    log_files = glob.glob(f"{log_directory}/*.log")
    
    # Initialize global stats
    all_rows = []
    
    # Create headers with list-based metrics
    headers = [
        "Filename", "Method", "Model", "Dataset", "Train_Size", "Learning_Rate", "Epoch",
        "Avg_Acc", "Avg_Chat", "Avg_ChatH", "Avg_Safe", "Avg_Reas",
        "Acc_List", "Chat_List", "ChatH_List", "Safe_List", "Reas_List"
    ]
    
    # Process each log file
    for log_file in log_files:
        print(f"\n{'='*50}")
        print(f"Parsing {os.path.basename(log_file)}...")
        print(f"{'='*50}")
        
        parser = LogParser()
        parser.parse_file(log_file)
        
        if parser.has_any_metrics():
            # Create base row
            config = parser.parse_filename(log_file)
            row = [
                os.path.basename(log_file),
                config['method'],
                config['model'],
                config['dataset'],
                config['train_size'],
                config['learning_rate'],
                config['epoch']
            ]
            
            # Add averages
            avg_metrics = parser.users[0]
            row.extend([
                "nan" if isinstance(avg_metrics.accuracy, float) and math.isnan(avg_metrics.accuracy) else f"{avg_metrics.accuracy:.3f}" if avg_metrics.accuracy is not None else "NA",
                f"{avg_metrics.chat:.3f}" if avg_metrics.chat is not None else "NA",
                f"{avg_metrics.chat_hard:.3f}" if avg_metrics.chat_hard is not None else "NA",
                f"{avg_metrics.safety:.3f}" if avg_metrics.safety is not None else "NA",
                f"{avg_metrics.reasoning:.3f}" if avg_metrics.reasoning is not None else "NA"
            ])
            
            # Get all users except 0 (average)
            users = sorted([user_id for user_id in parser.users.keys() if user_id != 0])
            
            # Create lists for each metric type
            acc_list = []
            chat_list = []
            chath_list = []
            safe_list = []
            reas_list = []
            
            for user_id in users:
                metrics = parser.users[user_id]
                
                # Add each metric to its respective list
                acc_list.append("nan" if isinstance(metrics.accuracy, float) and math.isnan(metrics.accuracy) 
                              else f"{metrics.accuracy:.3f}" if metrics.accuracy is not None else "NA")
                chat_list.append(f"{metrics.chat:.3f}" if metrics.chat is not None else "NA")
                chath_list.append(f"{metrics.chat_hard:.3f}" if metrics.chat_hard is not None else "NA")
                safe_list.append(f"{metrics.safety:.3f}" if metrics.safety is not None else "NA")
                reas_list.append(f"{metrics.reasoning:.3f}" if metrics.reasoning is not None else "NA")
            
            # Add lists to row
            row.extend([
                str(acc_list),
                str(chat_list),
                str(chath_list),
                str(safe_list),
                str(reas_list)
            ])
            
            all_rows.append(row)
        else:
            print("\nCould not parse any metrics from this file.")
            print("\nOriginal file content:")
            print("-" * 50)
            with open(log_file, 'r') as f:
                print(f.read())
            print("-" * 50)
    
    # Save results to CSV
    if all_rows:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(all_rows)
        print(f"\nResults saved to: {output_file}")
        
        # Also print the table for viewing
        if HAS_TABULATE:
            print("\nComplete Results Table:")
            print(tabulate(all_rows, headers=headers, tablefmt="grid", floatfmt=".3f"))

if __name__ == "__main__":
    import sys
    
    log_dir = '/home/yd358/rds/hpc-work/analysis_pers/baselines/results'
    output_file = '/home/yd358/rds/hpc-work/analysis_pers/parsed_results.csv'
    parse_logs(log_dir, output_file)