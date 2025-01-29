import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from statistics import mean
import math
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
                if value is not None:
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
        # More general pattern that matches any model name
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


    def parse_prm_metrics(self, content: str):
        """Parse PRM-specific metrics."""
        lines = content.split('\n')
        for line in lines:
            user_match = re.search(r'eval_rewards/user_(\d+)_accuracies: ([\d.]+)', line)
            if user_match:
                user_id = int(user_match.group(1))
                accuracy = float(user_match.group(2))
                self.users[user_id].accuracy = accuracy

    def parse_id_metrics(self, content: str):
        """Parse ID-specific metrics including dataset numbers and named metrics."""
        lines = content.split('\n')
        
        for line in lines:
            # Parse numbered dataset metrics (treat these as user accuracies)
            dataset_num_match = re.search(r"Metrics for dataset \[(\d+)\]: ([\d.]+)", line)
            if dataset_num_match:
                user_id = int(dataset_num_match.group(1))  # Use dataset number as user ID
                accuracy = float(dataset_num_match.group(2))
                self.users[user_id].accuracy = accuracy
                
            # Parse named dataset metrics (store in all users that have accuracies)
            dataset_match = re.search(r"Metrics for dataset \['([^']+)'\]: ([\d.]+)", line)
            if dataset_match:
                dataset, score = dataset_match.group(1), float(dataset_match.group(2))
                # Store these metrics for all users we've seen
                for user_metrics in self.users.values():
                    if dataset == 'chat':
                        user_metrics.chat = score
                    elif dataset == 'chat hard':
                        user_metrics.chat_hard = score
                    elif dataset == 'safety':
                        user_metrics.safety = score
                    elif dataset == 'reasoning':
                        user_metrics.reasoning = score
                    
    def parse_file(self, filepath: str):
        """Parse a log file based on its filename pattern."""
        with open(filepath, 'r') as f:
            content = f.read()

        if "rag" in filepath.lower() or 'pretrained' in filepath.lower():
            self.parse_rag_metrics(content)
        elif "ft_rm" in filepath.lower():
            # Try training metrics first, fall back to ID metrics if that fails
            try:
                self.parse_training_metrics(content)
                if not self.has_any_metrics():  # If no metrics were parsed successfully
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

    def get_table_data(self) -> tuple[List[str], List[List]]:
        """Convert metrics to table format."""
        headers = ["User", "Accuracy", "Chat", "Chat-Hard", "Safety", "Reasoning"]
        rows = []
        
        # Add average row first
        avg_metrics = self.users[0]
        rows.append([
            "Avg",
            "nan" if isinstance(avg_metrics.accuracy, float) and math.isnan(avg_metrics.accuracy) else f"{avg_metrics.accuracy:.3f}" if avg_metrics.accuracy is not None else "-",
            f"{avg_metrics.chat:.3f}" if avg_metrics.chat is not None else "-",
            f"{avg_metrics.chat_hard:.3f}" if avg_metrics.chat_hard is not None else "-",
            f"{avg_metrics.safety:.3f}" if avg_metrics.safety is not None else "-",
            f"{avg_metrics.reasoning:.3f}" if avg_metrics.reasoning is not None else "-"
        ])

        # Add user rows
        for user_id in sorted(self.users.keys()):
            if user_id == 0:  # Skip average metrics as we already added them
                continue
                
            metrics = self.users[user_id]
            row = [
                str(user_id),
                "nan" if isinstance(metrics.accuracy, float) and math.isnan(metrics.accuracy) else f"{metrics.accuracy:.3f}" if metrics.accuracy is not None else "-",
                f"{metrics.chat:.3f}" if metrics.chat is not None else "-",
                f"{metrics.chat_hard:.3f}" if metrics.chat_hard is not None else "-",
                f"{metrics.safety:.3f}" if metrics.safety is not None else "-",
                f"{metrics.reasoning:.3f}" if metrics.reasoning is not None else "-"
            ]
            rows.append(row)
            
        return headers, rows

    def get_formatted_table(self) -> str:
        """Generate a formatted table of metrics."""
        if not self.has_any_metrics():
            return None
            
        headers, rows = self.get_table_data()
        
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

def parse_logs(log_directory: str):
    """Parse all log files in the specified directory."""
    import glob
    import os
    
    # Get all .log files
    log_files = glob.glob(f"{log_directory}/*.log")
    
    # Parse each log file
    for log_file in log_files:
        # if 'rag' not in log_file:
        #     continue
        
        if "GPO" in log_file or "vae" in log_file:
            continue
        
        print(f"\n{'='*50}")
        print(f"Parsing {os.path.basename(log_file)}...")
        print(f"{'='*50}")
        
        # Parse and display results
        parser = LogParser()
        parser.parse_file(log_file)
        
        table = parser.get_formatted_table()
        if table:
            print("\nParsed Metrics:")
            print(table)
        else:
            print("\nCould not parse any metrics from this file.")
            print("\nOriginal file content:")
            print("-" * 50)
            with open(log_file, 'r') as f:
                print(f.read())
            print("-" * 50)

if __name__ == "__main__":
    import sys
    
    log_dir = '/home/yd358/rds/hpc-work/analysis_pers/baselines/results_Jan5'
    parse_logs(log_dir)