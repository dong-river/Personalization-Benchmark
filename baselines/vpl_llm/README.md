## Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning

####  [[Website]](https://weirdlabuw.github.io/vpl/) [[Paper]](https://arxiv.org/) 

[Sriyash Poddar<sup>1</sup>](https://sriya.sh), [Yanming Wan<sup>1</sup>](https://wanyanming.com/), [Hamish Ivison<sup>1</sup>](https://hamishivi.github.io/), [Abhishek Gupta<sup>1</sup>](https://homes.cs.washington.edu/~abhgupta), [Natasha Jaques<sup>1</sup>](https://natashajaques.ai)<br/>

<sup>1</sup>University of Washington

This repo is an implementation of the language experiments of VPL. VPL is a variational framework for learning from human feedback (binary preference labels) i.e. inferring a novel user-specific latent and learning reward models and policies conditioned on this latent without additional user-specific data. This is used for quick adaptation to specific user preferences without retraining the entire model or ignoring underrepresented groups.

For control experiments of VPL, please refer to [here](https://github.com/WEIRDLabUW/vpl).

## Instructions


#### Setting up repo
```
git clone git@github.com:WEIRDLabUW/vpl_llm.git
```

#### Install Dependencies
```
conda create -n vpl python=3.10
conda activate vpl
pip install -r requirements.txt
```

## Data and Pretrained models

Our datasets and checkpoints can be downloaded from [Google Drive link](https://drive.google.com/drive/folders/1dQ8zpNefRAtUB9TtbovOSn2MfV2Y-MbC?usp=sharing).

#### Datasets
The datasets needed for VPL experiments should be downloaded and unzipped to ``./data/``. There are three datasets in the folder: ``simple_pets``, ``P_survey_100``, and ``P_4_survey_100``.

#### Checkpoints
The checkpoints for VPL experiments should be downloaded to ``./logs``. We provide the checkpoints for VPL and other baseline models over each dataset.

## Dataset Generation
We also provide the code for generating our datasets. 
The following scripts will also give you the datasets in ``./data/``.
#### Pets
To generate ``./data/simple_pets``, run
```bash
bash generate_llm_embeddings_pets.sh gpt2
```

#### UF-P-2
To generate ``./data/P_survey_100``, run
```bash
python -m hidden_context.data_utils.ultrafeedback_augment -a 84 -n P
bash generate_llm_embeddings_UF_P_2.sh gpt2 84
```

#### UF-P-4
To generate ``./data/P_4_survey_100``, run
```bash
python -m hidden_context.data_utils.ultrafeedback_augment -a single -n P_4 -c
bash generate_llm_embeddings_UF_P_4.sh gpt2 single
```

## Running Experiments
In all the following scripts, ``<MODEL_TYPE>`` can be chosen from ``vae``, ``base``, ``categorical``, and ``mean_and_variance``. 
``vae`` corresponds to our VPL models, while the others are training baseline models.

The results are recorded on Wandb. Please refer to ``eval/accuracy`` on Wandb page for model's performance.

#### Pets
To train models on ``./data/simple_pets``, run
```bash
bash submit_job_pets.sh <MODEL_TYPE>
```
Note that the default settings are for Pets (full), 
please change the arguments as explained in the bash file if you want to train on Pets (controversial).

#### UF-P-2
To train models on ``./data/P_survey_100``, run
```bash
bash submit_job_UF_P_2.sh <MODEL_TYPE>
```

#### UF-P-4
To train models on ``./data/P_4_survey_100``, run
```bash
bash submit_job_UF_P_4.sh <MODEL_TYPE>
```

## Bibtex
If you find this code useful, please cite:

```
@article{poddar2024vpl,
    author    = {Poddar, Sriyash and Wan, Yanming and Ivision, Hamish and Gupta, Abhishek and Jaques, Natasha},
    title     = {Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning},
    booktitle = {ArXiv Preprint},
    year      = {2024},
}
```