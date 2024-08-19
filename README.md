# Online RLHF

TL;DL: this is a repo to align the large language models (LLMs) by [online iterative RLHF](https://arxiv.org/pdf/2312.11456.pdf). Also check out our [technical report](https://arxiv.org/pdf/2405.07863) and [Huggingface Repo](https://huggingface.co/RLHFlow)!

We present the workflow of Online Iterative Reinforcement Learning from Human Feedback (RLHF), which is widely reported to outperform its offline counterpart by a large margin in the recent LLM literature. However, existing open-source RLHF projects are still largely confined to the offline learning setting. In this repo, we aim to fill in this gap and provide a detailed recipe that is easy to be reproduced for online iterative RLHF. In particular, with our recipe, with **only open-source data**, we can achieve comparable or even better results than LLaMA3-8B-instruct. 

<img width="1589" alt="image" src="eval_result.png">

## Model Releases
- [SFT model](https://huggingface.co/RLHFlow/LLaMA3-SFT)
- [Reward model](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1)
- [RLHF model](https://huggingface.co/RLHFlow/LLaMA3-iterative-DPO-final)

## Installation instructions

It is recommeded to have two separate environments for **inference** and **training**, respectively. 

**Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!**


**Inference Environment**

```sh
conda create -n vllm python=3.10.9
conda activate vllm
pip install datasets
# The following code is tested for CUDA12.0-12.2. You may need to update the torch and flash-attention sources according to your own CUDA version
pip install https://github.com/vllm-project/vllm/releases/download/v0.5.1/vllm-0.5.1-cp310-cp310-manylinux1_x86_64.whl
pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.0.9/flashinfer-0.0.9+cu121torch2.3-cp310-cp310-linux_x86_64.whl

pip install accelerate==0.33.0
pip install deepspeed==0.14.5
pip install transformers==4.43.4
pip install numpy==1.26.4 #Note that the numpy version should be `numpy<2.0`.  `Numpy 2.0` will encounter unexpected issues!!!
```

**Training Environment**

```sh
conda create -n rlhflow python=3.10.9
conda activate rlhflow

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
#git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install flash-attn==2.6.3
pip install accelerate==0.33.0
```

You also need to install the wandb to record the training and login with your huggingface account so that you have access to the LLaMA3 models.

```sh
pip install wandb

wandb login
huggingface-cli login
```

## Get Started
We present a step-by-step guidance in this section. 

### Step 1 Supervised Fine-tuning
To start with, you should first preprocess your dataset into the standard format. Here is an [example](https://huggingface.co/datasets/RLHFlow/SFT-OpenHermes-2.5-Standard) of the dataset. You may need to adjust the hyper-parameters (batch size, packing size) according to your computational resources. To run SFT, you can use the following command.

```sh
# You can adjust the training parameters in ./sft/sft.py
accelerate launch ./sft/sft.py

# Train with deepspeed stage3 
# You may need to adjust ./configs/zero3.yaml, especially the num_processes (the number of GPUs) according to your environment
accelerate launch --config_file ./configs/zero3.yaml ./sft/sft.py
```

### Step 2 Reward Modeling
We refer the interested readers to [this repo](https://github.com/RLHFlow/RLHF-Reward-Modeling) for a detailed recipe to train the state-of-the-art open-source reward/preference models. We have trained several RMs and prepared them on the huggingface like [sfairXC/FsfairX-LLaMA3-RM-v0.1](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1) and [RLHFlow/pair-preference-model-LLaMA3-8B](https://huggingface.co/RLHFlow/pair-preference-model-LLaMA3-8B), which are SOTA open-source RMs so far (2024 May).

<img width="1589" alt="image" src="https://github.com/RLHFlow/Iterative-RLHF-dev/assets/90632760/956449aa-f382-496a-8691-12c10bd24ea2">

### Step 3.1 Data Generation
To accelerate data generation, we use the VLLM. We prepare two ways of using VLLM to inference for a more robust implementation, where you can try them out and choose the one that fits with your environment best. We use LLaMA3-8B as an example. For other models, you need to adjust the eos_ids.

We can also use API server to generate new responses.

```sh
my_world_size=4
infer_model=meta-llama/Meta-Llama-3-8B-Instruct
prompt_dir=RLHFlow/test_generation_2k
mkdir data
output_dir=./data/gen_data.json
conda activate vllm

# register the api server
bash ./generation/register_server.sh $infer_model
python ./generation/gen_hf.py --ports 8000 8001 8002 8003 8004 8005 8006 8007 --eos_ids 128009 --tokenizer $infer_model --dataset_name_or_path $prompt_dir --output_dir $output_dir --K 4 --temperature 1.0
```

### Step 3.2 Data Annotation
Then, we call the reward/preference model trained in step 2 to rank the generated responses. 

```sh
accelerate launch ./annotate_data/get_rewards.py --dataset_name_or_path ./data/gen_data.json --output_dir ./data/data_with_rewards.json --K 4
```
If you encounter error ``TypeError: Got unsupported ScalarType BFloat16'', considering pip install transformers==4.38.2

### Step 3.3 Training

```sh
conda activate rlhflow
model_path=meta-llama/Meta-Llama-3-8B-Instruct
initial_model=meta-llama/Meta-Llama-3-8B-Instruct
mkdir models
accelerate launch --config_file ./configs/zero2.yaml ./dpo_iteration/run_dpo.py --run_name rlhflow_iter1 --output_dir ./models/rlhflow_iter1 --model_name_or_path $model_path --ref_model $initial_model --learning_rate 2e-7 --max_steps 1200 --choose_type max_min --train_dir ./data/data_with_rewards.json --eval_dir ./data/data_with_rewards.json --loss_type sigmoid --lr_scheduler_type cosine
```
If you encounter ``RuntimeError: CUDA error: invalid device ordinal, CUDA kernel errors might be asynchronously reported at some other API call'', you need to adjust num_of_process in the config file according to your GPUs.

### Putting Everything Together
We put everything together so that the iterative training can run automatically. Note that we set sleep 1m to wait for registering the API for inference. You may need to adjust this parameter according to your environment.

```sh
bash run_loop.sh
```

## Acknowledgement

The authors would like to thank the great open-source communities, including the Huggingface TRL team, the Huggingface H4 team, the Allen Institute AI RewardBench team, the Meta LLaMA team, and Axolotl team for sharing the models, codes, and training sets. 

## Citation

If you find the content of this repo useful, please consider cite it as follows:

```bibtex
@misc{dong2024rlhf,
      title={RLHF Workflow: From Reward Modeling to Online RLHF}, 
      author={Hanze Dong and Wei Xiong and Bo Pang and Haoxiang Wang and Han Zhao and Yingbo Zhou and Nan Jiang and Doyen Sahoo and Caiming Xiong and Tong Zhang},
      year={2024},
      eprint={2405.07863},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
@inproceedings{xiong2023iterative,
  title={Iterative preference learning from human feedback: Bridging theory and practice for RLHF under KL-constraint},
  author={Xiong, Wei and Dong, Hanze and Ye, Chenlu and Wang, Ziqi and Zhong, Han and Ji, Heng and Jiang, Nan and Zhang, Tong},
  booktitle={ICLR 2024 Workshop on Mathematical and Empirical Understanding of Foundation Models}
}
```
