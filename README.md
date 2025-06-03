<h1 align="center"> <i>Soft Prompts Go Hard:</i>   <br>
Self-interpreting Adversarial Images </h1>

<img src="interesting_examples/paper_review.png" alt="drawing" width="1000"/>

[![arXiv](https://img.shields.io/badge/arXiv-2407.08970-b31b1b.svg)](https://arxiv.org/pdf/2407.08970)

We introduce a new type of indirect, cross-modal injection attacks against visual language models that enable creation of self-interpreting images. These images contain hidden "meta-instructions" that control how models answer users' questions about the image and steer their outputs to express an adversary-chosen style, sentiment, or point of view. 

Self-interpreting images act as soft prompts, conditioning the model to satisfy the adversary's (meta-)objective while still producing answers based on the image's visual content. Meta-instructions are thus a stronger form of prompt injection. Adversarial images look natural and the model's answers are coherent and plausible—yet they also follow the adversary-chosen interpretation, e.g., political spin, or even objectives that are not achievable with explicit text instructions. 

We evaluate the efficacy of self-interpreting images for a variety of models, interpretations, and user prompts. We describe how these attacks could cause harm by enabling creation of self-interpreting content that carries spam, misinformation, or spin. Finally, we discuss defenses.

## Setup

### LLaVA Setup
1. Follow the setup instructions from the [LLaVA repository](https://github.com/haotian-liu/LLaVA)
2. Download the Llama-2-13b-chat model from [Hugging Face](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf?library=true)
3. Save the downloaded model to: `./ckpts/llava_llama_2_13b_chat_freeze`

### MiniGPT-4 Setup
1. Follow the setup instructions from the [MiniGPT-4 repository](https://github.com/Vision-CAIR/MiniGPT-4/tree/main)
2. Download the 7B version of Vicuna V0 in `./ckpts/vicuna-7b`
3. Download the pretrained checkpoint from [here](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view) and update the path in `eval_configs/minigpt4_eval.yaml`

### Instruct BLIP Setup
1. Follow the setup instructions from the [LAVIS repository](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)
2. Select the 13B version model (blip2_vicuna_instruct-vicuna13b)
3. Download vicuna-13b v1.1 model to: `./ckpts/vicuna-13b-v1.1`

For additional guidance on visual adversarial examples, refer to the [Visual Adversarial Examples repository](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models?tab=readme-ov-file).

Hardware Requirements and Processing Time:
- MiniGPT-4: Single A40/A6000 48GB GPU, ~3.5 hours per image
- InstructBLIP: Single A40/A6000 48GB GPU, ~1 hour per image  
- LLaVA: Two A40/A6000 48GB GPUs, ~1.5 hours per image

## Generate and Evaluate Image Soft Prompts with Meta-objectives

### 1. Prepare Data
See [`instruction_data/README.md`](./instruction_data/README.md) for dataset and instruction details.

### 2. Generate Adversarial Image Soft Prompts
Run the attack script for your model and meta-objective. For example (MiniGPT-4, 'Negative' instruction):
```bash
python minigpt_visual_attack.py \
  --gpu_id 0 \
  --data_path instruction_data/0/Sentiment/dataset.csv \
  --instruction negative \
  --n_iters 2000 \
  --constrained constrained \
  --eps 32 \
  --alpha 1 \
  --image_file clean_images/0.png \
  --save_dir output/minigpt4/0/Sentiment/Negative/constrained_eps_32_batch_8
```

### 3. Inference and Baseline Evaluation
- **No Attack (Baseline 1):** Inference on clean image.
- **Explicit Instruction (Baseline 2):** Inference on clean image with explicit instruction.
- **Our Attack:** Inference on adversarial image.

Example (MiniGPT-4, Baseline 1):
```bash
python -u minigpt_inference.py \
  --gpu_id 0 \
  --data_path instruction_data/0/Sentiment/dataset.csv \
  --image_file clean_images/0.png \
  --output_file output/minigpt4/0/baseline_1/result.jsonl
```

For all experiment scripts, batch runs, advanced baselines, L2/transfer/content evaluation, and output structure, **see [`script/README.md`](script/README.md)**.

### 4. Additional Evaluations & Defenses
- **Meta-objective following:** See `eval_instruction_following.ipynb`.
- **Content preservation:** See `eval_content_preserving.ipynb`.
- **Content evaluation (LDI):** See `script/README.md` for commands.
- **JPEG defense:** See `script/README.md` for usage.
- **Anomaly detection:** See `eval_anomaly_detection.ipynb`.

