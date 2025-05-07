# Scripts Directory

This directory contains scripts for running experiments with different vision-language models (VLMs) on various datasets. The scripts are organized by model type and dataset.

## Directory Structure

```
script/
├── blip/
│   ├── blip_imagenet_script/
│   └── blip_coco_script/
├── llava/
│   ├── llava_example.sh
│   ├── llava_imagenet_script/
│   └── llava_coco_script/
└── minigpt4/
    ├── minigpt4_imagenet_script/
    └── minigpt4_coco_script/
```

## Models

The scripts are organized into three main model categories:

1. **LLaVA (Large Language and Vision Assistant)**
2. **MiniGPT-4**
3. **BLIP (Bootstrapping Language-Image Pre-training)**

## Script Types

### Attack Scripts
- Visual attack scripts for generating adversarial examples
- Parameters include:
  - `--n_iters`: Number of iterations (default: 2000)
  - `--constrained`: Constraint type ('constrained' or 'partial')
  - `--eps`: Epsilon value for perturbation (e.g., 32, 128)
  - `--alpha`: Alpha value for attack strength
  - `--image_file`: Input image path
  - `--save_dir`: Output directory for results

### Inference Scripts
- Scripts for running inference on generated adversarial examples
- Parameters include:
  - `--data_path`: Path to dataset
  - `--image_file`: Path to input image
  - `--output_file`: Path to save results

## Example Usage

### LLaVA Example (Sentiment Analysis - Positive)
```bash
# Running a visual attack for positive sentiment
python llava_llama_v2_visual_attack.py \
    --data_path instruction_data/eg4/dataset.csv \
    --instruction positive \
    --n_iters 2000 \
    --constrained constrained \
    --eps 128 \
    --alpha 1 \
    --image_file clean_images/eg1.png \
    --save_dir output/llava/eg1/Sentiment/Positive/partial_eps_128_batch_8

# Running inference on the generated adversarial example
python llava_llama_v2_inference.py \
    --data_path instruction_data/eg4/dataset.csv \
    --image_file output/llava/eg1/Sentiment/Positive/partial_eps_128_batch_8/bad_prompt.bmp \
    --output_file output/llava/eg1/Sentiment/Positive/partial_eps_128_batch_8/result.jsonl
```

### MiniGPT-4 Example (Sentiment Analysis - Positive)
```bash
# Running a visual attack for positive sentiment
python minigpt4_visual_attack.py \
    --data_path instruction_data/eg4/dataset.csv \
    --instruction positive \
    --n_iters 2000 \
    --constrained constrained \
    --eps 32 \
    --alpha 1 \
    --image_file clean_images/eg1.png \
    --save_dir output/minigpt4/eg1/Sentiment/Positive/partial_eps_128_batch_8

# Running inference on the generated adversarial example
python minigpt4_inference.py \
    --data_path instruction_data/eg1/dataset.csv \
    --image_file output/minigpt4/eg1/Sentiment/Positive/partial_eps_128_batch_8/bad_prompt.bmp \
    --output_file output/minigpt4/eg1/Sentiment/Positive/partial_eps_128_batch_8/result.jsonl
```

### BLIP Example (Sentiment Analysis - Positive)
```bash
# Running a visual attack for positive sentiment
python blip_visual_attack.py \
    --data_path instruction_data/eg4/dataset.csv \
    --instruction positive \
    --n_iters 2000 \
    --constrained constrained \
    --eps 32 \
    --alpha 1 \
    --image_file clean_images/eg4.png \
    --save_dir output/blip/eg4/Sentiment/Positive/partial_eps_128_batch_8

# Running inference on the generated adversarial example
python blip_inference.py \
    --data_path instruction_data/eg1/dataset.csv \
    --image_file output/blip/eg1/Sentiment/Positive/partial_eps_128_batch_8/bad_prompt.bmp \
    --output_file output/blip/eg1/Sentiment/Positive/partial_eps_128_batch_8/result.jsonl
```

## Dataset Support

The scripts support various datasets:
- ImageNet
- COCO
- Custom instruction datasets (e.g. examples you want to generate)

## Output Structure

Results are saved in the following structure:
```
output/
└── [model_name]/
    └── [experiment_name]/
        └── [task_type]/
            └── [constraint_type]_eps_[value]_batch_[size]/
                ├── bad_prompt.bmp
                └── result.jsonl
```