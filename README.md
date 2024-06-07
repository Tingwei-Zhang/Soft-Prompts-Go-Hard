<h1 align="center"> <i>Soft Prompts Go Hard:</i>   <br>
Steering Visual Language Models with Hidden Meta-Instructions </h1>


## Setup

* Set up the environment following the instructions of the original repository at: https://github.com/haotian-liu/LLaVA and download the model at https://huggingface.co/meta-llama/Llama-2-13b-chat-hf?library=true

* After get the model, save it to the path `./ckpts/llava_llama_2_13b_chat_freeze`


## Generate data with meta-objectives

see `./instruction_data/README.md` for details


## Train image soft prompt


### LLaVA-Llama2-13b  (more scripts see `./script/llava`)


  ```bash
  python llava_llama_v2_visual_attack.py --data_path instruction_data/0/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/llava/0/Sentiment/Positive/constrained_eps_32_batch_8
  python llava_llama_v2_visual_attack.py --data_path instruction_data/0/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/llava/0/Sentiment/Negative/constrained_eps_32_batch_8
  python llava_llama_v2_visual_attack.py --data_path instruction_data/0/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/llava/0/Sentiment/Neutral/constrained_eps_32_batch_8

  ```


### MiniGPT-4 (more scripts see `./script/minigpt4`)

  ```bash
  python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/minigpt4/0/Sentiment/Positive/constrained_eps_32_batch_8
  python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/minigpt4/0/Sentiment/Negative/constrained_eps_32_batch_8
  python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/minigpt4/0/Sentiment/Neutral/constrained_eps_32_batch_8

  ```


## Evaluating image soft prompt

 1. No attack (clean image + text prompt)
   
* LLaVA
  ```bash
  python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file clean_images/0.png --output_file output/llava/0/baseline_1/result.jsonl
  ```
* MiniGPT-4
  ```bash
  python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --image_file clean_images/0.png --output_file output/minigpt4/0/baseline_1/result.jsonl
  ```

 2. Explicit instruction (clean image + instuction 'talk positively/negatively/neutrally' + text prompt)
* LLaVA
  ```bash
  python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file clean_images/0.png --output_file output/llava/0/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
  python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file clean_images/0.png --output_file output/llava/0/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
  python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file clean_images/0.png --output_file output/llava/0/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
  ```
* MiniGPT-4
  ```bash
  python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --image_file clean_images/0.png --output_file output/minigpt4/0/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
  python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --image_file clean_images/0.png --output_file output/minigpt4/0/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
  python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --image_file clean_images/0.png --output_file output/minigpt4/0/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
  ```

 3. Our attack (image soft prompt + text prompt)
* LLaVA
  ```bash
  python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file output/llava/0/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/0/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
  python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file output/llava/0/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/0/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
  python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file output/llava/0/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/0/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl
  ```
* MiniGPT-4
  ```bash
  python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --image_file output/minigpt4/0/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/0/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
  python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --image_file output/minigpt4/0/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/0/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
  python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --image_file output/minigpt4/0/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/0/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl
  ```
 5. Evaluate the meta-objective following rate
    run eval_instruction_following.ipynb 

## Image preservation:

1. Similarity between clean and perturbed image
   
    run eval_content_preserving.ipynb

2. Binary clssification (query the victim model if the label describes the conten) of the image:
 * LLaVA
```bash
python -u llava_llama_v2_inference.py --image_file clean_images/0.png --image_index 0 --output_file output/llava/0/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
```
* MiniGPT-4
 ```bash
 python -u minigpt_inference.py --gpu_id 0 --image_file clean_images/0.png --image_index 0 --output_file output/minigpt4/0/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
 ```
3. Query oracle model:
   "with yes or no, determine if the output is relevant to the "label" in the image and answers the question "text prompt"."