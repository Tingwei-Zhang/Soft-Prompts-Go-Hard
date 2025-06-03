#Image 0
#Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/0/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/llava/0/Sentiment/Positive/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/0/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/llava/0/Sentiment/Negative/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/0/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/llava/0/Sentiment/Neutral/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file clean_images/0.png --output_file output/llava/0/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file clean_images/0.png --output_file output/llava/0/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file clean_images/0.png --output_file output/llava/0/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file clean_images/0.png --output_file output/llava/0/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
#Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file output/llava/0/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/0/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file output/llava/0/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/0/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/0/Sentiment/dataset.csv --image_file output/llava/0/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/0/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

#Image 1
#Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/1/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/1.png --save_dir output/llava/1/Sentiment/Positive/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/1/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/1.png --save_dir output/llava/1/Sentiment/Negative/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/1/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/1.png --save_dir output/llava/1/Sentiment/Neutral/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/1/Sentiment/dataset.csv --image_file clean_images/1.png --output_file output/llava/1/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/1/Sentiment/dataset.csv --image_file clean_images/1.png --output_file output/llava/1/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u llava_llama_v2_inference.py --data_path instruction_data/1/Sentiment/dataset.csv --image_file clean_images/1.png --output_file output/llava/1/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u llava_llama_v2_inference.py --data_path instruction_data/1/Sentiment/dataset.csv --image_file clean_images/1.png --output_file output/llava/1/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/1/Sentiment/dataset.csv --image_file output/llava/1/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/1/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/1/Sentiment/dataset.csv --image_file output/llava/1/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/1/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/1/Sentiment/dataset.csv --image_file output/llava/1/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/1/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

#Image 2
#Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/2/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/2.png --save_dir output/llava/2/Sentiment/Positive/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/2/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/2.png --save_dir output/llava/2/Sentiment/Negative/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/2/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/2.png --save_dir output/llava/2/Sentiment/Neutral/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/2/Sentiment/dataset.csv --image_file clean_images/2.png --output_file output/llava/2/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/2/Sentiment/dataset.csv --image_file clean_images/2.png --output_file output/llava/2/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u llava_llama_v2_inference.py --data_path instruction_data/2/Sentiment/dataset.csv --image_file clean_images/2.png --output_file output/llava/2/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u llava_llama_v2_inference.py --data_path instruction_data/2/Sentiment/dataset.csv --image_file clean_images/2.png --output_file output/llava/2/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
#Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/2/Sentiment/dataset.csv --image_file output/llava/2/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/2/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/2/Sentiment/dataset.csv --image_file output/llava/2/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/2/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/2/Sentiment/dataset.csv --image_file output/llava/2/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/2/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

#Image 3
#Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/3/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/3.png --save_dir output/llava/3/Sentiment/Positive/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/3/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/3.png --save_dir output/llava/3/Sentiment/Negative/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/3/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/3.png --save_dir output/llava/3/Sentiment/Neutral/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/3/Sentiment/dataset.csv --image_file clean_images/3.png --output_file output/llava/3/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/3/Sentiment/dataset.csv --image_file clean_images/3.png --output_file output/llava/3/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u llava_llama_v2_inference.py --data_path instruction_data/3/Sentiment/dataset.csv --image_file clean_images/3.png --output_file output/llava/3/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u llava_llama_v2_inference.py --data_path instruction_data/3/Sentiment/dataset.csv --image_file clean_images/3.png --output_file output/llava/3/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
#Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/3/Sentiment/dataset.csv --image_file output/llava/3/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/3/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/3/Sentiment/dataset.csv --image_file output/llava/3/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/3/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/3/Sentiment/dataset.csv --image_file output/llava/3/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/3/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

# Image 4
# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/4/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/4.png --save_dir output/llava/4/Sentiment/Positive/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/4/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/4.png --save_dir output/llava/4/Sentiment/Negative/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/4/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/4.png --save_dir output/llava/4/Sentiment/Neutral/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/4/Sentiment/dataset.csv --image_file clean_images/4.png --output_file output/llava/4/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/4/Sentiment/dataset.csv --image_file clean_images/4.png --output_file output/llava/4/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u llava_llama_v2_inference.py --data_path instruction_data/4/Sentiment/dataset.csv --image_file clean_images/4.png --output_file output/llava/4/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u llava_llama_v2_inference.py --data_path instruction_data/4/Sentiment/dataset.csv --image_file clean_images/4.png --output_file output/llava/4/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
#Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/4/Sentiment/dataset.csv --image_file output/llava/4/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/4/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/4/Sentiment/dataset.csv --image_file output/llava/4/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/4/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/4/Sentiment/dataset.csv --image_file output/llava/4/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/4/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

# evaluating the content
python -u llava_llama_v2_inference.py --image_file output/llava/0/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 0 --output_file output/llava/0/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl  --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/0/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 0 --output_file output/llava/0/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/0/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 0 --output_file output/llava/0/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/1/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 1 --output_file output/llava/1/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/1/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 1 --output_file output/llava/1/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/1/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 1 --output_file output/llava/1/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/2/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 2 --output_file output/llava/2/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/2/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 2 --output_file output/llava/2/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/2/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 2 --output_file output/llava/2/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/3/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 3 --output_file output/llava/3/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/3/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 3 --output_file output/llava/3/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/3/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 3 --output_file output/llava/3/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/4/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 4 --output_file output/llava/4/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/4/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 4 --output_file output/llava/4/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/4/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 4 --output_file output/llava/4/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation

# evaluating the content of clean images
python -u llava_llama_v2_inference.py --image_file clean_images/0.png --image_index 0 --output_file output/llava/0/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file clean_images/1.png --image_index 0 --output_file output/llava/1/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file clean_images/2.png --image_index 0 --output_file output/llava/2/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file clean_images/3.png --image_index 0 --output_file output/llava/3/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file clean_images/4.png --image_index 0 --output_file output/llava/4/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
