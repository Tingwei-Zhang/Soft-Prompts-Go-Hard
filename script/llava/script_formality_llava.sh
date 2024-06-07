# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/0/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/llava/0/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/0/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/llava/0/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/0/Formality/dataset.csv --image_file clean_images/0.png --output_file output/llava/0/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/0/Formality/dataset.csv --image_file clean_images/0.png --output_file output/llava/0/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/0/Formality/dataset.csv --image_file clean_images/0.png --output_file output/llava/0/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/0/Formality/dataset.csv --image_file output/llava/0/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/0/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/0/Formality/dataset.csv --image_file output/llava/0/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/0/Formality/Informal/constrained_eps_32_batch_8/result.jsonl

# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/1/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/1.png --save_dir output/llava/1/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/1/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/1.png --save_dir output/llava/1/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/1/Formality/dataset.csv --image_file clean_images/1.png --output_file output/llava/1/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/1/Formality/dataset.csv --image_file clean_images/1.png --output_file output/llava/1/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/1/Formality/dataset.csv --image_file clean_images/1.png --output_file output/llava/1/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/1/Formality/dataset.csv --image_file output/llava/1/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/1/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/1/Formality/dataset.csv --image_file output/llava/1/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/1/Formality/Informal/constrained_eps_32_batch_8/result.jsonl

# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/2/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/2.png --save_dir output/llava/2/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/2/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/2.png --save_dir output/llava/2/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/2/Formality/dataset.csv --image_file clean_images/2.png --output_file output/llava/2/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/2/Formality/dataset.csv --image_file clean_images/2.png --output_file output/llava/2/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/2/Formality/dataset.csv --image_file clean_images/2.png --output_file output/llava/2/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/2/Formality/dataset.csv --image_file output/llava/2/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/2/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/2/Formality/dataset.csv --image_file output/llava/2/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/2/Formality/Informal/constrained_eps_32_batch_8/result.jsonl

# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/3/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/3.png --save_dir output/llava/3/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/3/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/3.png --save_dir output/llava/3/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/3/Formality/dataset.csv --image_file clean_images/3.png --output_file output/llava/3/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/3/Formality/dataset.csv --image_file clean_images/3.png --output_file output/llava/3/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/3/Formality/dataset.csv --image_file clean_images/3.png --output_file output/llava/3/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/3/Formality/dataset.csv --image_file output/llava/3/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/3/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/3/Formality/dataset.csv --image_file output/llava/3/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/3/Formality/Informal/constrained_eps_32_batch_8/result.jsonl


# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/4/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/4.png --save_dir output/llava/4/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/4/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/4.png --save_dir output/llava/4/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/4/Formality/dataset.csv --image_file clean_images/4.png --output_file output/llava/4/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/4/Formality/dataset.csv --image_file clean_images/4.png --output_file output/llava/4/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/4/Formality/dataset.csv --image_file clean_images/4.png --output_file output/llava/4/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/4/Formality/dataset.csv --image_file output/llava/4/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/4/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/4/Formality/dataset.csv --image_file output/llava/4/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/4/Formality/Informal/constrained_eps_32_batch_8/result.jsonl

# evaluating the content
python -u llava_llama_v2_inference.py --image_file output/llava/0/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 0 --output_file output/llava/0/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl  --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/0/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 0 --output_file output/llava/0/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/1/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 1 --output_file output/llava/1/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/1/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 1 --output_file output/llava/1/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/2/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 2 --output_file output/llava/2/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/2/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 2 --output_file output/llava/2/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/3/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 3 --output_file output/llava/3/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/3/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 3 --output_file output/llava/3/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/4/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 4 --output_file output/llava/4/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/4/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 4 --output_file output/llava/4/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
