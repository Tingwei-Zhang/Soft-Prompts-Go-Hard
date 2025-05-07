# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_1/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_1.jpg --save_dir output/llava/coco_1/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_1/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_1.jpg --save_dir output/llava/coco_1/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_1/Formality/dataset.csv --image_file clean_images/coco_1.jpg --output_file output/llava/coco_1/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_1/Formality/dataset.csv --image_file clean_images/coco_1.jpg --output_file output/llava/coco_1/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_1/Formality/dataset.csv --image_file clean_images/coco_1.jpg --output_file output/llava/coco_1/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_1/Formality/dataset.csv --image_file output/llava/coco_1/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_1/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_1/Formality/dataset.csv --image_file output/llava/coco_1/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_1/Formality/Informal/constrained_eps_32_batch_8/result.jsonl

# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_2/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_2.jpg --save_dir output/llava/coco_2/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_2/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_2.jpg --save_dir output/llava/coco_2/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_2/Formality/dataset.csv --image_file clean_images/coco_2.jpg --output_file output/llava/coco_2/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_2/Formality/dataset.csv --image_file clean_images/coco_2.jpg --output_file output/llava/coco_2/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_2/Formality/dataset.csv --image_file clean_images/coco_2.jpg --output_file output/llava/coco_2/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_2/Formality/dataset.csv --image_file output/llava/coco_2/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_2/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_2/Formality/dataset.csv --image_file output/llava/coco_2/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_2/Formality/Informal/constrained_eps_32_batch_8/result.jsonl

# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_3/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_3.jpg --save_dir output/llava/coco_3/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_3/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_3.jpg --save_dir output/llava/coco_3/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_3/Formality/dataset.csv --image_file clean_images/coco_3.jpg --output_file output/llava/coco_3/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_3/Formality/dataset.csv --image_file clean_images/coco_3.jpg --output_file output/llava/coco_3/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_3/Formality/dataset.csv --image_file clean_images/coco_3.jpg --output_file output/llava/coco_3/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_3/Formality/dataset.csv --image_file output/llava/coco_3/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_3/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_3/Formality/dataset.csv --image_file output/llava/coco_3/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_3/Formality/Informal/constrained_eps_32_batch_8/result.jsonl


# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_4/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_4.jpg --save_dir output/llava/coco_4/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_4/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_4.jpg --save_dir output/llava/coco_4/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_4/Formality/dataset.csv --image_file clean_images/coco_4.jpg --output_file output/llava/coco_4/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_4/Formality/dataset.csv --image_file clean_images/coco_4.jpg --output_file output/llava/coco_4/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_4/Formality/dataset.csv --image_file clean_images/coco_4.jpg --output_file output/llava/coco_4/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_4/Formality/dataset.csv --image_file output/llava/coco_4/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_4/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_4/Formality/dataset.csv --image_file output/llava/coco_4/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_4/Formality/Informal/constrained_eps_32_batch_8/result.jsonl

# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_5/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_5.jpg --save_dir output/llava/coco_5/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_5/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_5.jpg --save_dir output/llava/coco_5/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_5/Formality/dataset.csv --image_file clean_images/coco_5.jpg --output_file output/llava/coco_5/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_5/Formality/dataset.csv --image_file clean_images/coco_5.jpg --output_file output/llava/coco_5/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_5/Formality/dataset.csv --image_file clean_images/coco_5.jpg --output_file output/llava/coco_5/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_5/Formality/dataset.csv --image_file output/llava/coco_5/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_5/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_5/Formality/dataset.csv --image_file output/llava/coco_5/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_5/Formality/Informal/constrained_eps_32_batch_8/result.jsonl

# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_6/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_6.jpg --save_dir output/llava/coco_6/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_6/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_6.jpg --save_dir output/llava/coco_6/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_6/Formality/dataset.csv --image_file clean_images/coco_6.jpg --output_file output/llava/coco_6/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_6/Formality/dataset.csv --image_file clean_images/coco_6.jpg --output_file output/llava/coco_6/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_6/Formality/dataset.csv --image_file clean_images/coco_6.jpg --output_file output/llava/coco_6/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_6/Formality/dataset.csv --image_file output/llava/coco_6/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_6/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_6/Formality/dataset.csv --image_file output/llava/coco_6/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_6/Formality/Informal/constrained_eps_32_batch_8/result.jsonl

# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_7/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_7.jpg --save_dir output/llava/coco_7/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_7/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_7.jpg --save_dir output/llava/coco_7/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_7/Formality/dataset.csv --image_file clean_images/coco_7.jpg --output_file output/llava/coco_7/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_7/Formality/dataset.csv --image_file clean_images/coco_7.jpg --output_file output/llava/coco_7/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_7/Formality/dataset.csv --image_file clean_images/coco_7.jpg --output_file output/llava/coco_7/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_7/Formality/dataset.csv --image_file output/llava/coco_7/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_7/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_7/Formality/dataset.csv --image_file output/llava/coco_7/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_7/Formality/Informal/constrained_eps_32_batch_8/result.jsonl

# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_8/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_8.jpg --save_dir output/llava/coco_8/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_8/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_8.jpg --save_dir output/llava/coco_8/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_8/Formality/dataset.csv --image_file clean_images/coco_8.jpg --output_file output/llava/coco_8/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_8/Formality/dataset.csv --image_file clean_images/coco_8.jpg --output_file output/llava/coco_8/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_8/Formality/dataset.csv --image_file clean_images/coco_8.jpg --output_file output/llava/coco_8/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_8/Formality/dataset.csv --image_file output/llava/coco_8/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_8/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_8/Formality/dataset.csv --image_file output/llava/coco_8/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_8/Formality/Informal/constrained_eps_32_batch_8/result.jsonl

# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_9/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_9.jpg --save_dir output/llava/coco_9/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_9/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_9.jpg --save_dir output/llava/coco_9/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_9/Formality/dataset.csv --image_file clean_images/coco_9.jpg --output_file output/llava/coco_9/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_9/Formality/dataset.csv --image_file clean_images/coco_9.jpg --output_file output/llava/coco_9/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_9/Formality/dataset.csv --image_file clean_images/coco_9.jpg --output_file output/llava/coco_9/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_9/Formality/dataset.csv --image_file output/llava/coco_9/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_9/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_9/Formality/dataset.csv --image_file output/llava/coco_9/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_9/Formality/Informal/constrained_eps_32_batch_8/result.jsonl

# Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_10/Formality/dataset.csv --instruction formal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_10.jpg --save_dir output/llava/coco_10/Formality/Formal/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/coco_10/Formality/dataset.csv --instruction informal --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_10.jpg --save_dir output/llava/coco_10/Formality/Informal/constrained_eps_32_batch_8
#Baseline 1
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_10/Formality/dataset.csv --image_file clean_images/coco_10.jpg --output_file output/llava/coco_10/baseline_1/result.jsonl
#Baseline 2
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_10/Formality/dataset.csv --image_file clean_images/coco_10.jpg --output_file output/llava/coco_10/Formality/Formal/baseline_2/result.jsonl --instruction formal
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_10/Formality/dataset.csv --image_file clean_images/coco_10.jpg --output_file output/llava/coco_10/Formality/Informal/baseline_2/result.jsonl --instruction informal
# Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_10/Formality/dataset.csv --image_file output/llava/coco_10/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_10/Formality/Formal/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/coco_10/Formality/dataset.csv --image_file output/llava/coco_10/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/coco_10/Formality/Informal/constrained_eps_32_batch_8/result.jsonl

# evaluating the content
python -u llava_llama_v2_inference.py --image_file output/llava/coco_1/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 11 --output_file output/llava/coco_1/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_1/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 11 --output_file output/llava/coco_1/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_2/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 12 --output_file output/llava/coco_2/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_2/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 12 --output_file output/llava/coco_2/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_3/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 13 --output_file output/llava/coco_3/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_3/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 13 --output_file output/llava/coco_3/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_4/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 14 --output_file output/llava/coco_4/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_4/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 14 --output_file output/llava/coco_4/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_5/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 15 --output_file output/llava/coco_5/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl  --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_5/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 15 --output_file output/llava/coco_5/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_6/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 16 --output_file output/llava/coco_6/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_6/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 16 --output_file output/llava/coco_6/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_7/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 17 --output_file output/llava/coco_7/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_7/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 17 --output_file output/llava/coco_7/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_8/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 18 --output_file output/llava/coco_8/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_8/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 18 --output_file output/llava/coco_8/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_9/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 19 --output_file output/llava/coco_9/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_9/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 19 --output_file output/llava/coco_9/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_10/Formality/Formal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 20 --output_file output/llava/coco_10/Formality/Formal/constrained_eps_32_batch_8/content_classification_result.jsonl  --instruction inference_content_evaluation
python -u llava_llama_v2_inference.py --image_file output/llava/coco_10/Formality/Informal/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 20 --output_file output/llava/coco_10/Formality/Informal/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
