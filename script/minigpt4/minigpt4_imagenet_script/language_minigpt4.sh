#Image 0
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/0/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/minigpt4/0/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/0/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/minigpt4/0/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/0/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/minigpt4/0/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Language/dataset.csv --image_file clean_images/0.png --output_file output/minigpt4/0/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Language/dataset.csv --image_file clean_images/0.png --output_file output/minigpt4/0/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Language/dataset.csv --image_file clean_images/0.png --output_file output/minigpt4/0/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Language/dataset.csv --image_file clean_images/0.png --output_file output/minigpt4/0/Language/Spanish/baseline_2/result.jsonl --instruction spanish
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Language/dataset.csv --image_file output/minigpt4/0/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/0/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Language/dataset.csv --image_file output/minigpt4/0/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/0/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Language/dataset.csv --image_file output/minigpt4/0/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/0/Language/English/constrained_eps_32_batch_8/result.jsonl

#Image 1
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/1/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/1.png --save_dir output/minigpt4/1/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/1/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/1.png --save_dir output/minigpt4/1/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/1/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/1.png --save_dir output/minigpt4/1/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/1/Language/dataset.csv --image_file clean_images/1.png --output_file output/minigpt4/1/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/1/Language/dataset.csv --image_file clean_images/1.png --output_file output/minigpt4/1/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/1/Language/dataset.csv --image_file clean_images/1.png --output_file output/minigpt4/1/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/1/Language/dataset.csv --image_file clean_images/1.png --output_file output/minigpt4/1/Language/Spanish/baseline_2/result.jsonl --instruction spanish
#Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/1/Language/dataset.csv --image_file output/minigpt4/1/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/1/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/1/Language/dataset.csv --image_file output/minigpt4/1/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/1/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/1/Language/dataset.csv --image_file output/minigpt4/1/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/1/Language/English/constrained_eps_32_batch_8/result.jsonl

#Image 2
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/2/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/2.png --save_dir output/minigpt4/2/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/2/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/2.png --save_dir output/minigpt4/2/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/2/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/2.png --save_dir output/minigpt4/2/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/2/Language/dataset.csv --image_file clean_images/2.png --output_file output/minigpt4/2/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/2/Language/dataset.csv --image_file clean_images/2.png --output_file output/minigpt4/2/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/2/Language/dataset.csv --image_file clean_images/2.png --output_file output/minigpt4/2/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/2/Language/dataset.csv --image_file clean_images/2.png --output_file output/minigpt4/2/Language/Spanish/baseline_2/result.jsonl --instruction spanish
#Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/2/Language/dataset.csv --image_file output/minigpt4/2/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/2/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/2/Language/dataset.csv --image_file output/minigpt4/2/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/2/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/2/Language/dataset.csv --image_file output/minigpt4/2/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/2/Language/English/constrained_eps_32_batch_8/result.jsonl

#Image 3
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/3/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/3.png --save_dir output/minigpt4/3/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/3/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/3.png --save_dir output/minigpt4/3/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/3/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/3.png --save_dir output/minigpt4/3/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/3/Language/dataset.csv --image_file clean_images/3.png --output_file output/minigpt4/3/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/3/Language/dataset.csv --image_file clean_images/3.png --output_file output/minigpt4/3/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/3/Language/dataset.csv --image_file clean_images/3.png --output_file output/minigpt4/3/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/3/Language/dataset.csv --image_file clean_images/3.png --output_file output/minigpt4/3/Language/Spanish/baseline_2/result.jsonl --instruction spanish
#Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/3/Language/dataset.csv --image_file output/minigpt4/3/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/3/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/3/Language/dataset.csv --image_file output/minigpt4/3/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/3/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/3/Language/dataset.csv --image_file output/minigpt4/3/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/3/Language/English/constrained_eps_32_batch_8/result.jsonl

#Image 4
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/4/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/4.png --save_dir output/minigpt4/4/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/4/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/4.png --save_dir output/minigpt4/4/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/4/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/4.png --save_dir output/minigpt4/4/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/4/Language/dataset.csv --image_file clean_images/4.png --output_file output/minigpt4/4/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/4/Language/dataset.csv --image_file clean_images/4.png --output_file output/minigpt4/4/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/4/Language/dataset.csv --image_file clean_images/4.png --output_file output/minigpt4/4/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/4/Language/dataset.csv --image_file clean_images/4.png --output_file output/minigpt4/4/Language/Spanish/baseline_2/result.jsonl --instruction spanish
#Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/4/Language/dataset.csv --image_file output/minigpt4/4/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/4/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/4/Language/dataset.csv --image_file output/minigpt4/4/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/4/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/4/Language/dataset.csv --image_file output/minigpt4/4/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/4/Language/English/constrained_eps_32_batch_8/result.jsonl


# Content perservation evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Language/dataset.csv --image_file output/minigpt4/0/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 0 --output_file output/minigpt4/0/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Language/dataset.csv --image_file output/minigpt4/0/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 0 --output_file output/minigpt4/0/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Language/dataset.csv --image_file output/minigpt4/0/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 0 --output_file output/minigpt4/0/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation

python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/1/Language/dataset.csv --image_file output/minigpt4/1/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 1 --output_file output/minigpt4/1/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/1/Language/dataset.csv --image_file output/minigpt4/1/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 1 --output_file output/minigpt4/1/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/1/Language/dataset.csv --image_file output/minigpt4/1/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 1 --output_file output/minigpt4/1/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation

python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/2/Language/dataset.csv --image_file output/minigpt4/2/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 2 --output_file output/minigpt4/2/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/2/Language/dataset.csv --image_file output/minigpt4/2/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 2 --output_file output/minigpt4/2/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/2/Language/dataset.csv --image_file output/minigpt4/2/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 2 --output_file output/minigpt4/2/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation

python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/3/Language/dataset.csv --image_file output/minigpt4/3/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 3 --output_file output/minigpt4/3/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/3/Language/dataset.csv --image_file output/minigpt4/3/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 3 --output_file output/minigpt4/3/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/3/Language/dataset.csv --image_file output/minigpt4/3/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 3 --output_file output/minigpt4/3/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation

python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/4/Language/dataset.csv --image_file output/minigpt4/4/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 4 --output_file output/minigpt4/4/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/4/Language/dataset.csv --image_file output/minigpt4/4/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 4 --output_file output/minigpt4/4/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/4/Language/dataset.csv --image_file output/minigpt4/4/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 4 --output_file output/minigpt4/4/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation

