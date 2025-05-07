# Attack
python instructblip_visual_attack.py --data_path instruction_data/0/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/blip/0/Language/English/constrained_eps_32_batch_8
python instructblip_visual_attack.py --data_path instruction_data/0/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/blip/0/Language/French/constrained_eps_32_batch_8
python instructblip_visual_attack.py --data_path instruction_data/0/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/0.png --save_dir output/blip/0/Language/Spanish/constrained_eps_32_batch_8
#Baseline 1
python -u instructblip_inference.py --data_path instruction_data/0/Language/dataset.csv --image_file clean_images/0.png --output_file output/blip/0/baseline_1/result.jsonl
#Baseline 2
python -u instructblip_inference.py --data_path instruction_data/0/Language/dataset.csv --image_file clean_images/0.png --output_file output/blip/0/Language/English/baseline_2/result.jsonl --instruction english
python -u instructblip_inference.py --data_path instruction_data/0/Language/dataset.csv --image_file clean_images/0.png --output_file output/blip/0/Language/French/baseline_2/result.jsonl --instruction french
python -u instructblip_inference.py --data_path instruction_data/0/Language/dataset.csv --image_file clean_images/0.png --output_file output/blip/0/Language/Spanish/baseline_2/result.jsonl --instruction spanish
# Our method
python -u instructblip_inference.py --data_path instruction_data/0/Language/dataset.csv --image_file output/blip/0/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/0/Language/English/constrained_eps_32_batch_8/result.jsonl
python -u instructblip_inference.py --data_path instruction_data/0/Language/dataset.csv --image_file output/blip/0/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/0/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u instructblip_inference.py --data_path instruction_data/0/Language/dataset.csv --image_file output/blip/0/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/0/Language/Spanish/constrained_eps_32_batch_8/result.jsonl

# Attack
python instructblip_visual_attack.py --data_path instruction_data/1/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/1.png --save_dir output/blip/1/Language/English/constrained_eps_32_batch_8
python instructblip_visual_attack.py --data_path instruction_data/1/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/1.png --save_dir output/blip/1/Language/French/constrained_eps_32_batch_8
python instructblip_visual_attack.py --data_path instruction_data/1/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/1.png --save_dir output/blip/1/Language/Spanish/constrained_eps_32_batch_8
#Baseline 1
python -u instructblip_inference.py --data_path instruction_data/1/Language/dataset.csv --image_file clean_images/1.png --output_file output/blip/1/baseline_1/result.jsonl
#Baseline 2
python -u instructblip_inference.py --data_path instruction_data/1/Language/dataset.csv --image_file clean_images/1.png --output_file output/blip/1/Language/English/baseline_2/result.jsonl --instruction english
python -u instructblip_inference.py --data_path instruction_data/1/Language/dataset.csv --image_file clean_images/1.png --output_file output/blip/1/Language/French/baseline_2/result.jsonl --instruction french
python -u instructblip_inference.py --data_path instruction_data/1/Language/dataset.csv --image_file clean_images/1.png --output_file output/blip/1/Language/Spanish/baseline_2/result.jsonl --instruction spanish
# Our method
python -u instructblip_inference.py --data_path instruction_data/1/Language/dataset.csv --image_file output/blip/1/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/1/Language/English/constrained_eps_32_batch_8/result.jsonl
python -u instructblip_inference.py --data_path instruction_data/1/Language/dataset.csv --image_file output/blip/1/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/1/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u instructblip_inference.py --data_path instruction_data/1/Language/dataset.csv --image_file output/blip/1/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/1/Language/Spanish/constrained_eps_32_batch_8/result.jsonl

# Attack
python instructblip_visual_attack.py --data_path instruction_data/2/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/2.png --save_dir output/blip/2/Language/English/constrained_eps_32_batch_8
python instructblip_visual_attack.py --data_path instruction_data/2/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/2.png --save_dir output/blip/2/Language/French/constrained_eps_32_batch_8
python instructblip_visual_attack.py --data_path instruction_data/2/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/2.png --save_dir output/blip/2/Language/Spanish/constrained_eps_32_batch_8
#Baseline 1
python -u instructblip_inference.py --data_path instruction_data/2/Language/dataset.csv --image_file clean_images/2.png --output_file output/blip/2/baseline_1/result.jsonl
#Baseline 2
python -u instructblip_inference.py --data_path instruction_data/2/Language/dataset.csv --image_file clean_images/2.png --output_file output/blip/2/Language/English/baseline_2/result.jsonl --instruction english
python -u instructblip_inference.py --data_path instruction_data/2/Language/dataset.csv --image_file clean_images/2.png --output_file output/blip/2/Language/French/baseline_2/result.jsonl --instruction french
python -u instructblip_inference.py --data_path instruction_data/2/Language/dataset.csv --image_file clean_images/2.png --output_file output/blip/2/Language/Spanish/baseline_2/result.jsonl --instruction spanish
# Our method
python -u instructblip_inference.py --data_path instruction_data/2/Language/dataset.csv --image_file output/blip/2/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/2/Language/English/constrained_eps_32_batch_8/result.jsonl
python -u instructblip_inference.py --data_path instruction_data/2/Language/dataset.csv --image_file output/blip/2/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/2/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u instructblip_inference.py --data_path instruction_data/2/Language/dataset.csv --image_file output/blip/2/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/2/Language/Spanish/constrained_eps_32_batch_8/result.jsonl

# Attack
python instructblip_visual_attack.py --data_path instruction_data/3/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/3.png --save_dir output/blip/3/Language/English/constrained_eps_32_batch_8
python instructblip_visual_attack.py --data_path instruction_data/3/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/3.png --save_dir output/blip/3/Language/French/constrained_eps_32_batch_8
python instructblip_visual_attack.py --data_path instruction_data/3/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/3.png --save_dir output/blip/3/Language/Spanish/constrained_eps_32_batch_8
#Baseline 1
python -u instructblip_inference.py --data_path instruction_data/3/Language/dataset.csv --image_file clean_images/3.png --output_file output/blip/3/baseline_1/result.jsonl
#Baseline 2
python -u instructblip_inference.py --data_path instruction_data/3/Language/dataset.csv --image_file clean_images/3.png --output_file output/blip/3/Language/English/baseline_2/result.jsonl --instruction english
python -u instructblip_inference.py --data_path instruction_data/3/Language/dataset.csv --image_file clean_images/3.png --output_file output/blip/3/Language/French/baseline_2/result.jsonl --instruction french
python -u instructblip_inference.py --data_path instruction_data/3/Language/dataset.csv --image_file clean_images/3.png --output_file output/blip/3/Language/Spanish/baseline_2/result.jsonl --instruction spanish
# Our method
python -u instructblip_inference.py --data_path instruction_data/3/Language/dataset.csv --image_file output/blip/3/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/3/Language/English/constrained_eps_32_batch_8/result.jsonl
python -u instructblip_inference.py --data_path instruction_data/3/Language/dataset.csv --image_file output/blip/3/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/3/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u instructblip_inference.py --data_path instruction_data/3/Language/dataset.csv --image_file output/blip/3/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/3/Language/Spanish/constrained_eps_32_batch_8/result.jsonl

# Attack
python instructblip_visual_attack.py --data_path instruction_data/4/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/4.png --save_dir output/blip/4/Language/English/constrained_eps_32_batch_8
python instructblip_visual_attack.py --data_path instruction_data/4/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/4.png --save_dir output/blip/4/Language/French/constrained_eps_32_batch_8
python instructblip_visual_attack.py --data_path instruction_data/4/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/4.png --save_dir output/blip/4/Language/Spanish/constrained_eps_32_batch_8
#Baseline 1
python -u instructblip_inference.py --data_path instruction_data/4/Language/dataset.csv --image_file clean_images/4.png --output_file output/blip/4/baseline_1/result.jsonl
#Baseline 2
python -u instructblip_inference.py --data_path instruction_data/4/Language/dataset.csv --image_file clean_images/4.png --output_file output/blip/4/Language/English/baseline_2/result.jsonl --instruction english
python -u instructblip_inference.py --data_path instruction_data/4/Language/dataset.csv --image_file clean_images/4.png --output_file output/blip/4/Language/French/baseline_2/result.jsonl --instruction french
python -u instructblip_inference.py --data_path instruction_data/4/Language/dataset.csv --image_file clean_images/4.png --output_file output/blip/4/Language/Spanish/baseline_2/result.jsonl --instruction spanish
# Our method
python -u instructblip_inference.py --data_path instruction_data/4/Language/dataset.csv --image_file output/blip/4/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/4/Language/English/constrained_eps_32_batch_8/result.jsonl
python -u instructblip_inference.py --data_path instruction_data/4/Language/dataset.csv --image_file output/blip/4/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/4/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u instructblip_inference.py --data_path instruction_data/4/Language/dataset.csv --image_file output/blip/4/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/blip/4/Language/Spanish/constrained_eps_32_batch_8/result.jsonl

#evaluating the content
python -u instructblip_inference.py --image_file output/blip/0/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 0 --output_file output/blip/0/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl  --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/0/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 0 --output_file output/blip/0/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/0/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 0 --output_file output/blip/0/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/1/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 1 --output_file output/blip/1/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/1/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 1 --output_file output/blip/1/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/1/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 1 --output_file output/blip/1/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/2/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 2 --output_file output/blip/2/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/2/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 2 --output_file output/blip/2/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/2/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 2 --output_file output/blip/2/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/3/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 3 --output_file output/blip/3/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/3/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 3 --output_file output/blip/3/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/3/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 3 --output_file output/blip/3/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/4/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 4 --output_file output/blip/4/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/4/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 4 --output_file output/blip/4/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u instructblip_inference.py --image_file output/blip/4/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --image_index 4 --output_file output/blip/4/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
