#Image 1
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_1/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_1.jpg --save_dir output/minigpt4/coco_1/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_1/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_1.jpg --save_dir output/minigpt4/coco_1/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_1/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_1.jpg --save_dir output/minigpt4/coco_1/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Language/dataset.csv --image_file clean_images/coco_1.jpg --output_file output/minigpt4/coco_1/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Language/dataset.csv --image_file clean_images/coco_1.jpg --output_file output/minigpt4/coco_1/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Language/dataset.csv --image_file clean_images/coco_1.jpg --output_file output/minigpt4/coco_1/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Language/dataset.csv --image_file clean_images/coco_1.jpg --output_file output/minigpt4/coco_1/Language/Spanish/baseline_2/result.jsonl --instruction spanish
#Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Language/dataset.csv --image_file output/minigpt4/coco_1/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_1/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Language/dataset.csv --image_file output/minigpt4/coco_1/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_1/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Language/dataset.csv --image_file output/minigpt4/coco_1/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_1/Language/English/constrained_eps_32_batch_8/result.jsonl

#Image 2
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_2/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_2.jpg --save_dir output/minigpt4/coco_2/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_2/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_2.jpg --save_dir output/minigpt4/coco_2/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_2/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_2.jpg --save_dir output/minigpt4/coco_2/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Language/dataset.csv --image_file clean_images/coco_2.jpg --output_file output/minigpt4/coco_2/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Language/dataset.csv --image_file clean_images/coco_2.jpg --output_file output/minigpt4/coco_2/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Language/dataset.csv --image_file clean_images/coco_2.jpg --output_file output/minigpt4/coco_2/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Language/dataset.csv --image_file clean_images/coco_2.jpg --output_file output/minigpt4/coco_2/Language/Spanish/baseline_2/result.jsonl --instruction spanish
#Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Language/dataset.csv --image_file output/minigpt4/coco_2/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_2/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Language/dataset.csv --image_file output/minigpt4/coco_2/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_2/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Language/dataset.csv --image_file output/minigpt4/coco_2/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_2/Language/English/constrained_eps_32_batch_8/result.jsonl

#Image 3
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_3/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_3.jpg --save_dir output/minigpt4/coco_3/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_3/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_3.jpg --save_dir output/minigpt4/coco_3/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_3/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_3.jpg --save_dir output/minigpt4/coco_3/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Language/dataset.csv --image_file clean_images/coco_3.jpg --output_file output/minigpt4/coco_3/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Language/dataset.csv --image_file clean_images/coco_3.jpg --output_file output/minigpt4/coco_3/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Language/dataset.csv --image_file clean_images/coco_3.jpg --output_file output/minigpt4/coco_3/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Language/dataset.csv --image_file clean_images/coco_3.jpg --output_file output/minigpt4/coco_3/Language/Spanish/baseline_2/result.jsonl --instruction spanish
#Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Language/dataset.csv --image_file output/minigpt4/coco_3/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_3/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Language/dataset.csv --image_file output/minigpt4/coco_3/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_3/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Language/dataset.csv --image_file output/minigpt4/coco_3/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_3/Language/English/constrained_eps_32_batch_8/result.jsonl

#Image 4
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_4/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_4.jpg --save_dir output/minigpt4/coco_4/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_4/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_4.jpg --save_dir output/minigpt4/coco_4/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_4/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_4.jpg --save_dir output/minigpt4/coco_4/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Language/dataset.csv --image_file clean_images/coco_4.jpg --output_file output/minigpt4/coco_4/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Language/dataset.csv --image_file clean_images/coco_4.jpg --output_file output/minigpt4/coco_4/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Language/dataset.csv --image_file clean_images/coco_4.jpg --output_file output/minigpt4/coco_4/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Language/dataset.csv --image_file clean_images/coco_4.jpg --output_file output/minigpt4/coco_4/Language/Spanish/baseline_2/result.jsonl --instruction spanish
#Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Language/dataset.csv --image_file output/minigpt4/coco_4/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_4/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Language/dataset.csv --image_file output/minigpt4/coco_4/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_4/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Language/dataset.csv --image_file output/minigpt4/coco_4/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_4/Language/English/constrained_eps_32_batch_8/result.jsonl

#Image 5
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_5/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_5.jpg --save_dir output/minigpt4/coco_5/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_5/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_5.jpg --save_dir output/minigpt4/coco_5/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_5/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_5.jpg --save_dir output/minigpt4/coco_5/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Language/dataset.csv --image_file clean_images/coco_5.jpg --output_file output/minigpt4/coco_5/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Language/dataset.csv --image_file clean_images/coco_5.jpg --output_file output/minigpt4/coco_5/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Language/dataset.csv --image_file clean_images/coco_5.jpg --output_file output/minigpt4/coco_5/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Language/dataset.csv --image_file clean_images/coco_5.jpg --output_file output/minigpt4/coco_5/Language/Spanish/baseline_2/result.jsonl --instruction spanish
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Language/dataset.csv --image_file output/minigpt4/coco_5/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_5/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Language/dataset.csv --image_file output/minigpt4/coco_5/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_5/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Language/dataset.csv --image_file output/minigpt4/coco_5/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_5/Language/English/constrained_eps_32_batch_8/result.jsonl

#Image 6
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_6/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_6.jpg --save_dir output/minigpt4/coco_6/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_6/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_6.jpg --save_dir output/minigpt4/coco_6/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_6/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_6.jpg --save_dir output/minigpt4/coco_6/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Language/dataset.csv --image_file clean_images/coco_6.jpg --output_file output/minigpt4/coco_6/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Language/dataset.csv --image_file clean_images/coco_6.jpg --output_file output/minigpt4/coco_6/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Language/dataset.csv --image_file clean_images/coco_6.jpg --output_file output/minigpt4/coco_6/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Language/dataset.csv --image_file clean_images/coco_6.jpg --output_file output/minigpt4/coco_6/Language/Spanish/baseline_2/result.jsonl --instruction spanish
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Language/dataset.csv --image_file output/minigpt4/coco_6/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_6/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Language/dataset.csv --image_file output/minigpt4/coco_6/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_6/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Language/dataset.csv --image_file output/minigpt4/coco_6/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_6/Language/English/constrained_eps_32_batch_8/result.jsonl

#Image 7
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_7/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_7.jpg --save_dir output/minigpt4/coco_7/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_7/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_7.jpg --save_dir output/minigpt4/coco_7/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_7/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_7.jpg --save_dir output/minigpt4/coco_7/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Language/dataset.csv --image_file clean_images/coco_7.jpg --output_file output/minigpt4/coco_7/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Language/dataset.csv --image_file clean_images/coco_7.jpg --output_file output/minigpt4/coco_7/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Language/dataset.csv --image_file clean_images/coco_7.jpg --output_file output/minigpt4/coco_7/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Language/dataset.csv --image_file clean_images/coco_7.jpg --output_file output/minigpt4/coco_7/Language/Spanish/baseline_2/result.jsonl --instruction spanish
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Language/dataset.csv --image_file output/minigpt4/coco_7/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_7/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Language/dataset.csv --image_file output/minigpt4/coco_7/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_7/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Language/dataset.csv --image_file output/minigpt4/coco_7/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_7/Language/English/constrained_eps_32_batch_8/result.jsonl

#Image 8
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_8/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_8.jpg --save_dir output/minigpt4/coco_8/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_8/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_8.jpg --save_dir output/minigpt4/coco_8/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_8/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_8.jpg --save_dir output/minigpt4/coco_8/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Language/dataset.csv --image_file clean_images/coco_8.jpg --output_file output/minigpt4/coco_8/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Language/dataset.csv --image_file clean_images/coco_8.jpg --output_file output/minigpt4/coco_8/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Language/dataset.csv --image_file clean_images/coco_8.jpg --output_file output/minigpt4/coco_8/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Language/dataset.csv --image_file clean_images/coco_8.jpg --output_file output/minigpt4/coco_8/Language/Spanish/baseline_2/result.jsonl --instruction spanish
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Language/dataset.csv --image_file output/minigpt4/coco_8/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_8/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Language/dataset.csv --image_file output/minigpt4/coco_8/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_8/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Language/dataset.csv --image_file output/minigpt4/coco_8/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_8/Language/English/constrained_eps_32_batch_8/result.jsonl

#Image 9
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_9/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_9.jpg --save_dir output/minigpt4/coco_9/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_9/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_9.jpg --save_dir output/minigpt4/coco_9/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_9/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_9.jpg --save_dir output/minigpt4/coco_9/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Language/dataset.csv --image_file clean_images/coco_9.jpg --output_file output/minigpt4/coco_9/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Language/dataset.csv --image_file clean_images/coco_9.jpg --output_file output/minigpt4/coco_9/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Language/dataset.csv --image_file clean_images/coco_9.jpg --output_file output/minigpt4/coco_9/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Language/dataset.csv --image_file clean_images/coco_9.jpg --output_file output/minigpt4/coco_9/Language/Spanish/baseline_2/result.jsonl --instruction spanish
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Language/dataset.csv --image_file output/minigpt4/coco_9/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_9/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Language/dataset.csv --image_file output/minigpt4/coco_9/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_9/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Language/dataset.csv --image_file output/minigpt4/coco_9/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_9/Language/English/constrained_eps_32_batch_8/result.jsonl

#Image 10
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_10/Language/dataset.csv --instruction english --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_10.jpg --save_dir output/minigpt4/coco_10/Language/English/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_10/Language/dataset.csv --instruction french --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_10.jpg --save_dir output/minigpt4/coco_10/Language/French/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_10/Language/dataset.csv --instruction spanish --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_10.jpg --save_dir output/minigpt4/coco_10/Language/Spanish/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Language/dataset.csv --image_file clean_images/coco_10.jpg --output_file output/minigpt4/coco_10/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Language/dataset.csv --image_file clean_images/coco_10.jpg --output_file output/minigpt4/coco_10/Language/English/baseline_2/result.jsonl --instruction english
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Language/dataset.csv --image_file clean_images/coco_10.jpg --output_file output/minigpt4/coco_10/Language/French/baseline_2/result.jsonl --instruction french
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Language/dataset.csv --image_file clean_images/coco_10.jpg --output_file output/minigpt4/coco_10/Language/Spanish/baseline_2/result.jsonl --instruction spanish
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Language/dataset.csv --image_file output/minigpt4/coco_10/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_10/Language/Spanish/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Language/dataset.csv --image_file output/minigpt4/coco_10/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_10/Language/French/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Language/dataset.csv --image_file output/minigpt4/coco_10/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_10/Language/English/constrained_eps_32_batch_8/result.jsonl

# Content perservation evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Language/dataset.csv --image_file output/minigpt4/coco_1/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 11 --output_file output/minigpt4/coco_1/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Language/dataset.csv --image_file output/minigpt4/coco_1/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 11 --output_file output/minigpt4/coco_1/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Language/dataset.csv --image_file output/minigpt4/coco_1/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 11 --output_file output/minigpt4/coco_1/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Language/dataset.csv --image_file output/minigpt4/coco_2/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 12 --output_file output/minigpt4/coco_2/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Language/dataset.csv --image_file output/minigpt4/coco_2/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 12 --output_file output/minigpt4/coco_2/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Language/dataset.csv --image_file output/minigpt4/coco_2/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 12 --output_file output/minigpt4/coco_2/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Language/dataset.csv --image_file output/minigpt4/coco_3/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 13 --output_file output/minigpt4/coco_3/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Language/dataset.csv --image_file output/minigpt4/coco_3/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 13 --output_file output/minigpt4/coco_3/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Language/dataset.csv --image_file output/minigpt4/coco_3/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 13 --output_file output/minigpt4/coco_3/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Language/dataset.csv --image_file output/minigpt4/coco_4/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 14 --output_file output/minigpt4/coco_4/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Language/dataset.csv --image_file output/minigpt4/coco_4/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 14 --output_file output/minigpt4/coco_4/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Language/dataset.csv --image_file output/minigpt4/coco_4/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 14 --output_file output/minigpt4/coco_4/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Language/dataset.csv --image_file output/minigpt4/coco_5/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 15 --output_file output/minigpt4/coco_5/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Language/dataset.csv --image_file output/minigpt4/coco_5/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 15 --output_file output/minigpt4/coco_5/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Language/dataset.csv --image_file output/minigpt4/coco_5/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 15 --output_file output/minigpt4/coco_5/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Language/dataset.csv --image_file output/minigpt4/coco_6/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 16 --output_file output/minigpt4/coco_6/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Language/dataset.csv --image_file output/minigpt4/coco_6/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 16 --output_file output/minigpt4/coco_6/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Language/dataset.csv --image_file output/minigpt4/coco_6/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 16 --output_file output/minigpt4/coco_6/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Language/dataset.csv --image_file output/minigpt4/coco_7/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 17 --output_file output/minigpt4/coco_7/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Language/dataset.csv --image_file output/minigpt4/coco_7/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 17 --output_file output/minigpt4/coco_7/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Language/dataset.csv --image_file output/minigpt4/coco_7/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 17 --output_file output/minigpt4/coco_7/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Language/dataset.csv --image_file output/minigpt4/coco_8/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 18 --output_file output/minigpt4/coco_8/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Language/dataset.csv --image_file output/minigpt4/coco_8/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 18 --output_file output/minigpt4/coco_8/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Language/dataset.csv --image_file output/minigpt4/coco_8/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 18 --output_file output/minigpt4/coco_8/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Language/dataset.csv --image_file output/minigpt4/coco_9/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 19 --output_file output/minigpt4/coco_9/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Language/dataset.csv --image_file output/minigpt4/coco_9/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 19 --output_file output/minigpt4/coco_9/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Language/dataset.csv --image_file output/minigpt4/coco_9/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 19 --output_file output/minigpt4/coco_9/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Language/dataset.csv --image_file output/minigpt4/coco_10/Language/Spanish/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 20 --output_file output/minigpt4/coco_10/Language/Spanish/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Language/dataset.csv --image_file output/minigpt4/coco_10/Language/French/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 20 --output_file output/minigpt4/coco_10/Language/French/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Language/dataset.csv --image_file output/minigpt4/coco_10/Language/English/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 20 --output_file output/minigpt4/coco_10/Language/English/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation


