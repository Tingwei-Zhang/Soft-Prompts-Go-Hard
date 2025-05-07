#Image 1
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_1.jpg --save_dir output/minigpt4/coco_1/Sentiment/Positive/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_1.jpg --save_dir output/minigpt4/coco_1/Sentiment/Negative/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_1.jpg --save_dir output/minigpt4/coco_1/Sentiment/Neutral/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file clean_images/coco_1.jpg --output_file output/minigpt4/coco_1/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file clean_images/coco_1.jpg --output_file output/minigpt4/coco_1/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file clean_images/coco_1.jpg --output_file output/minigpt4/coco_1/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file clean_images/coco_1.jpg --output_file output/minigpt4/coco_1/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
#Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file output/minigpt4/coco_1/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_1/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file output/minigpt4/coco_1/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_1/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file output/minigpt4/coco_1/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_1/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

#Image 2
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_2/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_2.jpg --save_dir output/minigpt4/coco_2/Sentiment/Positive/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_2/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_2.jpg --save_dir output/minigpt4/coco_2/Sentiment/Negative/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_2/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_2.jpg --save_dir output/minigpt4/coco_2/Sentiment/Neutral/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Sentiment/dataset.csv --image_file clean_images/coco_2.jpg --output_file output/minigpt4/coco_2/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Sentiment/dataset.csv --image_file clean_images/coco_2.jpg --output_file output/minigpt4/coco_2/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Sentiment/dataset.csv --image_file clean_images/coco_2.jpg --output_file output/minigpt4/coco_2/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Sentiment/dataset.csv --image_file clean_images/coco_2.jpg --output_file output/minigpt4/coco_2/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
#Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Sentiment/dataset.csv --image_file output/minigpt4/coco_2/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_2/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Sentiment/dataset.csv --image_file output/minigpt4/coco_2/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_2/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Sentiment/dataset.csv --image_file output/minigpt4/coco_2/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_2/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

#Image 3
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_3/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_3.jpg --save_dir output/minigpt4/coco_3/Sentiment/Positive/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_3/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_3.jpg --save_dir output/minigpt4/coco_3/Sentiment/Negative/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_3/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_3.jpg --save_dir output/minigpt4/coco_3/Sentiment/Neutral/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Sentiment/dataset.csv --image_file clean_images/coco_3.jpg --output_file output/minigpt4/coco_3/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Sentiment/dataset.csv --image_file clean_images/coco_3.jpg --output_file output/minigpt4/coco_3/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Sentiment/dataset.csv --image_file clean_images/coco_3.jpg --output_file output/minigpt4/coco_3/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Sentiment/dataset.csv --image_file clean_images/coco_3.jpg --output_file output/minigpt4/coco_3/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
#Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Sentiment/dataset.csv --image_file output/minigpt4/coco_3/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_3/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Sentiment/dataset.csv --image_file output/minigpt4/coco_3/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_3/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Sentiment/dataset.csv --image_file output/minigpt4/coco_3/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_3/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

#Image 4
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_4/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_4.jpg --save_dir output/minigpt4/coco_4/Sentiment/Positive/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_4/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_4.jpg --save_dir output/minigpt4/coco_4/Sentiment/Negative/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_4/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_4.jpg --save_dir output/minigpt4/coco_4/Sentiment/Neutral/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Sentiment/dataset.csv --image_file clean_images/coco_4.jpg --output_file output/minigpt4/coco_4/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Sentiment/dataset.csv --image_file clean_images/coco_4.jpg --output_file output/minigpt4/coco_4/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Sentiment/dataset.csv --image_file clean_images/coco_4.jpg --output_file output/minigpt4/coco_4/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Sentiment/dataset.csv --image_file clean_images/coco_4.jpg --output_file output/minigpt4/coco_4/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
#Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Sentiment/dataset.csv --image_file output/minigpt4/coco_4/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_4/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Sentiment/dataset.csv --image_file output/minigpt4/coco_4/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_4/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Sentiment/dataset.csv --image_file output/minigpt4/coco_4/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_4/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

#Image 5
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_5/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_5.jpg --save_dir output/minigpt4/coco_6/Sentiment/Positive/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_5/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_5.jpg --save_dir output/minigpt4/coco_5/Sentiment/Negative/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_5/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_5.jpg --save_dir output/minigpt4/coco_5/Sentiment/Neutral/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Sentiment/dataset.csv --image_file clean_images/coco_5.jpg --output_file output/minigpt4/coco_5/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Sentiment/dataset.csv --image_file clean_images/coco_5.jpg --output_file output/minigpt4/coco_5/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Sentiment/dataset.csv --image_file clean_images/coco_5.jpg --output_file output/minigpt4/coco_5/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Sentiment/dataset.csv --image_file clean_images/coco_5.jpg --output_file output/minigpt4/coco_5/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Sentiment/dataset.csv --image_file output/minigpt4/coco_5/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_5/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Sentiment/dataset.csv --image_file output/minigpt4/coco_5/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_5/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Sentiment/dataset.csv --image_file output/minigpt4/coco_5/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_5/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl


#Image 6
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_6/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_6.jpg --save_dir output/minigpt4/coco_6/Sentiment/Positive/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_6/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_6.jpg --save_dir output/minigpt4/coco_6/Sentiment/Negative/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_6/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_6.jpg --save_dir output/minigpt4/coco_6/Sentiment/Neutral/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Sentiment/dataset.csv --image_file clean_images/coco_6.jpg --output_file output/minigpt4/coco_6/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Sentiment/dataset.csv --image_file clean_images/coco_6.jpg --output_file output/minigpt4/coco_6/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Sentiment/dataset.csv --image_file clean_images/coco_6.jpg --output_file output/minigpt4/coco_6/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Sentiment/dataset.csv --image_file clean_images/coco_6.jpg --output_file output/minigpt4/coco_6/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Sentiment/dataset.csv --image_file output/minigpt4/coco_6/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_6/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Sentiment/dataset.csv --image_file output/minigpt4/coco_6/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_6/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Sentiment/dataset.csv --image_file output/minigpt4/coco_6/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_6/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

#Image 7
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_7/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_7.jpg --save_dir output/minigpt4/coco_7/Sentiment/Positive/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_7/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_7.jpg --save_dir output/minigpt4/coco_7/Sentiment/Negative/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_7/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_7.jpg --save_dir output/minigpt4/coco_7/Sentiment/Neutral/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Sentiment/dataset.csv --image_file clean_images/coco_7.jpg --output_file output/minigpt4/coco_7/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Sentiment/dataset.csv --image_file clean_images/coco_7.jpg --output_file output/minigpt4/coco_7/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Sentiment/dataset.csv --image_file clean_images/coco_7.jpg --output_file output/minigpt4/coco_7/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Sentiment/dataset.csv --image_file clean_images/coco_7.jpg --output_file output/minigpt4/coco_7/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Sentiment/dataset.csv --image_file output/minigpt4/coco_7/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_7/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Sentiment/dataset.csv --image_file output/minigpt4/coco_7/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_7/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Sentiment/dataset.csv --image_file output/minigpt4/coco_7/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_7/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

#Image 8
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_8/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_8.jpg --save_dir output/minigpt4/coco_8/Sentiment/Positive/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_8/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_8.jpg --save_dir output/minigpt4/coco_8/Sentiment/Negative/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_8/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_8.jpg --save_dir output/minigpt4/coco_8/Sentiment/Neutral/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Sentiment/dataset.csv --image_file clean_images/coco_8.jpg --output_file output/minigpt4/coco_8/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Sentiment/dataset.csv --image_file clean_images/coco_8.jpg --output_file output/minigpt4/coco_8/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Sentiment/dataset.csv --image_file clean_images/coco_8.jpg --output_file output/minigpt4/coco_8/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Sentiment/dataset.csv --image_file clean_images/coco_8.jpg --output_file output/minigpt4/coco_8/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Sentiment/dataset.csv --image_file output/minigpt4/coco_8/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_8/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Sentiment/dataset.csv --image_file output/minigpt4/coco_8/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_8/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Sentiment/dataset.csv --image_file output/minigpt4/coco_8/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_8/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

#Image 9
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_9/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_9.jpg --save_dir output/minigpt4/coco_9/Sentiment/Positive/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_9/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_9.jpg --save_dir output/minigpt4/coco_9/Sentiment/Negative/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_9/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_9.jpg --save_dir output/minigpt4/coco_9/Sentiment/Neutral/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Sentiment/dataset.csv --image_file clean_images/coco_9.jpg --output_file output/minigpt4/coco_9/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Sentiment/dataset.csv --image_file clean_images/coco_9.jpg --output_file output/minigpt4/coco_9/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Sentiment/dataset.csv --image_file clean_images/coco_9.jpg --output_file output/minigpt4/coco_9/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Sentiment/dataset.csv --image_file clean_images/coco_9.jpg --output_file output/minigpt4/coco_9/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Sentiment/dataset.csv --image_file output/minigpt4/coco_9/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_9/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Sentiment/dataset.csv --image_file output/minigpt4/coco_9/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_9/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Sentiment/dataset.csv --image_file output/minigpt4/coco_9/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_9/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

#Image 10
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_10/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_10.jpg --save_dir output/minigpt4/coco_10/Sentiment/Positive/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_10/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_10.jpg --save_dir output/minigpt4/coco_10/Sentiment/Negative/constrained_eps_32_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/coco_10/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/coco_10.jpg --save_dir output/minigpt4/coco_10/Sentiment/Neutral/constrained_eps_32_batch_8
#Inference 
#Baseline 1
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Sentiment/dataset.csv --image_file clean_images/coco_10.jpg --output_file output/minigpt4/coco_10/baseline_1/result.jsonl
#Baseline 2
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Sentiment/dataset.csv --image_file clean_images/coco_10.jpg --output_file output/minigpt4/coco_10/Sentiment/Positive/baseline_2/result.jsonl --instruction positive
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Sentiment/dataset.csv --image_file clean_images/coco_10.jpg --output_file output/minigpt4/coco_10/Sentiment/Negative/baseline_2/result.jsonl --instruction negative
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Sentiment/dataset.csv --image_file clean_images/coco_10.jpg --output_file output/minigpt4/coco_10/Sentiment/Neutral/baseline_2/result.jsonl --instruction neutral
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Sentiment/dataset.csv --image_file output/minigpt4/coco_10/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_10/Sentiment/Neutral/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Sentiment/dataset.csv --image_file output/minigpt4/coco_10/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_10/Sentiment/Negative/constrained_eps_32_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Sentiment/dataset.csv --image_file output/minigpt4/coco_10/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_10/Sentiment/Positive/constrained_eps_32_batch_8/result.jsonl

# # Content perservation evaluation
python -u minigpt_inference.py --gpu_id 0 --image_file clean_images/coco_1.jpg --image_index 11 --output_file output/minigpt4/coco_1/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --image_file clean_images/coco_2.jpg --image_index 12 --output_file output/minigpt4/coco_2/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --image_file clean_images/coco_3.jpg --image_index 13 --output_file output/minigpt4/coco_3/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --image_file clean_images/coco_4.jpg --image_index 14 --output_file output/minigpt4/coco_4/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --image_file clean_images/coco_5.jpg --image_index 15 --output_file output/minigpt4/coco_5/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --image_file clean_images/coco_6.jpg --image_index 16 --output_file output/minigpt4/coco_6/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --image_file clean_images/coco_7.jpg --image_index 17 --output_file output/minigpt4/coco_7/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --image_file clean_images/coco_8.jpg --image_index 18 --output_file output/minigpt4/coco_8/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --image_file clean_images/coco_9.jpg --image_index 19 --output_file output/minigpt4/coco_9/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --image_file clean_images/coco_10.jpg --image_index 20 --output_file output/minigpt4/coco_10/baseline_1/content_classification_result.jsonl --instruction inference_content_evaluation

python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file output/minigpt4/coco_1/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 11 --output_file output/minigpt4/coco_1/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file output/minigpt4/coco_1/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 11 --output_file output/minigpt4/coco_1/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file output/minigpt4/coco_1/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 11 --output_file output/minigpt4/coco_1/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Sentiment/dataset.csv --image_file output/minigpt4/coco_2/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 12 --output_file output/minigpt4/coco_2/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Sentiment/dataset.csv --image_file output/minigpt4/coco_2/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 12 --output_file output/minigpt4/coco_2/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_2/Sentiment/dataset.csv --image_file output/minigpt4/coco_2/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 12 --output_file output/minigpt4/coco_2/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Sentiment/dataset.csv --image_file output/minigpt4/coco_3/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 13 --output_file output/minigpt4/coco_3/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Sentiment/dataset.csv --image_file output/minigpt4/coco_3/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 13 --output_file output/minigpt4/coco_3/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_3/Sentiment/dataset.csv --image_file output/minigpt4/coco_3/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 13 --output_file output/minigpt4/coco_3/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Sentiment/dataset.csv --image_file output/minigpt4/coco_4/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 14 --output_file output/minigpt4/coco_4/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Sentiment/dataset.csv --image_file output/minigpt4/coco_4/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 14 --output_file output/minigpt4/coco_4/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_4/Sentiment/dataset.csv --image_file output/minigpt4/coco_4/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 14 --output_file output/minigpt4/coco_4/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Sentiment/dataset.csv --image_file output/minigpt4/coco_5/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 15 --output_file output/minigpt4/coco_5/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Sentiment/dataset.csv --image_file output/minigpt4/coco_5/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 15 --output_file output/minigpt4/coco_5/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_5/Sentiment/dataset.csv --image_file output/minigpt4/coco_5/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 15 --output_file output/minigpt4/coco_5/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Sentiment/dataset.csv --image_file output/minigpt4/coco_6/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 16 --output_file output/minigpt4/coco_6/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Sentiment/dataset.csv --image_file output/minigpt4/coco_6/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 16 --output_file output/minigpt4/coco_6/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_6/Sentiment/dataset.csv --image_file output/minigpt4/coco_6/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 16 --output_file output/minigpt4/coco_6/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Sentiment/dataset.csv --image_file output/minigpt4/coco_7/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 17 --output_file output/minigpt4/coco_7/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Sentiment/dataset.csv --image_file output/minigpt4/coco_7/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 17 --output_file output/minigpt4/coco_7/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_7/Sentiment/dataset.csv --image_file output/minigpt4/coco_7/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 17 --output_file output/minigpt4/coco_7/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Sentiment/dataset.csv --image_file output/minigpt4/coco_8/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 18 --output_file output/minigpt4/coco_8/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Sentiment/dataset.csv --image_file output/minigpt4/coco_8/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 18 --output_file output/minigpt4/coco_8/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_8/Sentiment/dataset.csv --image_file output/minigpt4/coco_8/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 18 --output_file output/minigpt4/coco_8/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Sentiment/dataset.csv --image_file output/minigpt4/coco_9/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 19 --output_file output/minigpt4/coco_9/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Sentiment/dataset.csv --image_file output/minigpt4/coco_9/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 19 --output_file output/minigpt4/coco_9/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_9/Sentiment/dataset.csv --image_file output/minigpt4/coco_9/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 19 --output_file output/minigpt4/coco_9/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Sentiment/dataset.csv --image_file output/minigpt4/coco_10/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 20 --output_file output/minigpt4/coco_10/Sentiment/Neutral/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Sentiment/dataset.csv --image_file output/minigpt4/coco_10/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 20 --output_file output/minigpt4/coco_10/Sentiment/Negative/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_10/Sentiment/dataset.csv --image_file output/minigpt4/coco_10/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp  --image_index 20 --output_file output/minigpt4/coco_10/Sentiment/Positive/constrained_eps_32_batch_8/content_classification_result.jsonl --instruction inference_content_evaluation


