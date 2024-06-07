#Image 0
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/0.png --save_dir output/minigpt4/0/Sentiment/Positive/constrained_eps_16_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/0.png --save_dir output/minigpt4/0/Sentiment/Negative/constrained_eps_16_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/0.png --save_dir output/minigpt4/0/Sentiment/Neutral/constrained_eps_16_batch_8
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --image_file output/minigpt4/0/Sentiment/Neutral/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/0/Sentiment/Neutral/constrained_eps_16_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --image_file output/minigpt4/0/Sentiment/Negative/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/0/Sentiment/Negative/constrained_eps_16_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/0/Sentiment/dataset.csv --image_file output/minigpt4/0/Sentiment/Positive/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/0/Sentiment/Positive/constrained_eps_16_batch_8/result.jsonl

#Image 1
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/1/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/1.png --save_dir output/minigpt4/1/Sentiment/Positive/constrained_eps_16_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/1/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/1.png --save_dir output/minigpt4/1/Sentiment/Negative/constrained_eps_16_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/1/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/1.png --save_dir output/minigpt4/1/Sentiment/Neutral/constrained_eps_16_batch_8
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/1/Sentiment/dataset.csv --image_file output/minigpt4/1/Sentiment/Neutral/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/1/Sentiment/Neutral/constrained_eps_16_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/1/Sentiment/dataset.csv --image_file output/minigpt4/1/Sentiment/Negative/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/1/Sentiment/Negative/constrained_eps_16_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/1/Sentiment/dataset.csv --image_file output/minigpt4/1/Sentiment/Positive/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/1/Sentiment/Positive/constrained_eps_16_batch_8/result.jsonl

#Image 2
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/2/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/2.png --save_dir output/minigpt4/2/Sentiment/Positive/constrained_eps_16_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/2/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/2.png --save_dir output/minigpt4/2/Sentiment/Negative/constrained_eps_16_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/2/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/2.png --save_dir output/minigpt4/2/Sentiment/Neutral/constrained_eps_16_batch_8
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/2/Sentiment/dataset.csv --image_file output/minigpt4/2/Sentiment/Neutral/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/2/Sentiment/Neutral/constrained_eps_16_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/2/Sentiment/dataset.csv --image_file output/minigpt4/2/Sentiment/Negative/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/2/Sentiment/Negative/constrained_eps_16_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/2/Sentiment/dataset.csv --image_file output/minigpt4/2/Sentiment/Positive/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/2/Sentiment/Positive/constrained_eps_16_batch_8/result.jsonl

#Image 3
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/3/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/3.png --save_dir output/minigpt4/3/Sentiment/Positive/constrained_eps_16_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/3/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/3.png --save_dir output/minigpt4/3/Sentiment/Negative/constrained_eps_16_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/3/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/3.png --save_dir output/minigpt4/3/Sentiment/Neutral/constrained_eps_16_batch_8
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/3/Sentiment/dataset.csv --image_file output/minigpt4/3/Sentiment/Neutral/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/3/Sentiment/Neutral/constrained_eps_16_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/3/Sentiment/dataset.csv --image_file output/minigpt4/3/Sentiment/Negative/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/3/Sentiment/Negative/constrained_eps_16_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/3/Sentiment/dataset.csv --image_file output/minigpt4/3/Sentiment/Positive/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/3/Sentiment/Positive/constrained_eps_16_batch_8/result.jsonl

#Image 4
#Attack
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/4/Sentiment/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/4.png --save_dir output/minigpt4/4/Sentiment/Positive/constrained_eps_16_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/4/Sentiment/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/4.png --save_dir output/minigpt4/4/Sentiment/Negative/constrained_eps_16_batch_8
python minigpt_visual_attack.py --gpu_id 0 --data_path instruction_data/4/Sentiment/dataset.csv --instruction neutral --n_iters 2000 --constrained constrained --eps 16 --alpha 1 --image_file clean_images/4.png --save_dir output/minigpt4/4/Sentiment/Neutral/constrained_eps_16_batch_8
# Our method
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/4/Sentiment/dataset.csv --image_file output/minigpt4/4/Sentiment/Neutral/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/4/Sentiment/Neutral/constrained_eps_16_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/4/Sentiment/dataset.csv --image_file output/minigpt4/4/Sentiment/Negative/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/4/Sentiment/Negative/constrained_eps_16_batch_8/result.jsonl
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/4/Sentiment/dataset.csv --image_file output/minigpt4/4/Sentiment/Positive/constrained_eps_16_batch_8/bad_prompt.bmp --output_file output/minigpt4/4/Sentiment/Positive/constrained_eps_16_batch_8/result.jsonl
