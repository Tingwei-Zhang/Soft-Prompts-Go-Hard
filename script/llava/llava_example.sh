python llava_llama_v2_visual_attack.py --data_path instruction_data/eg1/Political/dataset.csv --instruction right --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/eg1.png --save_dir output/llava/eg1/Political/Right/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/eg1/Political/dataset.csv --instruction left --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/eg1.png --save_dir output/llava/eg1/Political/Left/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/eg2/Political/dataset.csv --instruction right --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/eg2.png --save_dir output/llava/eg2/Political/Right/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/eg2/Political/dataset.csv --instruction left --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/eg2.png --save_dir output/llava/eg2/Political/Left/constrained_eps_32_batch_8

python -u llava_llama_v2_inference.py --data_path instruction_data/eg1/Political/dataset.csv --image_file output/llava/eg1/Political/Left/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg1/Political/Left/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg1/Political/dataset.csv --image_file output/llava/eg1/Political/Right/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg1/Political/Right/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg2/Political/dataset.csv --image_file output/llava/eg2/Political/Left/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg2/Political/Left/constrained_eps_32_batch_8/result.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg2/Political/dataset.csv --image_file output/llava/eg2/Political/Right/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg2/Political/Right/constrained_eps_32_batch_8/result.jsonl


python llava_llama_v2_visual_attack.py --data_path instruction_data/eg1/Political/dataset.csv --instruction positive --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/eg7.png --save_dir output/llava/eg7/Positive/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/eg1/Political/dataset.csv --instruction negative --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/eg7.png --save_dir output/llava/eg7/Negative/constrained_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/eg1/Political/dataset.csv --instruction fix --n_iters 2000 --constrained constrained --eps 32 --alpha 1 --image_file clean_images/eg7.png --save_dir output/llava/eg7/Fix/constrained_eps_32_batch_8


#eg4
#Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/eg4/dataset.csv --instruction positive --n_iters 2000 --constrained partial --eps 128 --alpha 1 --image_file clean_images/eg4.png --save_dir output/llava/eg4/Sentiment/Positive/partial_eps_128_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/eg4/dataset.csv --instruction negative --n_iters 2000 --constrained partial --eps 128 --alpha 1 --image_file clean_images/eg4.png --save_dir output/llava/eg4/Sentiment/Negative/partial_eps_128_batch_8
#Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Negative/partial_eps_128_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Negative/partial_eps_128_batch_8/result1.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Negative/partial_eps_128_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Negative/partial_eps_128_batch_8/result2.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Negative/partial_eps_128_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Negative/partial_eps_128_batch_8/result3.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Negative/partial_eps_128_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Negative/partial_eps_128_batch_8/result4.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Negative/partial_eps_128_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Negative/partial_eps_128_batch_8/result5.jsonl

python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Positive/partial_eps_128_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Positive/partial_eps_128_batch_8/result1.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Positive/partial_eps_128_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Positive/partial_eps_128_batch_8/result2.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Positive/partial_eps_128_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Positive/partial_eps_128_batch_8/result3.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Positive/partial_eps_128_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Positive/partial_eps_128_batch_8/result4.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Positive/partial_eps_128_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Positive/partial_eps_128_batch_8/result5.jsonl

#Attack
python llava_llama_v2_visual_attack.py --data_path instruction_data/eg4/dataset.csv --instruction positive --n_iters 2000 --constrained partial --eps 32 --alpha 1 --image_file clean_images/eg4.png --save_dir output/llava/eg4/Sentiment/Positive/partial_eps_32_batch_8
python llava_llama_v2_visual_attack.py --data_path instruction_data/eg4/dataset.csv --instruction negative --n_iters 2000 --constrained partial --eps 32 --alpha 1 --image_file clean_images/eg4.png --save_dir output/llava/eg4/Sentiment/Negative/partial_eps_32_batch_8
#Our method
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Negative/partial_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Negative/partial_eps_32_batch_8/result1.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Negative/partial_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Negative/partial_eps_32_batch_8/result2.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Negative/partial_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Negative/partial_eps_32_batch_8/result3.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Negative/partial_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Negative/partial_eps_32_batch_8/result4.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Negative/partial_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Negative/partial_eps_32_batch_8/result5.jsonl

python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Positive/partial_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Positive/partial_eps_32_batch_8/result1.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Positive/partial_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Positive/partial_eps_32_batch_8/result2.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Positive/partial_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Positive/partial_eps_32_batch_8/result3.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Positive/partial_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Positive/partial_eps__batch_8/result4.jsonl
python -u llava_llama_v2_inference.py --data_path instruction_data/eg4/dataset.csv --image_file output/llava/eg4/Sentiment/Positive/partial_eps_32_batch_8/bad_prompt.bmp --output_file output/llava/eg4/Sentiment/Positive/partial_eps_32_batch_8/result5.jsonl
