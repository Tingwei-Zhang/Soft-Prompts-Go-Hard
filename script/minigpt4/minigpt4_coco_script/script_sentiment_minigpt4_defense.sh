# JPEG Defense
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file output/minigpt4/coco_1/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_1/Sentiment/Neutral/constrained_eps_32_batch_8/jpeg_result.jsonl --defense jpeg
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file output/minigpt4/coco_1/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_1/Sentiment/Negative/constrained_eps_32_batch_8/jpeg_result.jsonl --defense jpeg
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file output/minigpt4/coco_1/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_1/Sentiment/Positive/constrained_eps_32_batch_8/jpeg_result.jsonl --defense jpeg
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv

# Gaussian Blur Defense
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file output/minigpt4/coco_1/Sentiment/Neutral/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_1/Sentiment/Neutral/constrained_eps_32_batch_8/gaussian_result.jsonl --defense gaussian
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file output/minigpt4/coco_1/Sentiment/Negative/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_1/Sentiment/Negative/constrained_eps_32_batch_8/gaussian_result.jsonl --defense gaussian
python -u minigpt_inference.py --gpu_id 0 --data_path instruction_data/coco_1/Sentiment/dataset.csv --image_file output/minigpt4/coco_1/Sentiment/Positive/constrained_eps_32_batch_8/bad_prompt.bmp --output_file output/minigpt4/coco_1/Sentiment/Positive/constrained_eps_32_batch_8/gaussian_result.jsonl --defense gaussian


# replace coco_1 with [coco_2, ..., coco_10]