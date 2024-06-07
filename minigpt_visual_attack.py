# 1. Download Vicuna's weights to ./models   (it's a delta version)
# 2. Download LLaMA's weight via: https://huggingface.co/huggyllama/llama-13b/tree/main
# 3. merge them and setup config
# 4. Download the mini-gpt4 compoents' pretrained ckpts
# 5. vision part will be automatically download when launching the model


import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
from minigpt_utils import visual_attacker, prompt_wrapper

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry



def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--data_path", type=str, default="instruction_data/0/Sentiment/dataset.csv")
    parser.add_argument("--instruction", type=str, default='positive',
                    choices=[ "positive", "negative", "neutral", "irony", "non_irony", "formal", "informal", "french", "english", "spanish", "left", "right","pirate","harry_potter", "hemingway"],
                    help="Instruction to be used for the attack.")
    parser.add_argument("--n_iters", type=int, default=5000, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", type=str, default='constrained',choices=[ "unconstrained", "constrained", "partial","l2", "jpeg"])
    parser.add_argument("--image_file", type=str, default='./image.bmp',
                        help="Image file")
    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================


print('>>> Initializing Models')

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

print('[Initialization Finished]\n')

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir,exist_ok=True)

import csv
def read_csv_data(filepath):
    with open(filepath, 'r') as file:
        # Use DictReader to read the CSV file, automatically using the first row for column names
        reader = csv.DictReader(file)
        data = [row for row in reader]

    # Extract column names from the CSV
    column_names = reader.fieldnames

    # Create a dictionary to hold lists for each column
    columns = {name: [] for name in column_names}

    # Populate the lists with data from each column
    for row in data:
        for name in column_names:
            columns[name].append(row[name])

    # Convert the dictionary into separate lists
    lists = [columns[name] for name in column_names]
    return lists, column_names

# Use the function to read the CSV data
lists, column_names = read_csv_data(args.data_path)

if args.instruction=='french':
    args.instruction='fr'
if args.instruction=='english':
    args.instruction='en'
if args.instruction=='spanish':
    args.instruction='es'
if args.instruction=='right':
    args.instruction='Republican'
if args.instruction=='left':
    args.instruction='Democrat'
    
#find the index of the "positive" in column_names list
print(column_names)
instruction_index = column_names.index(args.instruction)

print(args.instruction)
print(column_names[instruction_index])
instructions=lists[0]
targets=lists[instruction_index]
instructions=instructions[:-20]
targets=targets[:-20]

my_attacker = visual_attacker.Attacker(args, model, targets, instructions, device=model.device, is_rtp=False)

img = Image.open(args.image_file).convert('RGB')
img = vis_processor(img).unsqueeze(0).to(model.device)

text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt_text_attack


if args.constrained=='unconstrained':
    adv_img_prompt = my_attacker.attack_unconstrained(text_prompt_template,
                                                            img=img, batch_size = 8,
                                                            num_iter=2000, alpha=args.alpha/255)
elif args.constrained=='partial':
    adv_img_prompt = my_attacker.attack_partial_constrained(text_prompt_template,
                                                            img=img, batch_size = 8,
                                                            num_iter=2000, alpha=args.alpha/255, rows_to_change=50)
elif args.constrained=='l2':
    adv_img_prompt = my_attacker.attack_l2(text_prompt_template,
                                                            img=img, batch_size = 8,
                                                            num_iter=2000, alpha=args.alpha,epsilon=args.eps)
elif args.constrained=='jpeg':
    adv_img_prompt = my_attacker.attack_jpeg(text_prompt_template,
                                                            img=img, batch_size = 8,
                                                            num_iter=2000, alpha=args.alpha,epsilon=args.eps/255)
else:
    print("constrained")
    adv_img_prompt = my_attacker.attack_constrained(text_prompt_template,
                                                            img=img, batch_size=  8,
                                                            num_iter=2000, alpha=args.alpha / 255,
                                                            epsilon=args.eps / 255)

save_image(adv_img_prompt, '%s/bad_prompt.bmp' % args.save_dir)
print('[Done]')
