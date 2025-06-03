import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
import csv
from lavis.models import load_model_and_preprocess
from blip_utils import visual_attacker


def parse_args():

    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--data_path", type=str, default="instruction_data/0/Sentiment/dataset.csv")
    parser.add_argument("--instruction", type=str, default='positive',
                    choices=[ "positive", "negative", "neutral", "irony", "non_irony", "formal", "informal", "french", "english", "spanish", "left", "right",
                             "negative_injection","positive_injection","positive_spam","negative_spam","injection","spam"],
                    help="Instruction to be used for the attack.")
    parser.add_argument("--n_iters", type=int, default=5000, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", type=str, default='constrained',choices=[ "unconstrained", "constrained", "partial"])
    parser.add_argument("--image_file", type=str, default='./image.bmp',
                        help="Image file")
    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")

    args = parser.parse_args()
    return args

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

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

# ========================================
#             Model Initialization
# ========================================


print('>>> Initializing Models')

args = parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# remember to modify the parameter llm_model in ./lavis/configs/models/blip2/blip2_instruct_vicuna13b.yaml to the path that store the vicuna weights
model, vis_processor, _ = load_model_and_preprocess(
        name='blip2_vicuna_instruct',
        model_type='vicuna13b',
        is_eval=True,
        device=device,
    )
model.eval()
"""
Source code of the model in:
    ./lavis/models/blip2_models/blip2_vicuna_instruct.py
"""

print('[Initialization Finished]\n')

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)

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

instruction_index = column_names.index(args.instruction)

print(args.instruction)
print(column_names[instruction_index])
instructions=lists[0]
targets=lists[instruction_index]
instructions=instructions[:40]
targets=targets[:40]

my_attacker = visual_attacker.Attacker(args, model, targets, instructions, device=model.device, is_rtp=False)

img = Image.open(args.image_file).convert('RGB')
img = vis_processor["eval"](img).unsqueeze(0).to(device)

if not args.constrained:

    adv_img_prompt = my_attacker.attack_unconstrained(img=img, batch_size = 8,
                                                            num_iter=args.n_iters, alpha=args.alpha/255)

else:
    adv_img_prompt = my_attacker.attack_constrained(img=img, batch_size= 8,
                                                            num_iter=args.n_iters, alpha=args.alpha / 255,
                                                            epsilon=args.eps / 255)

save_image(adv_img_prompt, '%s/bad_prompt.bmp' % args.save_dir)
print('[Done]')