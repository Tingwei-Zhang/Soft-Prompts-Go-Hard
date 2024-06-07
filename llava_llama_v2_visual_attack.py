import argparse
import torch
import os
from torchvision.utils import save_image
import csv

from PIL import Image

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_path", type=str, default="ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--data_path", type=str, default="instruction_data/0/Sentiment/dataset.csv")
    parser.add_argument("--instruction", type=str, default='positive',
                    choices=[ "positive", "negative", "neutral", "irony", "non_irony", "formal", "informal", "french", "english", "spanish", "left", "right"],
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

from llava_llama_2.utils import get_model
args = parse_args()

print('model = ', args.model_path)

tokenizer, model, image_processor, model_name = get_model(args)
model.eval()
# model.to('cuda:{}'.format(args.gpu_id))
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



#find the index of the "positive" in column_names list
instruction_index = column_names.index(args.instruction)

print(args.instruction)
print(column_names[instruction_index])
instructions=lists[0]
targets=lists[instruction_index]
instructions=instructions[:-20]
# Adding the text to each element in the list
print('>>> Instructions:', instructions[0])
targets=targets[:-20]
# print('>>> Instructions:', instructions)
# print('>>> Targets:', targets)

from llava_llama_2_utils import visual_attacker

print('device = ', model.device)
my_attacker = visual_attacker.Attacker(args, model, tokenizer, targets, instructions, device=model.device, image_processor=image_processor)

image = load_image(args.image_file)
image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
print(image.shape)

from llava_llama_2_utils import prompt_wrapper
text_prompt_template = prompt_wrapper.prepare_text_prompt('')
print(text_prompt_template)

if args.constrained=='unconstrained':
    print('[unconstrained]')
    adv_img_prompt = my_attacker.attack_unconstrained(text_prompt_template,
                                                            img=image, batch_size = 8,
                                                            num_iter=args.n_iters, alpha=args.alpha/255)
elif args.constrained=='partial':
    adv_img_prompt = my_attacker.attack_partial_constrained(text_prompt_template,
                                                            img=image, batch_size= 8,
                                                            num_iter=args.n_iters, alpha=args.alpha / 255,
                                                            epsilon=args.eps / 255, rows_to_change=10)
else:
    adv_img_prompt = my_attacker.attack_constrained(text_prompt_template,
                                                            img=image, batch_size= 4,
                                                            num_iter=args.n_iters, alpha=args.alpha / 255,
                                                            epsilon=args.eps / 255)


save_image(adv_img_prompt, '%s/bad_prompt.bmp' % args.save_dir)
print('[Done]')

