import argparse, os, torch
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
                    choices=["positive", "negative", "neutral", "irony", "non_irony", "formal", "informal", 
                            "french", "english", "spanish", "left", "right", "negative_injection",
                            "positive_injection", "positive_spam", "negative_spam", "injection", "spam"])
    parser.add_argument("--n_iters", type=int, default=5000, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", type=str, default='constrained', choices=["unconstrained", "constrained", "partial"])
    parser.add_argument("--image_file", type=str, default='./image.bmp', help="Image file")
    parser.add_argument("--save_dir", type=str, default='output', help="save directory")
    return parser.parse_args()

def read_csv_data(filepath):
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
        column_names = reader.fieldnames
        columns = {name: [row[name] for row in data] for name in column_names}
        return [columns[name] for name in column_names], column_names

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model, vis_processor, _ = load_model_and_preprocess(
        name='blip2_vicuna_instruct', model_type='vicuna13b',
        is_eval=True, device=device
    )
    model.eval()
    
    # Setup directories and data
    os.makedirs(args.save_dir, exist_ok=True)
    lists, column_names = read_csv_data(args.data_path)
    
    # Map instruction names
    instruction_map = {'french': 'fr', 'english': 'en', 'spanish': 'es', 
                      'right': 'Republican', 'left': 'Democrat'}
    args.instruction = instruction_map.get(args.instruction, args.instruction)
    
    # Get instructions and targets
    instruction_index = column_names.index(args.instruction)
    instructions, targets = lists[0][:40], lists[instruction_index][:40]
    
    # Initialize attacker and process image
    my_attacker = visual_attacker.Attacker(args, model, targets, instructions, device=device)
    img = vis_processor["eval"](Image.open(args.image_file).convert('RGB')).unsqueeze(0).to(device)
    
    # Perform attack
    attack_fn = my_attacker.attack_unconstrained if not args.constrained else my_attacker.attack_constrained
    adv_img_prompt = attack_fn(img=img, batch_size=8, num_iter=args.n_iters, 
                             alpha=args.alpha/255, epsilon=args.eps/255 if args.constrained else None)
    
    save_image(adv_img_prompt, f'{args.save_dir}/bad_prompt.bmp')