import argparse, os, random, csv
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
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="instruction_data/0/Sentiment/dataset.csv")
    parser.add_argument("--instruction", type=str, default='positive',
                    choices=["positive", "negative", "neutral", "formal", "informal", 
                            "french", "english", "spanish", "left", "right", 
                            "injection", "spam"], help="Instruction to be used for the attack.")
    parser.add_argument("--n_iters", type=int, default=5000)
    parser.add_argument('--eps', type=int, default=32)
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument("--constrained", type=str, default='constrained',
                        choices=["unconstrained", "constrained", "partial", "l2", "jpeg"])
    parser.add_argument("--image_file", type=str, default='./image.bmp')
    parser.add_argument("--save_dir", type=str, default='output')
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def read_csv_data(filepath):
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
        column_names = reader.fieldnames
        columns = {name: [row[name] for row in data] for name in column_names}
        return [columns[name] for name in column_names], column_names

def main():
    args = parse_args()
    cfg = Config(args)
    
    # Model initialization
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model = registry.get_model_class(model_config.arch).from_config(model_config).to(f'cuda:{args.gpu_id}')
    model.eval()
    
    # Visual processor setup
    vis_processor = registry.get_processor_class(
        cfg.datasets_cfg.cc_sbu_align.vis_processor.train.name
    ).from_config(cfg.datasets_cfg.cc_sbu_align.vis_processor.train)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Process instruction
    instruction_map = {'french': 'fr', 'english': 'en', 'spanish': 'es', 
                      'right': 'Republican', 'left': 'Democrat'}
    args.instruction = instruction_map.get(args.instruction, args.instruction)
    
    # Load and process data
    lists, column_names = read_csv_data(args.data_path)
    instruction_index = column_names.index(args.instruction)
    instructions, targets = lists[0][:40], lists[instruction_index][:40]
    
    # Setup attacker
    my_attacker = visual_attacker.Attacker(args, model, targets, instructions, device=model.device)
    img = vis_processor(Image.open(args.image_file).convert('RGB')).unsqueeze(0).to(model.device)
    
    # Perform attack
    attack_methods = {
        'unconstrained': lambda: my_attacker.attack_unconstrained(
            prompt_wrapper.minigpt4_chatbot_prompt_text_attack, img, batch_size=8, 
            num_iter=2000, alpha=args.alpha/255),
        'partial': lambda: my_attacker.attack_partial_constrained(
            prompt_wrapper.minigpt4_chatbot_prompt_text_attack, img, batch_size=8,
            num_iter=2000, alpha=args.alpha/255, rows_to_change=50),
        'l2': lambda: my_attacker.attack_l2(
            prompt_wrapper.minigpt4_chatbot_prompt_text_attack, img, batch_size=8,
            num_iter=2000, alpha=args.alpha, epsilon=args.eps),
        'jpeg': lambda: my_attacker.attack_jpeg(
            prompt_wrapper.minigpt4_chatbot_prompt_text_attack, img, batch_size=8,
            num_iter=2000, alpha=args.alpha, epsilon=args.eps/255),
        'constrained': lambda: my_attacker.attack_constrained(
            prompt_wrapper.minigpt4_chatbot_prompt_text_attack, img, batch_size=8,
            num_iter=2000, alpha=args.alpha/255, epsilon=args.eps/255)
    }
    
    adv_img_prompt = attack_methods.get(args.constrained, attack_methods['constrained'])()
    save_image(adv_img_prompt, f'{args.save_dir}/bad_prompt.bmp')

if __name__ == "__main__":
    main()
