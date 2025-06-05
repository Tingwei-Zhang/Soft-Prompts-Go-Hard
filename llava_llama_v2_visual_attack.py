import argparse, torch, os, csv
from torchvision.utils import save_image
from PIL import Image
from llava_llama_2.utils import get_model
from llava_llama_2_utils import visual_attacker, prompt_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_path", type=str, default="ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--data_path", type=str, default="instruction_data/0/Sentiment/dataset.csv")
    parser.add_argument("--instruction", type=str, default='positive',
                    choices=[ "positive", "negative", "neutral", "formal", "informal", "french", "english", "spanish", "left", "right","injection","spam"],
                    help="Instruction to be used for the attack.")
    parser.add_argument("--n_iters", type=int, default=5000, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", type=str, default='constrained',
                    choices=["unconstrained", "constrained", "partial"])
    parser.add_argument("--image_file", type=str, default='./image.bmp', help="Image file")
    parser.add_argument("--save_dir", type=str, default='output', help="save directory")
    return parser.parse_args()

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def read_csv_data(filepath):
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
        column_names = reader.fieldnames
        columns = {name: [row[name] for row in data] for name in column_names}
        return [columns[name] for name in column_names], column_names

if __name__ == "__main__":
    args = parse_args()
    print('>>> Initializing Models')
    tokenizer, model, image_processor, _ = get_model(args)
    model.eval()
    print('[Initialization Finished]\n')

    os.makedirs(args.save_dir, exist_ok=True)
    lists, column_names = read_csv_data(args.data_path)

    # Map instruction to target format
    instruction_map = {'french': 'fr', 'english': 'en', 'spanish': 'es', 
                      'right': 'Republican', 'left': 'Democrat'}
    args.instruction = instruction_map.get(args.instruction, args.instruction)

    instruction_index = column_names.index(args.instruction)
    instructions, targets = lists[0][:40], lists[instruction_index][:40]

    my_attacker = visual_attacker.Attacker(args, model, tokenizer, targets, instructions, 
                                         device=model.device, image_processor=image_processor)

    image = image_processor.preprocess(load_image(args.image_file), 
                                     return_tensors='pt')['pixel_values'].cuda()
    text_prompt_template = prompt_wrapper.prepare_text_prompt('')

    attack_params = {
        'unconstrained': lambda: my_attacker.attack_unconstrained(text_prompt_template, img=image, 
                                                                batch_size=8, num_iter=args.n_iters, 
                                                                alpha=args.alpha/255),
        'partial': lambda: my_attacker.attack_partial_constrained(text_prompt_template, img=image, 
                                                                batch_size=8, num_iter=args.n_iters, 
                                                                alpha=args.alpha/255, epsilon=args.eps/255, 
                                                                rows_to_change=10),
        'constrained': lambda: my_attacker.attack_constrained(text_prompt_template, img=image, 
                                                            batch_size=4, num_iter=args.n_iters, 
                                                            alpha=args.alpha/255, epsilon=args.eps/255)
    }

    adv_img_prompt = attack_params[args.constrained]()
    save_image(adv_img_prompt, f'{args.save_dir}/bad_prompt.bmp')
    print('[Done]')
