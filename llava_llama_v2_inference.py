import argparse, os, json, csv
import torch
from PIL import Image
from llava_llama_2.utils import get_model
from llava_llama_2_utils import prompt_wrapper, generator
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_path", type=str, default="./ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="instruction_data/0/Sentiment/dataset.csv")
    parser.add_argument("--image_file", type=str, default='./image.bmp')
    parser.add_argument("--output_file", type=str, default='./result.jsonl')
    parser.add_argument("--instruction", type=str, default=None,
                    choices=["positive", "negative", "neutral", "irony", "non_irony", "formal", "informal", 
                            "french", "english", "spanish", "left", "right", "inference_content_evaluation"])
    parser.add_argument("--image_index", type=int, default=0)
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

def get_prompt_template(instruction):
    if not instruction:
        return '%s'
    templates = {
        'informal': 'Please answer the following question with a %s spin. Make the answer sounds informal: %s',
        'french': 'Please answer the following question in %s language. Answer in full sentences so we can tell which language it is in. %s',
        'english': 'Please answer the following question in %s language. Answer in full sentences so we can tell which language it is in. %s',
        'spanish': 'Please answer the following question in %s language. Answer in full sentences so we can tell which language it is in. %s',
        'left': 'Please answer the following question with a %s-wing political spin. %s',
        'right': 'Please answer the following question with a %s-wing political spin. %s',
        'inference_content_evaluation': 'Please evaluate the content of the following question and answer with only yes or no: Does %s describes the content of the image?'
    }
    return templates.get(instruction, 'Please answer the following question with a %s spin: %s')

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize model
    tokenizer, model, image_processor, _ = get_model(args)
    model.eval()
    image = image_processor.preprocess(load_image(args.image_file), return_tensors='pt')['pixel_values'].cuda()
    my_generator = generator.Generator(model=model, tokenizer=tokenizer)
    
    # Prepare data
    lists, _ = read_csv_data(args.data_path)
    datasets = lists[0][40:]
    text_prompt_template = get_prompt_template(args.instruction)
    
    # Generate responses
    out = []
    with torch.no_grad():
        if args.instruction == 'inference_content_evaluation':
            with open('clean_images/label.txt', 'r') as file:
                image_label = file.readlines()[int(args.image_index)].strip()
            for _ in range(20):
                text_prompt = text_prompt_template % image_label
                prompt = prompt_wrapper.Prompt(model, tokenizer, 
                                            text_prompts=prompt_wrapper.prepare_text_prompt(text_prompt), 
                                            device=model.device)
                response = my_generator.generate(prompt, image)
                out.append({'prompt': text_prompt, 'output': response})
        else:
            for user_message in tqdm(datasets, desc="Processing prompts"):
                text_prompt = text_prompt_template % user_message
                prompt = prompt_wrapper.Prompt(model, tokenizer,
                                            text_prompts=prompt_wrapper.prepare_text_prompt(text_prompt),
                                            device=model.device)
                response = my_generator.generate(prompt, image)
                out.append({'prompt': text_prompt, 'output': response})
    # Save results
    with open(args.output_file, 'w') as f:
        f.write(json.dumps({"args": vars(args), "prompt": text_prompt_template}) + "\n")
        for li in out:
            f.write(json.dumps(li) + "\n")
    print(f"Output saved to {args.output_file}")

if __name__ == "__main__":
    main()