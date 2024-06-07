# Training data for image soft prompt

instruction_data folder stores the training data of generating image soft prompt with different instructions.

## Meta-objectives:

[Sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest): Positive, Negative, Neutral

[Politics](https://huggingface.co/m-newhauser/distilbert-political-tweets): Democrat, Republican

[Language](https://huggingface.co/papluca/xlm-roberta-base-language-detection): English, French, Spanish

[Formality](https://huggingface.co/s-nlp/roberta-base-formality-ranker): Formal, Informal

## Prompt to synthesize training data with Chatgpt

1. Generate 20 questions about the 'label' (e.g. cassette player) in the image. Make them a table
2. Generate 20 different questions about the 'label' (e.g. cassette player) in the image. Make them a table
3. Generate 20 different questions about the 'label' (e.g. cassette player) in the image. Make them a table
4. Answer the following questions about the (cassette player) in the image with a (positive/negative/neutral) spin. Make them a table
   

## Validate synthesized data
run training_data_evaluation.ipynb to validate if synthesized data is correctly classsified by the evaluators 
