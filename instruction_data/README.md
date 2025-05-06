# Corpus for image soft prompt

Folders from 0 to 4 store the training data of generating image soft prompt with different instructions.

[Emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions): admiration, love, gratitude, fear, amusement, anger

[Gender](https://huggingface.co/padmajabfrl/Gender-Classification): Female, Male

[Irony](https://huggingface.co/cardiffnlp/twitter-roberta-base-irony): Irony, Non-irony

[Sentiment](https://huggingface.co/SamLowe/roberta-base-go_emotions): positive, negative, neutral 

# Prompt to synthesize Corpus with Chatgpt

1. Generate 20 questions about the 'label' (e.g. cassette player) in the image. Make them a table
2. Generate 20 more different questions about the 'label' (e.g. cassette player) in the image. Make them a table
3. Generate 20 more different questions about the 'label' (e.g. cassette player) in the image. Make them a table
4. Answer the following questions about the (cassette player) in the image with a (positive/negative/neutral) spin. Make them a table
   

# Validate synthesized data

run training_data_evaluation.ipynb to validate if synthesized data is correctly classsified by the evaluators 
