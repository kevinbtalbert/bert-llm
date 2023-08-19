from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Load pre-trained BERT model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Define context and question
context = '''Cloudera Machine Learning optimizes ML workflows across your business with native 
and robust tools for deploying, serving, and monitoring models. With extended SDX for models, 
govern and automate model cataloging and then seamlessly move results to collaborate across 
CDP experiences including Data Warehouse and Operational Database.'''
question = "What does Cloudera Machine Learning do"

# Define encoding
encoding = tokenizer.encode_plus(text=question,text_pair=context)

# Token embeddings
inputs = encoding['input_ids']

# Segment embeddings
sentence_embedding = encoding['token_type_ids']

# Input tokens
tokens = tokenizer.convert_ids_to_tokens(inputs) 

# Create output model
output = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
start_index = torch.argmax(output.start_logits)
end_index = torch.argmax(output.end_logits)
answer = ' '.join(tokens[start_index:end_index+1])

# Print model result
print(answer)