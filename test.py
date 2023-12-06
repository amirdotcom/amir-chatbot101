# This script loads the trained model from data.pth and tests it some sample sentences. Check if the chatbot responses are appropriate based on the training data.
import json
from nltk_utils import tokenize, stem, bag_of_words
import torch
from model import NeuralNet

def get_response(model, all_words, tags, sentence):
    # tokenize and preprocess the sentence
    sentence_words = tokenize(sentence)
    sentence_bag = bag_of_words(sentence_words, all_words)

    # convert the bag of words to a Pytorch tesnsor
    input_tensor = torch.from_numpy(sentence_bag).float()

    # make the prediction
    model.eval() #set the model to evaluation mode
    with torch.no_grad():
        output = model(input_tensor)

    
    # get the predicted tag
        _, predicted_idx = torch.max(output, 0)
    tag = tags[predicted_idx.item()]

    return tag
# load the trained model
data = torch.load('data.pth')
model_state = data['model_state']
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']

# create the model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# test the model with some sample sentences
sentences_to_test = ["hello", "How are you?", "is anyone there"]

for sentence in sentences_to_test:
    response_tag = get_response(model, all_words, tags, sentence)
    print(f"User: {sentence}, Chatbot Response: {response_tag}")


