import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sqlite3
from model import NeuralNet

# Establish a connection to the db.
conn = sqlite3.connect('chatbot.db')
c = conn.cursor()

# Retrieve patterns and responses from the database
c.execute('SELECT pattern, response FROM patterns JOIN responses ON patterns.intent_id = responses.intent_id')
data = c.fetchall()

# Close the connection
conn.close()

# Process the retrieved data
patterns = []
responses = []
for pattern, response in data:
    patterns.append(pattern)
    responses.append(response)

# Load intents from JSON
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []  # hold both patterns in json file

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)  # tag array
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)  # put into all words array
        xy.append((w, tag))  # know the pattern and corresponding tag

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]# see if the stemming works
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# train data
x_train = []
y_train = []

# loop xy array
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)  # crossEntropyLoss

# convert numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)

# create a new dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = torch.from_numpy(x_train).float()
        self.y_data = torch.from_numpy(y_train)

    # dataset idx
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000

# testing
# print(input_size, len(all_words))
# print(output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # check if the cpu available

# create model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Do actually training loop. Study back the epochs.
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

# save the data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. File saved to {FILE}')
