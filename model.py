import torch
import torch.nn as nn

#define Neural network class from the nn
class NeuralNet(nn.Module):
    # this is the conctructor method for initializing the neural network
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__() # call the consturctore of the parent class, the nn
        # define the layers of neural network
        self.l1 = nn.Linear(input_size, hidden_size) #first
        self.l2 = nn.Linear(hidden_size, hidden_size) #sec
        self.l3 = nn.Linear(hidden_size, num_classes) #third
        self.relu = nn.ReLU() #rectified linear unit (ReLU) activation funct

    # forward method to define the forward pass of the nn 
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax
        return out  #return output
