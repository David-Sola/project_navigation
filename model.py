import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer=[200, 100, 20]):
        '''
        Class to create a neural network with given input size(=state_size) output size(=action_size) and a number of
        hidden layers. The number of hidden layers can be adjusted without changing the class
        :param state_size: Size of the input layer of the NN
        :param action_size: Size of the output layer of the NN
        :param hidden_layer: Size(s) of the hidden layers
        '''
        super(QNetwork, self).__init__()
        # Create the NN here and start with the Input and first hidden layer
        # The state_size is the input to the NN
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layer[0])])
        # Extend the network with all specified hidden layers.
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2, in zip(hidden_layer[:-1], hidden_layer[1:])])
        self.output = nn.Linear(hidden_layer[-1], action_size)

    def forward(self, x):
        '''
        Return the value after the feedforward path has gone through the hidden layers and also through the last output
        layer. Calculate the forward path through the complete network with it´s hidden layers.
        As activation function the recitifier linear uinit has been choosen as it´s a pretty
        simple activation function in itself but breaks the problem to also be able to solf non - linear problems
        :param x: Input to the neural network which has to fit the state_size
        :return: return the feed_forward output of the neural network which has the size of action_size
        '''
        for each in self.hidden_layers:
            x = F.relu(each(x))
        return self.output(x)
