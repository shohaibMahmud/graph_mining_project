import torch.nn.functional as F
from tcn import TemporalConvNet
import torch.nn as nn

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout=0.2): #layer3 output layer
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)


    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        output = self.tcn(inputs)  # input should have dimension (N, C, L)
        #y2 = self.tcn(y1)
        #output = self.tcn(y2)
        return output