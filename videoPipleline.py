import torch.nn as nn
from videoDataSet import videoDataSet
import numpy as np
from torchvision import *
from torch.utils.data import DataLoader
from stackedTCN import TCN
import torch.optim as optim

train_dataset = videoDataSet('C:\\shuvo\\graph_mining_project\\train2',60,None)
test_dataset = videoDataSet('C:\\shuvo\\graph_mining_project\\test2',60,None)

batch_size = 1

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

n_epochs = 5
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()

net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 128)
net.fc = net.fc.cuda() if device else net.fc
net = net.cuda() if device else net
net.load_state_dict(torch.load('resnet.pt'))
net.eval()

tcnEncoder = TCN(128, [64], 51, 0.2)
tcnEncoder = tcnEncoder.cuda() if device else tcnEncoder
tcnDecoder = TCN(64, [64, 96, 128], 51, 0.2)
tcnDecoder = tcnDecoder.cuda() if device else tcnDecoder
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for epoch in range(1, n_epochs + 1):
    running_loss = 0.0
    correct = 0
    total = 0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_) in enumerate(train_dataloader):
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        outputsResNet = torch.empty(batch_size, 128, 60).to(device)
        for x,videoDataFrames in enumerate(data_):
            output = net(videoDataFrames)
            output = output.reshape(128, 60)
            outputsResNet[x,:,:] = output
        #print(outputsResNet.shape)
        outputsEncoder = tcnEncoder(outputsResNet)
        print(outputsEncoder.shape)
        outputsDecoder = tcnDecoder(outputsEncoder)
        loss = criterion(outputsDecoder, outputsResNet)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_loss.append(running_loss / total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}')
        batch_loss = 0
        tcnEncoder.train()
        tcnDecoder.train()