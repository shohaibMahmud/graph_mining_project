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

n_epochs = 100
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
val_precision = []
val_recall = []
train_loss = []
train_acc = []
total_step = len(train_dataloader)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss()

net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 128)
net.fc = net.fc.cuda() if device else net.fc
net = net.cuda() if device else net
net.load_state_dict(torch.load('resnet.pt'))
net.eval()

tcn = TCN(128, [128, 96, 64], 51, 0.2)
tcn = tcn.cuda() if device else tcn

aeggNetwork = nn.Sequential(
    nn.Flatten(),
    nn.Linear(60*64, 64),
    nn.Sigmoid(),
    nn.Linear(64, 32),
    nn.Sigmoid()
)
outputLayer = nn.Linear(32,1)
outputLayer = outputLayer.cuda() if device else outputLayer
aeggNetwork = aeggNetwork.cuda() if device else aeggNetwork
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
activation = nn.Sigmoid()


aeggNetwork.train()
outputLayer.train()
tcn.train()

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
            output = output.reshape(1, 128, 60)
            outputsResNet[x,:,:] = output

        tcnoutput = tcn(outputsResNet)

        aeggOutput = aeggNetwork(tcnoutput)
        output = activation(outputLayer(aeggOutput))
        target_ = target_.unsqueeze(1).to(torch.float32)
        loss = criterion(output, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = torch.round(output)

        correct += torch.sum(pred == target_).item()
        total += target_.size(0)
        if (batch_idx) % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')

    batch_loss = 0
    total_t = 0
    correct_t = 0
    TP = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        aeggNetwork.eval()
        outputLayer.eval()
        tcn.eval()
        for data_t, target_t in (test_dataloader):
            data_t, target_t = data_t.to(device), target_t.to(device)

            ResNet_outputs_t = torch.empty(batch_size, 128, 60).to(device)
            for x, videoDataFrames in enumerate(data_t):
                output = net(videoDataFrames)
                output = output.reshape(1, 128, 60)
                ResNet_outputs_t[x, :, :] = output

            tcnoutput = tcn(ResNet_outputs_t)
            aeggOutput = aeggNetwork(tcnoutput)
            output_t = activation(outputLayer(aeggOutput))
            target_t = target_t.unsqueeze(1).to(torch.float32)
            loss_t = criterion(output_t, target_t)
            batch_loss += loss_t.item()
            pred_t = torch.round(output_t)

            correct_t += torch.sum(pred_t == target_t).item()
            total_t += target_t.size(0)
            TP += torch.sum((pred_t == target_t)*target_t).item()
            FP += torch.sum((pred_t != target_t)*pred_t).item()
            FN += torch.sum((pred_t != target_t)*target_t).item()

        val_acc.append(100 * correct_t / total_t)
        val_loss.append(batch_loss / len(test_dataloader))
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        val_precision.append(precision)
        val_recall.append(recall)
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\
        ,validation precision:{100*precision}:.4f, validation recall:{100*recall}:.4f\n')

        if network_learned:
            valid_loss_min = batch_loss
            torch.save(tcn.state_dict(), 'tcn.pt')
            torch.save(aeggNetwork.state_dict(), 'aegg.pt')
            torch.save(outputLayer.state_dict(), 'outputLayer.pt')
            print('Improvement-Detected, save-model')
        tcn.train()
        aeggNetwork.train()
        outputLayer.train()
