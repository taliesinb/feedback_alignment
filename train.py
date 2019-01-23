import torch

from torch import nn

import matplotlib.pyplot as plt
import numpy

import torch.optim as optim
import time

def train_net(net, data, epochs=50, sample_every=5, print_loss=False, live_plot=False, regression=True, updates=1, optimizer='SGD', lr=0.01):
    criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()
    
    if isinstance(optimizer, str):
        optclass = getattr(optim, optimizer)
        optfactory = lambda params: optclass(params, lr=lr)
    elif isinstance(optimizer, dict):
        optclass = getattr(optim, optimizer.pop('type'))
        optfactory = lambda params: optclass(params, lr=lr, **optimizer)
    else:
        raise Exception('bad optimizer')

    optimizer = optfactory(net.parameters())
    losses = []

    if live_plot:
        plt.ion()
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(1,1,1)

    tick = 1
    running_loss = 0.0
    for epoch in range(epochs):
        for batch in data:
            inputs, targets = batch

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            for u in range(updates):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            tick += 1
            if tick % sample_every == 1:
                running_loss /= sample_every
                losses.append(running_loss)
                if print_loss:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
                if live_plot:
                    ax.clear()
                    ax.semilogy(losses)
                    fig.canvas.draw()
                running_loss = 0.0

    if live_plot:
        plt.ioff()
        plt.cla()
        plt.close()

    return {'loss': losses}

def test_categorical_accuracy(net, data_loader):
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = net(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(data_loader.dataset)
