import torch
import torchvision
from torchvision import transforms
from model import Net
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import dataset

def train():
    # params
    n_epochs = 5
    batch_size_train = 16
    learning_rate = 0.01
    momentum = 0.5  

    # data
    trainloader = torch.utils.data.DataLoader(dataset.trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)

    # model
    net = Net()
    
    # optimization
    optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum)
    criterition = torch.nn.CrossEntropyLoss()

    # train
    net.train()
    train_losses = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad() # zero gradient
            outputs = net.forward(inputs) # forward pass
            loss = criterition(outputs, labels) # compute the loss
            loss.backward() # calculate the gradient
            optimizer.step() # make actual changes to learnable parameter

            running_loss += loss.item()
            train_losses.append(loss.item())

            if i % 10 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0
                
    print('Finished training')
    torch.save(net.state_dict(), 'net.pth')
    plt.plot(train_losses)
    plt.xlabel('training samples')
    plt.ylabel('loss')
    plt.show()

if __name__ == '__main__':
    train()