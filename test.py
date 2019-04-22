import torch
import torchvision
from model import Net
from torchvision import transforms
import dataset

def test():
    correct = 0
    total =0

    # data
    testloader = torch.utils.data.DataLoader(dataset.testset)

    # net
    net = Net()
    net.load_state_dict(torch.load('net.pth'))

    for data in testloader:
        images, labels = data
        output = net.forward(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuraty: %d / %d (%f)' % (correct, total, correct/total))

if __name__=='__main__':
    test()
