---
title: "An Introduction to GoogLeNet and a Small Experiment"
date: 2020-10-23 07:15:00
tags: [ai, deep learning, googlenet, pytorch, python]
des: "This post briefly introduces GoogLeNet and shares my experimental results after casually modifying the model."
lang: en
translation_key: googlenet
---

## Introduction

GoogLeNet was first introduced in Google’s paper, [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf). The paper proposes the Inception V1 / GoogLeNet architecture, which ranked 1st in the classification track of ILSVRC-2014 (Top-5 Error = 6.67%). The model has only about 6.8 million parameters—9× fewer than AlexNet, and 20× fewer than VGG-16—so it is much more lightweight.

In this post, I briefly introduce GoogLeNet and share the results of a small experiment where I “randomly” modified the GoogLeNet model.

## GoogLeNet Architecture Overview

In many cases, it’s not obvious when we should use max-pooling versus convolution. GoogLeNet essentially uses them all at once: it applies convolutions with different kernel sizes and max-pooling in parallel, then concatenates their outputs. This structure is called an Inception module, and GoogLeNet is composed of many stacked Inception modules.

![GoogLeNet](https://user-images.githubusercontent.com/18013815/96935484-bd937500-14f6-11eb-9c1e-a87e2050bef4.png)

The diagram above illustrates an Inception module. Another key idea in GoogLeNet is the concept of a *bottleneck*. In the figure, the left is a “standard” Inception module, while the right is a modified version that introduces 1×1 convolutions. With 1×1 conv, we can greatly reduce the number of parameters—hence the term bottleneck.

![GoogLeNet Inception Parameters](https://user-images.githubusercontent.com/18013815/96935814-75c11d80-14f7-11eb-9a73-70d52ce63e1d.png)

Without the bottleneck, the number of MACs (Multiply–Accumulate Operations) is $((28\times 28\times 5\times 5)\times 192)\times 32 ≃ 120$.

With the help of 1×1 conv to reduce computation, the MACs become the first layer $((28\times 28\times 1\times 1)\times 192)\times 16 ≃ 2.4M$ plus the second layer $((28\times 28\times 5\times 5)\times 16)\times 32 ≃ 10M $—about $12.4M$ total. You can see that the MAC count drops by roughly an order of magnitude, and in practice the parameter count is also reduced by around 10×.

![GoogLeNet Architecture](https://user-images.githubusercontent.com/18013815/96936889-7fe41b80-14f9-11eb-8159-ffd97a34bd91.png)

The figure above shows the overall GoogLeNet architecture. Roughly speaking, it contains nine Inception modules.

![GoogLeNet Parameter Table](https://user-images.githubusercontent.com/18013815/96937010-cafe2e80-14f9-11eb-8e33-d787171acff9.png)

For detailed configuration, refer to the table above.

## Implementing GoogLeNet in PyTorch

The code isn’t long, so I’m pasting it in full. It’s basically the MNIST example from PyTorch, adapted to use CIFAR, and it uses the GoogLeNet module from the [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100) project.

If you copy the code below, it should run directly. My environment is Python 3 + PyTorch 1.6 + CUDA 10.2.

<details>

```python
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        # 1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1conv -> 5x5conv branch
        # we use 2 3x3 conv filters stacked instead
        # of 1 5x5 filters to obtain the same receptive
        # field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        # 3x3pooling -> 1x1conv
        # same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self,\times):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        # although we only use 1 conv layer as prelayer,
        # we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        # """In general, an Inception network is a network consisting of
        # modules of the above type stacked upon each other, with occasional
        # max-pooling layers with stride 2 to halve the resolution of the
        # grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        # input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

    def forward(self,\times):
        output = self.prelayer(x)
        output = self.a3(output)
        output = self.b3(output)

        output = self.maxpool(output)

        output = self.a4(output)
        output = self.b4(output)
        output = self.c4(output)
        output = self.d4(output)
        output = self.e4(output)

        output = self.maxpool(output)

        output = self.a5(output)
        output = self.b5(output)

        # """It was found that a move from fully connected layers to
        # average pooling improved the top-1 accuracy by about 0.6%,
        # however the use of dropout remained essential even after
        # removing the fully connected layers."""
        output = self.avgpool(output)
        output = self.dropout(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)

        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    top5count = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)

            v, result = output.topk(5, 1, True, True)
            top5count += torch.eq(result, target.view(-1, 1)
                                  ).sum().int().item()

            # sum up batch loss
            test_loss += F.cross_entropy(output,
                                         target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Top 1 Error: {}/{} ({:.2f}), Top 5 Error: {}/{} ({:.2f})\n'.format(
        test_loss,
        len(test_loader.dataset) - correct, len(test_loader.dataset),
        1 - correct / len(test_loader.dataset),
        len(test_loader.dataset) - top5count, len(test_loader.dataset),
        1 - top5count / len(test_loader.dataset),
    ))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default:   64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 10)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, **train_kwargs)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, **test_kwargs)

    model = GoogleNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, trainloader, optimizer, epoch)
        test(model, device, testloader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Parameters:", params)


if __name__ == '__main__':
    main()
```

</details>

## A Small GoogLeNet Experiment

Next, I ran a few experiments to see how GoogLeNet performs. I tested:

- The bottleneck version of GoogLeNet
- A naïve version without bottlenecks (“Naïve GoogLeNet”)
- A “GoogLeNet Long” variant by arbitrarily adding two Inception modules
- A “GoogLeNet Short” variant by arbitrarily removing some Inception layers

The most aggressively reduced one is “GoogLeNet Short4”, which only has two Inception modules left. You can roughly infer the model size by looking at the parameter count.

I ran these models on both CIFAR-100 and CIFAR-10, and recorded Top-1 Error, Top-5 Error, Parameters, and Time.

**GoogLeNet on CIFAR-100**:

|                         |     Top   1 Error     |     Top 5 Error    |     Parameters    |     Time(14 epoch)    |
|-------------------------|-----------------------|--------------------|-------------------|-----------------------|
|     GoogleNet Naïve     |     0.36              |     0.09           |     65736148      |     52m38s            |
|     GoogleNet           |     0.34              |     0.10           |     6258500       |     29m8s             |
|     GoogleNet Long      |     0.35              |     0.10           |     9641924       |     36m41s            |
|     GoogleNet Short     |     0.32              |     0.09           |     5271652       |     23m11s            |
|     GoogleNet Short2    |     0.32              |     0.09           |     3523556       |     16m29s            |
|     GoogleNet Short3    |     0.36              |     0.11           |     1985220       |     9m3s              |
|     GoogleNet Short4    |     0.44              |     0.15           |     1650084       |     8m56s             |

**GoogLeNet on CIFAR-10**:

|                         |     Top   1 Error     |     Top 5 Error    |     Parameters    |     Time(14 epoch)    |
|-------------------------|-----------------------|--------------------|-------------------|-----------------------|
|     GoogleNet Naïve     |     0.15              |     0.01           |     65291098      |     52m51s            |
|     GoogleNet           |     0.10              |     0.00           |     6166250       |     28m45s            |
|     GoogleNet Long      |     0.11              |     0.00           |     9549674       |     40m12s            |
|     GoogleNet Short     |     0.10              |     0.00           |     5179402       |     27m30s            |
|     GoogleNet Short2    |     0.10              |     0.00           |     3431306       |     31m57s            |
|     GoogleNet Short3    |     0.11              |     0.00           |     1892970       |     26m31s            |
|     GoogleNet Short4    |     0.15              |     0.01           |     1557834       |     25m30s            |

First, you can see that the parameter count of the naïve version is indeed about 10× larger, but the accuracy is not dramatically different. Also, except for Short4, almost all variants perform similarly: on CIFAR-100, Top-1 Error is roughly around 0.35 and Top-5 Error around 0.10; on CIFAR-10, Top-1 Error is around 0.10 and Top-5 Error is around 0.00 (nearly zero).

My guess is that CIFAR-100 and CIFAR-10 are not complex enough, so for image classification the depth of the model doesn’t matter that much. In addition to depth, the number of channels also affects accuracy. If the number of channels is large enough, perhaps you don’t need such a deep network. I also tried randomly adding max-pooling layers and dropout layers to GoogLeNet, but the errors were largely the same. This suggests that for image classification, the model itself has fairly high tolerance.

Of course, the conclusions above only apply to simple datasets. On more challenging benchmarks, even a few percentage points of accuracy matter a lot—tiny differences after the decimal point often represent countless engineering ideas and hard work. Still, the most impressive part of GoogLeNet is that it removes a huge number of parameters while achieving nearly the same accuracy!

