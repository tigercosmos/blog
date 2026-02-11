---
title: "GoogLeNet の概要と簡単な実験"
date: 2020-10-23 07:15:00
tags: [ai, deep learning, googlenet, pytorch, python]
des: "本記事では GoogLeNet を簡単に紹介し、モデルを少し改造して試した実験結果を共有します。"
lang: jp
translation_key: googlenet
---

## イントロダクション

GoogLeNet は、Google の論文 [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf) で初めて提案されました。この論文では Inception V1 / GoogLeNet アーキテクチャが紹介され、ILSVRC-2014 の分類部門で 1 位（Top-5 Error = 6.67%）を獲得しています。パラメータ数は約 680 万で、AlexNet より 9 倍少なく、VGG-16 より 20 倍少ないため、非常に軽量なモデルです。

本記事では GoogLeNet を簡単に紹介し、私が適当に（？）改造した GoogLeNet で行った小さな実験結果を共有します。

## GoogLeNet モデル概要

多くの場合、Max-pooling と Convolution をどのタイミングで使うべきかは直感的に分かりづらいです。そこで GoogLeNet は、異なるカーネルサイズの Convolution と Max-pooling を並列に適用し、それらの出力を Concatenate して 1 つの出力にまとめます。これを Inception モジュールと呼び、GoogLeNet は多数の Inception モジュールを積み重ねて構成されています。

![GoogLeNet](https://user-images.githubusercontent.com/18013815/96935484-bd937500-14f6-11eb-9c1e-a87e2050bef4.png)

上図は Inception モジュールの模式図です。GoogLeNet のもう 1 つの重要なアイデアは、*Bottleneck* の概念です。図の左が通常の Inception モジュールで、右が改造版です。改造版では 1×1 Convolution を導入しており、1×1 Conv によってパラメータ数を大幅に削減できます。これを Bottleneck と呼びます。

![GoogLeNet Inception Parameters](https://user-images.githubusercontent.com/18013815/96935814-75c11d80-14f7-11eb-9a73-70d52ce63e1d.png)

Bottleneck を使わない場合、MAC（Multiply–Accumulate Operation）の回数は $((28\times 28\times 5\times 5)\times 192)\times 32 ≃ 120$ になります。

一方で 1×1 Conv によって計算量を削減すると、MAC は第 1 層が $((28\times 28\times 1\times 1)\times 192)\times 16 ≃ 2.4M$、第 2 層が $((28\times 28\times 5\times 5)\times 16)\times 32 ≃ 10M $ となり、合計でおよそ $12.4M$ になります。MAC は概ね 10 倍程度減り、実際のパラメータ数も同様に約 10 倍減ることが分かります。

![GoogLeNet Architecture](https://user-images.githubusercontent.com/18013815/96936889-7fe41b80-14f9-11eb-8159-ffd97a34bd91.png)

上図は GoogLeNet 全体のアーキテクチャです。大まかには 9 個の Inception モジュールから構成されています。

![GoogLeNet Parameter Table](https://user-images.githubusercontent.com/18013815/96937010-cafe2e80-14f9-11eb-8e33-d787171acff9.png)

詳細な構成は上の表を参照してください。

## PyTorch による GoogLeNet 実装

コードはそれほど長くないので、そのまま全部貼ります。基本的には PyTorch の MNIST サンプルを土台にしつつ、[pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100) プロジェクトにある GoogLeNet モジュールを利用しています。

以下のコードをコピーすれば、そのまま実行できるはずです。私の環境は Python3 + PyTorch 1.6 + CUDA 10.2 です。

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

## GoogLeNet の小さな実験

次に、GoogLeNet の性能を見てみるためにいくつか実験をしました。Bottleneck ありの GoogLeNet と、Bottleneck なしの Naïve GoogLeNet を用意しました。さらに、元の GoogLeNet に Inception を適当に 2 層追加したものを GoogLeNet Long、Inception 層を適当に減らしたものを GoogLeNet Short と呼ぶことにします。最も削減したのが GoogLeNet Short4 で、Inception は 2 層しか残っていません。パラメータ数を見ると、モデルの大きさを大まかに把握できます。

これらのモデルで CIFAR-100 と CIFAR-10 を学習し、Top-1 Error、Top-5 Error、Parameters、Time を記録しました。

**CIFAR-100 での結果**：

|                         |     Top   1 Error     |     Top 5 Error    |     Parameters    |     Time(14 epoch)    |
|-------------------------|-----------------------|--------------------|-------------------|-----------------------|
|     GoogleNet Naïve     |     0.36              |     0.09           |     65736148      |     52m38s            |
|     GoogleNet           |     0.34              |     0.10           |     6258500       |     29m8s             |
|     GoogleNet Long      |     0.35              |     0.10           |     9641924       |     36m41s            |
|     GoogleNet Short     |     0.32              |     0.09           |     5271652       |     23m11s            |
|     GoogleNet Short2    |     0.32              |     0.09           |     3523556       |     16m29s            |
|     GoogleNet Short3    |     0.36              |     0.11           |     1985220       |     9m3s              |
|     GoogleNet Short4    |     0.44              |     0.15           |     1650084       |     8m56s             |

**CIFAR-10 での結果**：

|                         |     Top   1 Error     |     Top 5 Error    |     Parameters    |     Time(14 epoch)    |
|-------------------------|-----------------------|--------------------|-------------------|-----------------------|
|     GoogleNet Naïve     |     0.15              |     0.01           |     65291098      |     52m51s            |
|     GoogleNet           |     0.10              |     0.00           |     6166250       |     28m45s            |
|     GoogleNet Long      |     0.11              |     0.00           |     9549674       |     40m12s            |
|     GoogleNet Short     |     0.10              |     0.00           |     5179402       |     27m30s            |
|     GoogleNet Short2    |     0.10              |     0.00           |     3431306       |     31m57s            |
|     GoogleNet Short3    |     0.11              |     0.00           |     1892970       |     26m31s            |
|     GoogleNet Short4    |     0.15              |     0.01           |     1557834       |     25m30s            |

まず、Naïve はパラメータ数が確かに 10 倍ほど多いものの、Accuracy の差はそれほど大きくありません。さらに Short4 を除けば、ほぼすべてのモデルの結果は同程度です。CIFAR-100 では Top-1 Error がだいたい 0.35、Top-5 Error が 0.10 前後で、CIFAR-10 では Top-1 Error が 0.10、Top-5 Error が 0.00（ほぼ 0）前後になっています。

これは CIFAR-100 / CIFAR-10 がそこまで複雑ではないため、画像認識においてはモデルの深さの影響が相対的に小さいからだと推測しています。また、深さ以外にもチャンネル数は精度に影響します。チャンネル数が十分大きければ、そこまで深いネットワークが必要ないのかもしれません。GoogLeNet に Max-Pooling 層や Dropout 層を適当に足してみる実験もしましたが、Error はほぼ同じでした。画像認識ではモデル自体の「許容度（頑健性）」がかなり高いのだと思います。

ただし以上の推論は簡単なデータセットでの話に限られます。複雑なベンチマークでは、数 % の精度差が非常に重要であり、小数点以下の僅かな違いの裏に膨大な工夫と労力が隠れています。それでも、GoogLeNet のすごいところは、パラメータを大幅に減らしながら精度がほとんど変わらない点です！

