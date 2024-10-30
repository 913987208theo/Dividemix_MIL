'''
1. 引入模块和设置参数
'''
from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid) #设置 GPU 设备的 ID，指定使用哪个 GPU 进行训练。
random.seed(args.seed) #设置 Python 内置的 random 模块的种子，用于确保程序中的任何随机操作在不同的运行中结果保持一致。
torch.manual_seed(args.seed)  #设置 PyTorch 中 CPU 的随机种子，确保所有 CPU 上的随机操作（如数据加载和初始化）是可重复的。
torch.cuda.manual_seed_all(args.seed)  #设置所有 GPU 上的随机种子，以确保所有与 GPU 相关的随机操作（如随机 dropout 和模型权重初始化）在不同的运行中保持一致。

'''2. train 函数 —— 半监督训练
训练模型。该函数执行 半监督学习 训练，一个网络负责固定评估，另一个负责训练。用 MixMatch 方式进行训练数据增强。
'''
# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):  #有标签和无标签数据的 DataLoader。
    net.train()
    net2.eval()  # 将 net 设为训练模式，将 net2 设为评估模式。这样只训练 net，而利用 net2 来提供无标签数据的预测。

    unlabeled_train_iter = iter(unlabeled_trainloader) #获取无标签数据迭代器
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1  #计算训练时每个 epoch 中需要进行的迭代次数 (num_iter)。

    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        '''
        这一行代码通过遍历 labeled_trainloader 来获取有标签数据集的批次数据。
        inputs_x 和 inputs_x2：这两个变量表示同一批数据在不同数据增强方法下的两组图像（MixMatch 方法中常用的一种策略）。
        labels_x：这表示有标签数据的标签。
        w_x：这是每个样本的权重，通常与该样本是否为噪声样本有关。
        batch_idx：这是批次的索引（0, 1, 2, ...）。
        '''
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next() #从新的迭代器中获取一个无标签数据批次。
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)  #获取有标签数据批次的大小，即每个批次中的样本数量。
        '''
        该代码片段的目的是：
        从 labeled_trainloader 中加载有标签数据，用于有监督训练。
        同时从 unlabeled_trainloader 中加载无标签数据，用于半监督学习。
        '''

        '''将标签转换为 one-hot 编码，以便进行后续的计算。
        将原始标签转换为 one-hot 编码，便于在交叉熵损失等计算中使用。
        '''
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        '''      
这段代码的主要功能是生成有标签和无标签数据的目标概率分布，应用于半监督学习中。具体来说，它包括了对无标签数据的"猜测标签"以及对有标签数据的"标签精炼"。
        '''
        #这里主要是为了生成伪标签，因此梯度更新不是必要的。
        with torch.no_grad():
            # label co-guessing of unlabeled samples

            '''!!!无标签数据的标签共同猜测 (Label Co-guessing)
            inputs_u 和 inputs_u2是无标签样本
            '''
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            '''最后将所有四个概率的平均值求出，得到最终的预测概率 pu。这样做是为了获得更稳定和更准确的标签预测，降低单一模型预测的不确定性。'''
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
                  torch.softmax(outputs_u21,dim=1) + torch.softmax(outputs_u22, dim=1)) / 4

            #控制模型的输出概率分布的平滑程度。args.T 通常是一个小于 1 的值，目的是让概率分布变得更加尖锐，从而提高模型对少数派类别的区分能力。
            ptu = pu ** (1 / args.T)  # 温度削尖 temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # 归一化： 确保每行（样本）的概率和为 1
            targets_u = targets_u.detach() #使用 detach() 从计算图中分离出来，因为这个部分只是生成伪标签，不需要计算梯度。

            '''!!!有标签数据的标签精炼 (Label Refinement)'''
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px  #px 会与原始的标签 labels_x 进行混合，混合系数由权重 w_x 决定。w_x 代表对原始标签的信任程度，介于 0 和 1 之间。
            ptx = px ** (1 / args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach() #因为这个部分同样是生成标签，不需要计算梯度。


        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)  #生成一个 Beta 分布参数为 args.alpha 的随机数 l。
        l = max(l, 1 - l)  #Beta 分布的输出值介于 0 到 1 之间，通常用于数据混合权重的选择。
                            # 确保混合比例 l 更接近于中间值，从而避免数据被完全偏向一个数据样本。

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0) #将有标签的数据 (inputs_x, inputs_x2) 和无标签的数据 (inputs_u, inputs_u2) 进行拼接，形成一个完整的输入集 all_inputs。
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)  #目标标签 (targets_x, targets_u) 也进行拼接形成 all_targets。
        #这种拼接是为了让有标签和无标签的数据在同一批次中进行混合操作。

        idx = torch.randperm(all_inputs.size(0)) #生成一个随机排列，用于将数据打乱。

        #input_a, input_b = all_inputs, all_inputs[idx]：input_b 是 all_inputs 的随机排列，类似地，target_b 是 all_targets 的随机排列。
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]    #通过这种方式，数据被随机 配对，便于后续的混合操作。

        mixed_input = l * input_a + (1 - l) * input_b  #数据混合 (Mixing Inputs)
        mixed_target = l * target_a + (1 - l) * target_b  #标签混合 (Mixing Targets)   #标签平滑 (Label Smoothing)

        logits = net(mixed_input)  #将混合后的输入 mixed_input 传入神经网络 net 进行前向传播，得到预测值 logits。
        logits_x = logits[:batch_size * 2] #logits_x：属于有标签数据的预测结果。
        logits_u = logits[batch_size * 2:]  #logits_u：属于无标签数据的预测结果。

        '''
        调用损失函数 criterion，计算有标签和无标签数据的损失：
        Lx：有标签数据的监督损失。
        Lu：无标签数据的损失。
        lamb：损失的权重，通常是随着训练的进行动态调整的。
        criterion 结合了有监督损失和无监督损失，这样可以确保在训练过程中充分利用有标签和无标签的数据，提高模型的泛化性能。
        '''

        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:],
                                 epoch + batch_idx / num_iter, warm_up)

        # 正则化
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))
        #该正则化项可以理解为一个约束，用于迫使模型输出的类别概率分布尽可能接近均匀分布，防止模型对某个类别的过拟合。

        #通过这种正则化的方式，模型会被激励在训练过程中对所有类别保持较为均衡的输出，避免某些类别的输出概率过高而导致模型的偏差，
        # 特别是在数据分布不均匀或者数据较少的情况下，这种技术可以帮助模型更好地泛化。

        '''
        Lx：有监督损失，用于有标签数据。
        lamb * Lu：无监督损失，用一个权重 lamb 动态调节，确保无监督部分的逐渐引入。
        penalty：有时也会添加正则化项（例如在这里为了确保输出的分布不会偏向特定类别，加入了 penalty 项）
        '''
        loss = Lx + lamb * Lu + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            Lx.item(), Lu.item()))
        sys.stdout.flush()

'''
3. warmup 函数 —— 初始预热阶段训练
'''
def warmup(epoch, net, optimizer, dataloader):  #net：要训练的模型。
    net.train()  #设置模型为训练模式，以启用 dropout 和 batch normalization 的训练行为。
    #计算每个 epoch 中的总迭代次数，用于在输出中显示训练进度。
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)

        '''噪声模式处理'''
        #指定了噪声模式，通常有对称噪声 (sym) 和非对称噪声 (asym) 两种情况。
        if args.noise_mode == 'asym':  # 非对称噪声（asym），则对高置信度的预测样本添加惩罚项（penalty），以减少模型对噪声的过度拟合。
            penalty = conf_penalty(outputs)
            L = loss + penalty  #表示最终的损失，包含了基础损失和噪声处理的惩罚项。
        elif args.noise_mode == 'sym':
            L = loss

        '''反向传播和优化'''
        L.backward()
        optimizer.step()

        #进度显示
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()  #使用 sys.stdout.write 和 sys.stdout.flush 实现无换行的实时训练进度更新。

'''
4.test 函数 —— 测试模型精度
'''
def test(epoch, net1, net2):
    '''设置模型为评估模式'''
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()  #使用 .cuda() 将输入和标签移动到 GPU 上。
            '''模型前向传播'''
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            '''预测标签'''
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, acc))
    test_log.flush()

'''
5. eval_train 函数 —— 评估训练损失
'''
'''
这个 eval_train 函数用于在训练集上评估模型的表现，并对数据进行噪声标注的区分。
它使用高斯混合模型（GMM）来识别哪些样本的标签是可信的，哪些可能是噪声标签。
'''
def eval_train(model, all_loss):
    '''设置模型为评估模式'''
    model.eval()
    '''初始化损失数组 '''
    losses = torch.zeros(50000)
    '''计算每个样本的损失'''
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b] #将每个样本的损失值存储在对应的索引位置 losses[index[b]] 中。
    '''对所有样本的损失值进行归一化，将它们缩放到 [0, 1] 范围内，以便后续使用高斯混合模型进行处理。'''
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    '''将当前 epoch 的所有样本损失保存到 all_loss 列表中，便于后续分析和处理。'''
    all_loss.append(losses)
    '''平均损失（可选）'''
    if args.r == 0.9:  # 如果噪声比率 r 为 0.9，则计算最近 5 个 epoch 的平均损失，以提高模型的收敛稳定性。
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)  #将平均后的损失或当前的损失重塑为形状为 (50000, 1) 的张量，便于高斯混合模型处理。
    else:
        input_loss = losses.reshape(-1, 1)


    '''!!!!使用高斯混合模型!!!'''
    # fit a two-component GMM to the loss
    '''创建一个高斯混合模型 gmm，其中 n_components=2 表示使用两个高斯分布来对损失值进行拟合（一个分布通常用于表示干净样本，另一个用于表示噪声样本）。
    高斯混合模型（GMM）：通过拟合损失分布，将样本划分为两类（干净和噪声），并返回每个样本是干净样本的概率，用于后续训练。
    '''
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss) #使用 gmm.fit(input_loss) 拟合损失值。
    prob = gmm.predict_proba(input_loss) #计算每个样本属于这两个分布的概率，prob 包含每个样本属于两类的概率。
    prob = prob[:, gmm.means_.argmin()]  #通过 gmm.means_.argmin() 找到哪个分布的均值较小，通常均值较小的分布对应的是干净样本，然后选择对应的概率列，得到每个样本是干净样本的概率 prob。
    return prob, all_loss  #返回每个样本的干净概率 prob 和损失历史 all_loss，这些信息会用于下一阶段的训练中，以判断哪些样本是干净的，哪些样本是噪声样本。


'''
6. linear_rampup 函数 —— 线性增加权重

这个函数 linear_rampup 用于实现一种线性学习率的提升策略（线性增长阶段），
其目的是在训练的前期逐渐增加某些训练超参数（例如，无监督损失的权重），以防止模型在训练早期因为不稳定的梯度而产生不良影响
'''
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)

'''
7. SemiLoss 类 —— 半监督损失计算
'''
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        '''有标签损失 (Lx)  #交叉熵损失（Cross Entropy Loss）'''
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))  #这是有标签数据的交叉熵损失的核心计算部分，表示每个类别上的 log 概率与真实标签的乘积。
        '''无标签损失 (Lu)  #均方误差损失（Mean Squared Error Loss）'''
        Lu = torch.mean((probs_u - targets_u) ** 2)   #计算预测概率与伪标签之间的平方误差。这是均方误差（MSE），用于度量无标签样本预测的准确性。

        return Lx, Lu, linear_rampup(epoch, warm_up)  #动态调整无监督损失权重 (linear_rampup)

'''
8. NegEntropy 类 —— 负熵惩罚
作用：用于计算负熵，作为一种正则化方法来减小模型对噪声的敏感性，尤其在 asym 模式(在预热中使用）下增加置信度较高样本的惩罚。
'''
class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))

'''
9. create_model 函数 —— 模型创建函数
'''
def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

'''
10. 主函数 (包含在 if __name__ == '__main__': 中)
'''
if __name__ == '__main__':
    args = parser.parse_args()

    # Set GPU and random seed
    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Logging
    stats_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_stats.txt', 'w')
    test_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_acc.txt', 'w')

    # 设置 warm_up
    if args.dataset == 'cifar10':
        warm_up = 10
    elif args.dataset == 'cifar100':
        warm_up = 30

    # Load dataset
    loader = dataloader.cifar_dataloader(
        args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
        num_workers=5, root_dir=args.data_path, log=stats_log,
        noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode)
    )

    # Build and initialize models
    print('| Building net')
    net1 = create_model()
    net2 = create_model()
    cudnn.benchmark = True

    criterion = SemiLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)  #使用 SGD（随机梯度下降）作为优化器
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    if args.noise_mode == 'asym':
        conf_penalty = NegEntropy()

    all_loss = [[], []]  # save the history of losses from two networks

    # Training loop
    for epoch in range(args.num_epochs + 1):
        lr = args.lr
        if epoch >= 150:  #在训练的第 150 个 epoch 之后，学习率 lr 会被缩小到原来的 1/10，这是一种 学习率衰减策略，通常在训练达到一定阶段时减小学习率，可以让模型更加稳定地收敛。
            lr /= 10
        #optimizer1.param_groups 和 optimizer2.param_groups 中存储了优化器的参数组，通过直接修改 param_group['lr'] 可以动态调整学习率。
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr

        test_loader = loader.run('test')  #loader.run('test')：为测试数据创建数据加载器，用于测试模型在每个 epoch 结束后的准确率。
        eval_loader = loader.run('eval_train')  #为模型的训练数据创建数据加载器，用于评估训练过程中的损失，特别是在热身阶段或获取标签的置信度时。

        '''热身阶段（Warm-up Phase）'''
        if epoch < warm_up:
            warmup_trainloader = loader.run('warmup')
            print('Warmup Net1')
            warmup(epoch, net1, optimizer1, warmup_trainloader)
            print('\nWarmup Net2')
            warmup(epoch, net2, optimizer2, warmup_trainloader)
            '''主训练阶段'''
        else:
            prob1, all_loss[0] = eval_train(net1, all_loss[0])
            prob2, all_loss[1] = eval_train(net2, all_loss[1])

            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)

            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)
            train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)
            train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)

        test(epoch, net1, net2)