# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import utils
import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
# import ipdb
import numpy as np
import wideresnet
import json
# Sampling
from tqdm import tqdm  # tqdm是 Python 进度条库，可以在 Python长循环中添加一个进度提示信息。用户只需要封装任意的迭代器，是一个快速、扩展性强的进度条工具库。


t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
im_sz = 32  # cifar 彩色图片的size update: dspirte = 64
n_ch = 1  # RGB通道数  update: dspirte = 1


class DataSubset(Dataset):  # 取数据子集
    def __init__(self, base_dataset, inds=None, size=-1):  # 从dataset中随机取子集
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)  # 在base_dataset中取size个数,允许重复
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


class F(nn.Module):  # F
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=3):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)  # 一般用28x10
        self.energy_output = nn.Linear(self.f.last_dim, 1)   # 能量输出
        self.class_output = nn.Linear(self.f.last_dim, n_classes)  # 传统分类输出

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


class CCF(F):  # F 条件输出
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=3):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate, n_classes=n_classes)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])


def cycle(loader):
    while True:
        for data in loader:
            yield data


def grad_norm(m):
    total_norm = 0
    for p in m.parameters():
        param_grad = p.grad
        if param_grad is not None:
            param_norm = param_grad.data.norm(2) ** 2
            total_norm += param_norm
    total_norm = total_norm ** (1. / 2)
    return total_norm.item()


def grad_vals(m):
    ps = []
    for p in m.parameters():
        if p.grad is not None:
            ps.append(p.grad.data.view(-1))
    ps = t.cat(ps)
    return ps.mean().item(), ps.std(), ps.abs().mean(), ps.abs().std(), ps.abs().min(), ps.abs().max()


def init_random(args, bs):  # bs = batch size
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def get_model_and_buffer(args, device, sample_q): # 设置model 以及 对replay_buffer进行一个随机初始化
    model_cls = F if args.uncond else CCF
    f = model_cls(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size)
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]

    f = f.to(device)
    return f, replay_buffer


class DSpritesDataset(Dataset):
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        subsample : int Only load every |subsample| number of images.
        """
        data = np.load(path_to_data)
        # subsample 这个语句的意思就是以多少步长取数据，-1就是倒序取数
        self.imgs = data['imgs'][::subsample]
        self.labels = data['latents_classes'][::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):  # idx就是第几张图片，即index
        label = self.labels[idx][1]
        # Each image in the dataset has binary values so multiply by 255 to get pixel values
        sample = self.imgs[idx] * 255
        # Add extra dimension to turn shape into (H, W) -> (H, W, C)
        sample = sample.reshape(sample.shape + (1,))
        if self.transform:
            sample = self.transform(sample)

        return (sample, label)


def get_data(args):
    # 选取 transform ,进行一个数据增强
    transform_train = tr.Compose(
        [tr.ToPILImage(),
         tr.Resize(32),
         # tr.Pad(4, padding_mode="reflect"),
         # tr.RandomCrop(32),
         # tr.RandomHorizontalFlip(),
         tr.ToTensor(),
         tr.Normalize((.5), (.5)),
         lambda x: x + args.sigma * t.randn_like(x)]
    )

    transform_test = tr.Compose(
        [tr.ToPILImage(),
         tr.Resize(32),
         tr.ToTensor(),
         tr.Normalize((.5), (.5)),
         lambda x: x + args.sigma * t.randn_like(x)]
    )

    # get all training inds
    full_train = DSpritesDataset(path_to_data=args.data_root, transform=transform_train, subsample=1)
    full_test = DSpritesDataset(path_to_data=args.data_root, transform=transform_test, subsample=1)

    all_inds = list(range(len(full_train)))

    # set seed 设置种子
    np.random.seed(1234)

    # shuffle  混洗
    np.random.shuffle(all_inds)

    # seperate out validation set 在训练集中分出验证集
    # 这里100000是暂定 总共为737280 0.75/0.167/0.83   552960/122880/61440
    valid_inds, train_inds, test_inds = all_inds[:61400], all_inds[61400:552980], all_inds[552980:]
    train_inds = np.array(train_inds)
    train_labeled_inds = []
    other_inds = []
    train_labels = np.array([full_train[ind][1] for ind in train_inds])

    # 建立对应标签集
    if args.labels_per_class > 0:  # 每一类的标记样本数>0
        for i in range(args.n_classes):
            print(i)
            train_labeled_inds.extend(train_inds[train_labels == i][:args.labels_per_class])
            other_inds.extend(train_inds[train_labels == i][args.labels_per_class:])
    else:
        train_labeled_inds = train_inds

    # 分别取未标记子集，标记子集，测试子集
    dset_train = DataSubset(full_train, inds=train_inds)
    dset_train_labeled = DataSubset(full_train, inds=train_labeled_inds)
    dset_valid = DataSubset(full_test, inds=valid_inds)
    dset_test = DataSubset(full_test, inds=test_inds)  # 这里是需要分出不同的图片的一个训练集，原为 dataset_fn(False, transform_test)

    # dataloader
    dload_train = DataLoader(dset_train, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)

    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=4, drop_last=True)  # 这里为啥改成了size=100.然后drop_last=1(可能为整倍数)

    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=4, drop_last=True)
    return dload_train, dload_train_labeled, dload_valid, dload_test


def get_sample_q(args, device):
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(args, bs), []
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
        inds = t.randint(0, buffer_size, (bs,))

        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        buffer_samples = replay_buffer[inds]
        random_samples = init_random(args, bs)
        choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None, None, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(device), inds

    def sample_q(f, replay_buffer, y=None, n_steps=args.n_steps):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.R
        """
        f.eval()
        # get batch size
        bs = args.batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
            x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k)
        f.train()
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples

    return sample_q


def eval_classification(f, dload, device):
    corrects, losses = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        logits = f.classify(x_p_d)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


def checkpoint(f, buffer, tag, args, device):  # 节点检测
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "replay_buffer": buffer
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)


def main(args):  # 主程序
    # 保存路径，参数以及日志
    utils.makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    # seed
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    # datasets and device
    dload_train, dload_train_labeled, dload_valid, dload_test = get_data(args)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    # get_sample_q
    sample_q = get_sample_q(args, device)
    f, replay_buffer = get_model_and_buffer(args, device, sample_q)

    # 2 lambda function to save image
    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    # optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    # 记录
    best_valid_acc = 0.0  # 该epoch的最佳valid_acc
    cur_iter = 0  # 当前iteration

    # train process
    for epoch in range(args.n_epochs):
        # <1> decay_epochs
        # : decay learning rate by decay_rate at these epochs
        # default=[160, 180]
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * args.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))

        # <2> major
        for i, (x_p_d, _) in tqdm(enumerate(dload_train)):  # tqdm 进度条 enumerate 遍历
            # warmup_iters = 1000
            if cur_iter <= args.warmup_iters:
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            # .to(device) 的意思就是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
            x_p_d = x_p_d.to(device)
            x_lab, y_lab = dload_train_labeled.__next__()  # 为什么要取下一个？？
            x_lab, y_lab = x_lab.to(device), y_lab.to(device)

            L = 0.  # loss

            # (1) maximize log p(x)
            if args.p_x_weight > 0:
                if args.class_cond_p_x_sample:
                    assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                    y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                    x_q = sample_q(f, replay_buffer, y=y_q)
                else:
                    x_q = sample_q(f, replay_buffer)  # sample from log-sumexp

                fp_all = f(x_p_d)
                fq_all = f(x_q)
                fp = fp_all.mean()
                fq = fq_all.mean()
                l_p_x = -(fp - fq)

                # 输出阶段性结果
                if cur_iter % args.print_every == 0:
                    print('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                   fp - fq))
                L += args.p_x_weight * l_p_x

            # (2) maximize log p(y | x)
            if args.p_y_given_x_weight > 0:
                logits = f.classify(x_lab)
                l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)
                if cur_iter % args.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,
                                                                                 cur_iter,
                                                                                 l_p_y_given_x.item(),
                                                                                 acc.item()))
                L += args.p_y_given_x_weight * l_p_y_given_x

            # (3) maximize log p(x, y)
            if args.p_x_y_weight > 0:
                assert not args.uncond, "this objective can only be trained for class-conditional EBM DUUUUUUUUHHHH!!!"
                x_q_lab = sample_q(f, replay_buffer, y=y_lab)
                fp, fq = f(x_lab, y_lab).mean(), f(x_q_lab, y_lab).mean()
                l_p_x_y = -(fp - fq)
                if cur_iter % args.print_every == 0:
                    print('P(x, y) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                      fp - fq))

                L += args.p_x_y_weight * l_p_x_y

            # Error break  = loss diverged
            # break if the loss diverged...easier for poppa to run experiments this way
            if L.abs().item() > 1e8:
                print("BAD BOIIIIIIIIII")
                1 / 0

            optim.zero_grad()   # 将梯度值归零
            L.backward()        # 反向传播计算得到每个参数的梯度值
            optim.step()        # 通过梯度下降执行一步参数更新
            cur_iter += 1

            # 保存样本
            if cur_iter % 100 == 0:
                if args.plot_uncond:
                    if args.class_cond_p_x_sample:
                        assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                        y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                        x_q = sample_q(f, replay_buffer, y=y_q)
                    else:
                        x_q = sample_q(f, replay_buffer)
                    plot('{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)

                if args.plot_cond:  # generate class-conditional samples
                    y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(
                        -1).to(device)
                    x_q_y = sample_q(f, replay_buffer, y=y)
                    plot('{}/x_q_y{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q_y)

        # <3> checkpoint
        if epoch % args.ckpt_every == 0:
            checkpoint(f, replay_buffer, f'ckpt_{epoch}.pt', args, device)

        # <4> eval_point
        if epoch % args.eval_every == 0 and (args.p_y_given_x_weight > 0 or args.p_x_y_weight > 0):
            f.eval()
            with t.no_grad():
                # validation set
                correct, loss = eval_classification(f, dload_valid, device)
                print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, replay_buffer, "best_valid_ckpt.pt", args, device)
                # test set
                correct, loss = eval_classification(f, dload_test, device)
                print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
            f.train()
        checkpoint(f, replay_buffer, "last_ckpt.pt", args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models on Dsprites")
    # parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100"])
    parser.add_argument("--data_root", type=str, default="/content/drive/MyDrive/data/dsprites.npz")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)  # **
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--p_x_weight", type=float, default=1.)
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    parser.add_argument("--p_x_y_weight", type=float, default=0.)
    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2,   # **
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"],
                        help="norm to add to weights, none works fine")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=120000)
    parser.add_argument("--reinit_freq", type=float, default=.05)  # ** ?
    parser.add_argument("--sgld_lr", type=float, default=1.0)  # **
    parser.add_argument("--sgld_std", type=float, default=1e-2) # **
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=10, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
    # parser.add_argument("--n_valid", type=int, default=5000)

    args = parser.parse_args()
    args.n_classes = 3  # 根据dataset设定分类数
    main(args)  # 调用训练主体过程
