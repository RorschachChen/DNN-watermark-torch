import numpy as np
import sys
import json
import os
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from models.wide_resnet import Wide_ResNet
from utils import random_index_generator

RESULT_PATH = './result'


def get_layer_by(model, target_blk_num):
    return model.get_parameter(f'layer{target_blk_num}.0.conv2.weight')


def train(model, optimizer, dataloader, w, b, k, nb_epoch, target_blk_num):
    model.train()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    device = next(model.parameters()).device
    b = torch.tensor(b, dtype=torch.float32).to(device)
    w = torch.tensor(w, dtype=torch.float32).to(device)
    for ep in tqdm(range(nb_epoch)):
        for d, t in dataloader:
            d = d.to(device)
            t = t.to(device)
            optimizer.zero_grad()
            pred = model(d)
            loss = criterion(pred, t)
            regularized_loss = 0
            if target_blk_num > 0:
                p = get_layer_by(model, target_blk_num)
                x = torch.mean(p, dim=0)
                y = x.view(1, -1)
                regularized_loss = k * torch.sum(
                    F.binary_cross_entropy(input=torch.sigmoid(torch.matmul(y, w)), target=b))
            (loss + regularized_loss).backward()
            optimizer.step()


def build_wm(model, target_blk_num, embed_dim, wtype):
    if target_blk_num == 0:
        return np.array([])
    # get param
    p = get_layer_by(model, target_blk_num)
    w_rows = p.size()[1:4].numel()
    w_cols = embed_dim
    if wtype == 'random':
        w = np.random.randn(w_rows, w_cols)
    elif wtype == 'direct':
        w = np.zeros((w_rows, w_cols), dtype=None)
        rand_idx_gen = random_index_generator(w_rows)

        for col in range(w_cols):
            w[next(rand_idx_gen)][col] = 1.
    elif wtype == 'diff':
        w = np.zeros((w_rows, w_cols), dtype=None)
        rand_idx_gen = random_index_generator(w_rows)

        for col in range(w_cols):
            w[next(rand_idx_gen)][col] = 1.
            w[next(rand_idx_gen)][col] = -1.
    else:
        raise Exception('wtype="{}" is not supported'.format(wtype))
    return w


def save_wmark_signatures(prefix, target_blk_num, w, b):
    fname_w = prefix + '_layer{}_w.npy'.format(target_blk_num)
    fname_b = prefix + '_layer{}_b.npy'.format(target_blk_num)
    np.save(fname_w, w)
    np.save(fname_b, b)


if __name__ == '__main__':
    device = torch.device('cuda')
    settings_json_fname = sys.argv[1]
    train_settings = json.load(open(settings_json_fname))
    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    # read parameters
    batch_size = train_settings['batch_size']
    nb_epoch = train_settings['epoch']
    scale = train_settings['scale']
    embed_dim = train_settings['embed_dim']
    N = train_settings['N']
    k = train_settings['k']
    target_blk_id = train_settings['target_blk_id']
    base_modelw_fname = train_settings['base_modelw_fname']
    wtype = train_settings['wmark_wtype']
    randseed = train_settings['randseed'] if 'randseed' in train_settings else 'none'
    hist_hdf_path = 'WTYPE_{}/DIM{}/SCALE{}/N{}K{}B{}EPOCH{}/TBLK{}'.format(
        wtype, embed_dim, scale, N, k, batch_size, nb_epoch, target_blk_id)
    modelname_prefix = os.path.join(RESULT_PATH, 'wrn_' + hist_hdf_path.replace('/', '_'))
    # load dataset for learning
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    # initialize process for Watermark
    b = np.ones((1, embed_dim))
    model = Wide_ResNet(10, 4)
    model = model.to(device)

    # build wm
    w = build_wm(model, target_blk_id, embed_dim, wtype)

    # training process
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 160], gamma=0.2)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if len(base_modelw_fname) > 0:
        model.load_state_dict(torch.load(base_modelw_fname))
    print("Finished building")

    train(model, optimizer, trainloader, w, b, k, nb_epoch, target_blk_id)

    # validate training accuracy
    model.eval()
    loss_meter = 0
    acc_meter = 0
    with torch.no_grad():
        for d, t in testloader:
            data = d.to(device)
            target = t.to(device)
            pred = model(data)
            loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
            pred = pred.max(1, keepdim=True)[1]
            acc_meter += pred.eq(target.view_as(pred)).sum().item()
    print('Test loss:', loss_meter)
    print('Test accuracy:', acc_meter / len(testloader.dataset))
    # write model parameters to file
    torch.save(model.state_dict(), modelname_prefix + '.pth')
    # write watermark matrix and embedded signature to file
    if target_blk_id > 0:
        save_wmark_signatures(modelname_prefix, target_blk_id, w, b)
