import time
import os
import numpy as np
from event_utils.utils import *
from event_utils.dataset import DVS_Lip
from event_utils.mixup import mixup_data, mixup_criterion
from event_utils.label_smooth import LSR

from model.model.model import MTGA
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from warnings import filterwarnings
filterwarnings('ignore')
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

def test(args, net):
    with torch.no_grad():
        dataset = DVS_Lip('test', args, mode='test')
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)        
        v_acc = []
        label_pred = {i: [] for i in range(args.n_class)}
        net.eval()
        for data,voxel in tqdm(loader):
            data = {k: data[k].cuda(non_blocking=True) for k in data}
            input_data = {
                'event_frame': data['event_frame'],
            }

            label = data.get('label').long()

            with autocast():
                logit = net(input_data,voxel)

            v_acc.extend((logit.argmax(-1) == label).cpu().numpy().tolist())
            label_list = label.cpu().numpy().tolist()
            pred_list = logit.argmax(-1).cpu().numpy().tolist()
            for i in range(len(label_list)):
                label_pred[label_list[i]].append(pred_list[i])

        acc_p1, acc_p2 = compute_each_part_acc(label_pred)
        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'test acc: {:.5f}, acc part1: {:.5f}, acc part2: {:.5f}'.format(acc, acc_p1, acc_p2)
        return acc, acc_p1, acc_p2, msg

def train(args, net, optimizer, log_dir, scheduler):
    train_res = {
        'best_epoch': 0,
        'best_acc': 0,
        'each_acc': [],
        'finished': False
    }
    dataset = DVS_Lip('train', args, mode='train')
    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)

    best_acc, best_acc_p1, best_acc_p2 = 0.0, 0.0, 0.0


    scaler = GradScaler()


    # import time
    for epoch in range(args.max_epoch):
        time.sleep(0.3)
        print('epoch: {}'.format(epoch))
        time.sleep(0.3)



        if args.label_smooth:
            criterion = LSR()
        else:
            criterion = nn.CrossEntropyLoss()

        net.train()
        i_iter = -1

        k = time.time()
        loss_fn = torch.nn.CrossEntropyLoss()
        for data,voxel in tqdm(loader):
            i_iter += 1
            t = time.time()
            data = {k: data[k].cuda(non_blocking=True) for k in data}

            label = data.get('label').long()

            data = {
                'event_frame': data.get('event_frame'),
            }
            data, labels_a, labels_b, lam = mixup_data(x=data, y=label, alpha=args.mixup, use_cuda=True)

            input_data = {
                'epoch':epoch,
                'label':label,
                'event_frame': data['event_frame'],
            }


            t_inf = time.time()
            loss = {}
            

            with autocast():
                logit = net(input_data,voxel)
                loss_func = mixup_criterion(labels_a, labels_b, lam)
                loss_bp = loss_func(criterion, logit)
                

            loss['Total'] = loss_bp
            optimizer.zero_grad()
            scaler.scale(loss_bp).backward()  
            scaler.step(optimizer)
            scaler.update()
            
            k = time.time()

           

        epoch_acc = {
            'epoch': epoch
        }
        time.sleep(0.3)
        if args.test_train:
            acc_tt, acc_p1_tt, acc_p2_tt, msg_tt = test_train(args, net)
            epoch_acc['test_train_acc'] = '{:.5f}'.format(acc_tt)
            epoch_acc['test_train_acc1'] = '{:.5f}'.format(acc_p1_tt)
            epoch_acc['test_train_acc2'] = '{:.5f}'.format(acc_p2_tt)
            print(msg_tt)

        acc, acc_p1, acc_p2, msg = test(args, net)
        epoch_acc['test_test_acc'] = '{:.5f}'.format(acc)
        epoch_acc['test_test_acc1'] = '{:.5f}'.format(acc_p1)
        epoch_acc['test_test_acc2'] = '{:.5f}'.format(acc_p2)
        print(msg)
       


        print('acc best: {}\n'.format(best_acc))
        time.sleep(0.3)
        train_res['each_acc'].append(epoch_acc)
        scheduler.step()

    train_res['finished'] = 1

   


                


def fun(gpu=1,lr=3e-4, batch_size=32, optimizer='Adam', n_class=100, seq_len=30,
        num_workers=1, max_epoch=240, num_bins=1, log_dir='result', exp_name='log',
        event_root='../DVS-Lip',  voxel_graph_root='../DVS-Lip-Voxelgraph-0310-1',mixup=0.4, speech_speed_var=0, label_smooth=True,
        back_type='GRU', se=False, base_channel=64, net_type='single',
        alpha=4, beta=7, t2s_mul=2, word_boundary=False
        ):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', type=int, default=gpu)

    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--optimizer', type=str, default=optimizer)
    parser.add_argument('--n_class', type=int, default=n_class)
    parser.add_argument('--seq_len', type=int, default=seq_len)
    parser.add_argument('--num_workers', type=int, required=False, default=num_workers)
    parser.add_argument('--max_epoch', type=int, required=False, default=max_epoch)
    parser.add_argument('--num_bins', type=int, required=False, default=num_bins)  
    parser.add_argument('--log_dir', type=str, required=False, default=log_dir)
    parser.add_argument('--exp_name', type=str, default=exp_name)


    # 新增体素  
    parser.add_argument('--voxel_graph_root', type=str, default='../DVS-Lip-Voxelgraph-0310-1')
    
    parser.add_argument('--test_train', type=bool, default=False)
    parser.add_argument('--event_root', type=str, default=event_root)

    parser.add_argument('--speech_speed_var', type=float, default=speech_speed_var)
    parser.add_argument('--word_boundary', type=bool, default=word_boundary)
    parser.add_argument('--mixup', type=float, default=mixup)
    parser.add_argument('--label_smooth', type=bool, default=label_smooth)

    parser.add_argument('--back_type', type=str, default=back_type)
    parser.add_argument('--se', type=str2bool, default=se)
    parser.add_argument('--base_channel', type=int, default=base_channel)
    parser.add_argument('--net_type', type=str, default=net_type)

    parser.add_argument('--alpha', type=int, default=alpha)
    parser.add_argument('--beta', type=int, default=beta)
    parser.add_argument('--t2s_mul', type=int, default=t2s_mul)


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    net = MTGA(args).cuda()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'AdamW':# 2% lower
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=5e-6)
    net = nn.DataParallel(net)

    train(args, net, optimizer, log_dir, scheduler)
    


if __name__ == '__main__':
    fun()
