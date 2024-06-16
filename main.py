import torch
import torch.nn as nn
import datetime
import sys
import argparse

import model
import data_utils
import data_utils_T
import evaluate
import function

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=7, help="training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
parser.add_argument("--latent_dim", type=int, default=16, help="latent dim")
parser.add_argument("--ood", type=bool, default=False, help="whether use ood data")
parser.add_argument("--wocausal", type=bool, default=False, help="without causal")
parser.add_argument("--wosource", type=bool, default=False, help="without source")
parser.add_argument("--source", type=float, default=1.0, help="")
parser.add_argument("--dis", type=float, default=0.25, help="")
parser.add_argument("--sparsity", type=float, default=1.0, help="")
parser.add_argument("--dataset", type=str, default="douban", help="")
args = parser.parse_args()

curr_time = datetime.datetime.now()
timestamp = curr_time.strftime('%m-%d %H-%M')
sys.stdout = function.Logger('dataset/'+args.dataset+'/ready/' + timestamp + '.log', sys.stdout)

print('is_ood:{}'.format(args.ood))
print('batch_size:{},lr:{},latent_dim:{},sparsity:{}'.format(args.batch_size, args.lr, args.latent_dim, args.sparsity))
print('source:{},dis:{}'.format(args.source, args.dis))
if args.dataset=='douban':
    data_utils.split_data(args.ood, args.sparsity)
    train_tgt, train_src = data_utils.load_data()
    if args.ood:
        test, test_ood1, test_ood2, test_ood1_55, test_ood1_64, test_ood1_73, test_ood2_64, test_ood2_46, test_ood2_28 = data_utils.load_test(args.ood)
    else:
        test, test_ood1, test_ood2 = data_utils.load_test(args.ood)

    net = model.CDCOR(2106, 9555, 6777, args.latent_dim)
else:
    data_utils_T.split_data(args.ood, args.sparsity)
    train_tgt, train_src = data_utils_T.load_data()
    if args.ood:
        test, test_ood1, test_ood2, test_ood1_55, test_ood1_64, test_ood1_73, test_ood2_64, test_ood2_46, test_ood2_28 = data_utils_T.load_test(args.ood)
    else:
        test, test_ood1, test_ood2 = data_utils_T.load_test(args.ood)

    net = model.CDCOR(2151, 37041, 2869, args.latent_dim)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.wd)

for epoch in range(args.epoch):
    net.train()
    if args.dataset=='douban':
        train_loader = data_utils.resample(args.batch_size, train_tgt, train_src)
    else:
        train_loader = data_utils_T.resample(args.batch_size, train_tgt, train_src)
    if args.wocausal:
        net.adj_matrix.requires_grad = False
    elif args.wosource:
        net.adj_matrix.requires_grad = True
    else:
        if epoch < 3:
            net.adj_matrix.requires_grad = False
        else:
            net.adj_matrix.requires_grad = True

    losses = 0.0
    losses_s = 0.0
    losses_t = 0.0
    losses_ds = 0.0
    losses_dt = 0.0
    losses_cau = 0.0
    for i, (s_u, t_u, s_i, t_i, s_l, t_l) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        if args.wocausal:
            if epoch < 3:
                output = net.preforward(s_u, t_u, s_i, t_i)
            else:
                output = net(s_u, t_u, s_i, t_i)
        elif args.wosource:
            output = net.preforward(s_u, t_u, s_i, t_i)
        else:
            if epoch < 3:
                output = net.preforward(s_u, t_u, s_i, t_i)
            else:
                output = net(s_u, t_u, s_i, t_i)
        loss_s = loss_function(output[0], s_l)
        loss_t = loss_function(output[1], t_l)
        s_domain_label = torch.zeros(len(s_l), dtype=torch.int64)
        t_domain_label = torch.ones(len(t_l), dtype=torch.int64)
        loss_ds = loss_function(output[2], s_domain_label)
        loss_dt = loss_function(output[3], t_domain_label)
        loss_cau = function.causal_loss(output[4], output[5], output[6], output[7], output[8], args.wosource)
        if args.wocausal:
            if epoch < 3:
                loss = args.source * loss_s + loss_t + args.dis * loss_ds + args.dis * loss_dt
            else:
                loss = args.source * loss_s + loss_t + args.dis * loss_ds + args.dis * loss_dt
        elif args.wosource:
            loss = loss_t + loss_cau
        else:
            if epoch < 3:
                loss = args.source * loss_s + loss_t + args.dis * loss_ds + args.dis * loss_dt
            else:
                loss = args.source * loss_s + loss_t + args.dis * loss_ds + args.dis * loss_dt + loss_cau
        loss.backward()
        optimizer.step()
        losses += loss.item()
        losses_s += loss_s.item()
        losses_t += loss_t.item()
        losses_ds += loss_ds.item()
        losses_dt += loss_dt.item()
        losses_cau += loss_cau.item()
        if i % 100 == 0:
            print('epoch=', epoch + 1, 'batch=', i, 'loss=', losses / i,
                  'loss_s=', losses_s / i, 'loss_t=', losses_t / i)
            print('loss_ds=', losses_ds / i, 'loss_dt=',
                  losses_dt / i, 'loss_cau=', losses_cau / i, '\n')

    net.eval()
    HR5, HR10, NDCG5, NDCG10 = evaluate.metrics(net, test)
    print('HR@5=', HR5, 'HR@10=', HR10, 'NDCG@5=', NDCG5, 'NDCG@10=', NDCG10)
    HR5, HR10, NDCG5, NDCG10 = evaluate.metrics(net, test_ood1)
    print('HR@5=', HR5, 'HR@10=', HR10, 'NDCG@5=', NDCG5, 'NDCG@10=', NDCG10)
    HR5, HR10, NDCG5, NDCG10 = evaluate.metrics(net, test_ood2)
    print('HR@5=', HR5, 'HR@10=', HR10, 'NDCG@5=', NDCG5, 'NDCG@10=', NDCG10)
    if args.ood:
        HR5, HR10, NDCG5, NDCG10 = evaluate.metrics(net, test_ood1_55)
        print('HR@5=', HR5, 'HR@10=', HR10, 'NDCG@5=', NDCG5, 'NDCG@10=', NDCG10)
        HR5, HR10, NDCG5, NDCG10 = evaluate.metrics(net, test_ood1_64)
        print('HR@5=', HR5, 'HR@10=', HR10, 'NDCG@5=', NDCG5, 'NDCG@10=', NDCG10)
        HR5, HR10, NDCG5, NDCG10 = evaluate.metrics(net, test_ood1_73)
        print('HR@5=', HR5, 'HR@10=', HR10, 'NDCG@5=', NDCG5, 'NDCG@10=', NDCG10)
        HR5, HR10, NDCG5, NDCG10 = evaluate.metrics(net, test_ood2_64)
        print('HR@5=', HR5, 'HR@10=', HR10, 'NDCG@5=', NDCG5, 'NDCG@10=', NDCG10)
        HR5, HR10, NDCG5, NDCG10 = evaluate.metrics(net, test_ood2_46)
        print('HR@5=', HR5, 'HR@10=', HR10, 'NDCG@5=', NDCG5, 'NDCG@10=', NDCG10)
        HR5, HR10, NDCG5, NDCG10 = evaluate.metrics(net, test_ood2_28)
        print('HR@5=', HR5, 'HR@10=', HR10, 'NDCG@5=', NDCG5, 'NDCG@10=', NDCG10)
    print('###########################################################################', '\n')

torch.save(net.state_dict(), 'dataset/'+args.dataset+'/ready/model.pth')

