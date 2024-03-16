import numpy as np
import torch


def hit(item, pre_item):
    if item in pre_item:
        return 1
    return 0


def ndcg(item, pre_item):
    if item in pre_item:
        index = pre_item.index(item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(net, data_loader):
    HR5, HR10 = 0, 0
    NDCG5, NDCG10 = 0, 0
    softmax = torch.nn.Softmax(dim=1)
    net.threshold()

    batch = 0
    for x, y in data_loader:
        batch += 1
        score = net.predict(x, y)
        value5, index5 = torch.topk(softmax(score)[:, 1], 5)
        value10, index10 = torch.topk(softmax(score)[:, 1], 10)

        if 0 in index10:
            HR10 += 1
            pos = index10.tolist().index(0) + 2
            NDCG10 += np.reciprocal(np.log2(pos))
            if 0 in index5:
                HR5 += 1
                pos = index5.tolist().index(0) + 2
                NDCG5 += np.reciprocal(np.log2(pos))

    return HR5 / batch, HR10 / batch, NDCG5 / batch, NDCG10 / batch
