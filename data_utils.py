import random
import pandas as pd
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle


def negative(src, dataset, num, history, item_pool=None, neg_item=None):
    dataset = pd.DataFrame(np.repeat(dataset.values, num, axis=0), columns=dataset.columns)
    if src:
        item_set = set([i for i in range(1, 9566)])
    else:
        if item_pool:
            item_set = item_pool
        else:
            item_set = set([i for i in range(1, 6778)])

    uid = 0
    pos = []
    neg_list = []
    for index in tqdm.tqdm(range(len(dataset))):
        if index % num == 0:
            uid = dataset.iloc[index, 0]
            pos = history[uid]
            if neg_item:
                neg = neg_item[uid]
                neg_list = list(item_set - set(pos) - set(neg))
            else:
                neg_list = list(item_set - set(pos))
        else:
            dataset.iloc[index, 1] = random.sample(neg_list, 1)[0]
            dataset.iloc[index, 2] = 0

    return dataset


def split_data(ood, sparsity):
    book = pd.read_csv('dataset/Douban/book.csv')
    movie = pd.read_csv('dataset/Douban/movie.csv')
    implicit = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
    book['y'] = book['y'].map(implicit)
    movie['y'] = movie['y'].map(implicit)
    book = book[book.y == 1]
    movie = movie[movie.y == 1]
    history_tgt = {}
    history_src = {}
    for uid in set(book.uid.unique()):
        pos = book[(book.uid == uid) & (book.y == 1)].iid.values.tolist()
        history_tgt[uid] = pos
    for uid in set(movie.uid.unique()):
        pos = movie[(movie.uid == uid) & (movie.y == 1)].iid.values.tolist()
        history_src[uid] = pos

    train_tgt_raw = book.sample(frac=0.9, axis=0)
    test = book[~book.index.isin(train_tgt_raw.index)]
    train_tgt = train_tgt_raw.sample(frac=sparsity, axis=0)
    test = test[test.uid.isin(train_tgt.uid)]
    test = test[test.iid.isin(train_tgt.iid)]
    train_pn = negative(False, train_tgt, 3, history_tgt)
    train_src = negative(True, movie, 3, history_src)
    train_item = train_pn.iid.values.tolist()
    neg_dict = {}
    for uid in set(train_pn.uid.unique()):
        neg = train_pn[(train_pn.uid == uid) & (train_pn.y == 0)].iid.values.tolist()
        neg_dict[uid] = neg

    test_pn = negative(False, test, 100, history_tgt, set(train_item), neg_dict)

    train_pn.to_csv('dataset/Douban/ready/train_tgt.csv', sep=',', index=False)
    train_src.to_csv('dataset/Douban/ready/train_src.csv', sep=',', index=False)
    test_pn.to_csv('dataset/Douban/ready/test.csv', sep=',', index=False)

    user_list = []
    for user in set(test.uid.unique()):
        if train_tgt[train_tgt.uid == user].iid.size >= 100:
            user_list.append(user)
    test_ood = test[test.uid.isin(set(user_list))]
    test_pn_ood = negative(False, test_ood, 100, history_tgt, set(train_item), neg_dict)
    test_pn_ood.to_csv('dataset/Douban/ready/test_ood.csv', sep=',', index=False)
    if ood:
        pool = test[~test.uid.isin(set(user_list))]
        len_ood = len(test_ood)
        test_55 = pool.sample(n=len_ood)
        test_64 = pool.sample(n=round(len_ood * 2 / 3))
        test_73 = pool.sample(n=round(len_ood * 3 / 7))
        test_ood_55 = pd.concat([test_ood, test_55], axis=0, ignore_index=True)
        test_ood_64 = pd.concat([test_ood, test_64], axis=0, ignore_index=True)
        test_ood_73 = pd.concat([test_ood, test_73], axis=0, ignore_index=True)
        test_pn_ood_55 = negative(False, test_ood_55, 100, history_tgt, set(train_item), neg_dict)
        test_pn_ood_64 = negative(False, test_ood_64, 100, history_tgt, set(train_item), neg_dict)
        test_pn_ood_73 = negative(False, test_ood_73, 100, history_tgt, set(train_item), neg_dict)
        test_pn_ood_55.to_csv('dataset/Douban/ready/test_ood_55.csv', sep=',', index=False)
        test_pn_ood_64.to_csv('dataset/Douban/ready/test_ood_64.csv', sep=',', index=False)
        test_pn_ood_73.to_csv('dataset/Douban/ready/test_ood_73.csv', sep=',', index=False)


def load_data():
    train_tgt = pd.read_csv('dataset/Douban/ready/train_tgt.csv')
    train_src = pd.read_csv('dataset/Douban/ready/train_src.csv')

    train_tgt = shuffle(train_tgt)
    train_src = shuffle(train_src)

    return train_tgt, train_src


def resample(batch_size, train_tgt, train_src):
    user_tgt = torch.tensor(train_tgt['uid'].values.tolist(), dtype=torch.int64) - 1
    item_tgt = torch.tensor(train_tgt['iid'].values.tolist(), dtype=torch.int64) - 1
    rating_tgt = torch.tensor(train_tgt['y'].values.tolist(), dtype=torch.int64)

    train_sample = train_src.sample(n=len(user_tgt), axis=0)
    user_src = torch.tensor(train_sample['uid'].values.tolist(), dtype=torch.int64) - 1
    item_src = torch.tensor(train_sample['iid'].values.tolist(), dtype=torch.int64) - 1
    rating_src = torch.tensor(train_sample['y'].values.tolist(), dtype=torch.int64)

    dataset = TensorDataset(user_src, user_tgt, item_src, item_tgt, rating_src, rating_tgt)
    data_iter = DataLoader(dataset, batch_size, shuffle=True)

    return data_iter


def load_test(ood):
    test = pd.read_csv('dataset/Douban/ready/test.csv')
    data_iter = get_loader(test)

    test_ood = pd.read_csv('dataset/Douban/ready/test_ood.csv')
    data_iter_ood = get_loader(test_ood)

    if ood:
        test_ood_55 = pd.read_csv('dataset/Douban/ready/test_ood_55.csv')
        data_iter_ood_55 = get_loader(test_ood_55)

        test_ood_64 = pd.read_csv('dataset/Douban/ready/test_ood_64.csv')
        data_iter_ood_64 = get_loader(test_ood_64)

        test_ood_73 = pd.read_csv('dataset/Douban/ready/test_ood_73.csv')
        data_iter_ood_73 = get_loader(test_ood_73)
        return data_iter, data_iter_ood, data_iter_ood_55, data_iter_ood_64, data_iter_ood_73

    return data_iter, data_iter_ood


def get_loader(test):
    user = torch.tensor(test['uid'].values.tolist(), dtype=torch.int64) - 1
    item = torch.tensor(test['iid'].values.tolist(), dtype=torch.int64) - 1
    dataset = TensorDataset(user, item)
    data_iter = DataLoader(dataset, 100, shuffle=False)
    return data_iter
