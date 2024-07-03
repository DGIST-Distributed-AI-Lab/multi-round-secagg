#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 40, 1500
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()


    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    #print('idxs:',np.shape(idxs))


    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        #print(i,rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_noniid2(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 120, 500
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()


    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    #print('idxs:',np.shape(idxs))


    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        #print(i,rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users, num_class_per_user=1):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users*num_class_per_user, int(50000/(num_users*num_class_per_user))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    #print('idxs:',np.shape(idxs))


    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_class_per_user, replace=False))
        #print(i,rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def cifar_noniid_shared(dataset, num_users, num_class_per_user=1):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :each user has 1) shared dataset + 2) one class
    """
    
    #num_shards, num_imgs = num_users*num_class_per_user, int(50000/(num_users*num_class_per_user))

    num_shards = num_users
    
    num_imgs_shared = int(50000 * 0.004)
    num_imgs   = int((50000 - num_imgs_shared)/(num_users*num_class_per_user))
    
    num_class = 10
    num_users_per_class = int(num_users/num_class)

    print("number of non-shared images=",num_imgs)

    idx_shard = [i for i in range(num_shards)]
    
    idxs = np.arange(50000)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))

    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]

    idxs = idxs_labels[0,:]
    #print('idxs:',np.shape(idxs))

    dict_shares = np.array([], dtype='int64')
    for i in range(num_class):
        stt_pos = (i+1)*5000 - int(num_imgs_shared / num_class)
        end_pos = (i+1)*5000

        #print(stt_pos,end_pos)
        #print(idxs_labels[1,stt_pos:end_pos])
        dict_shares = np.concatenate((dict_shares, idxs[stt_pos:end_pos]), axis=0)

    print("number of shared images=",len(dict_shares))

    # divide and assign
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        #print(i,rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            i_ = rand % num_users_per_class
            j_ = int((rand-i_)/num_users_per_class)

            

            stt_pos = j_ * 5000 + i_ * num_imgs
            end_pos = j_ * 5000 + (i_+1) * num_imgs

            #print(rand, j_,i_, end_pos-stt_pos)

            dict_users[i] = np.concatenate((dict_users[i], idxs[stt_pos:end_pos]), axis=0)

        dict_users[i] = np.concatenate((dict_users[i], dict_shares), axis=0)

    print("number of images per user=",len(dict_users[0]))

    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
