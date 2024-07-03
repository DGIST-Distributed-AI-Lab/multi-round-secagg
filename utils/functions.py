import numpy as np
import random
import math
import copy
import torch
from torch import nn

import itertools

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def rref(A, tol=1.0e-12):
    m, n = A.shape
    i, j = 0, 0
    jb = []

    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j
        k = np.argmax(np.abs(A[i:m, j])) + i
        p = np.abs(A[k, j])
        if p <= tol:
            # The column is negligible, zero it out
            A[i:m, j] = 0.0
            j += 1
        else:
            # Remember the column index
            jb.append(j)
            if i != k:
                # Swap the i-th and k-th rows
                A[[i, k], j:n] = A[[k, i], j:n]
            # Divide the pivot row i by the pivot element A[i, j]
            A[i, j:n] = A[i, j:n] / A[i, j]
            # Subtract multiples of the pivot row from all the other rows
            for k in range(m):
                if k != i:
                    A[k, j:n] -= A[k, j] * A[i, j:n]
            i += 1
            j += 1
    # Finished
    return A

def sigmoid(z): # calculates the sigmoid function
    sigma = 1./(1+np.exp(-z)) # exp operates elementwise on vectors
    is_largerthanone = (sigma>0.9999)
    is_zero = (sigma<0.0001)
    sigma = is_largerthanone * 0.9999 + (1-is_largerthanone)*sigma
    sigma = is_zero*0.0001 + (1-is_zero)*sigma
    
    return sigma
    


def CodeBookDesign_Bin(N,K,T_0):
    Bin_num = int(N/T_0)

    Bin_Sel_num = int(K/T_0)
    Bin_size = T_0

    B = int(nCr(Bin_num, Bin_Sel_num))

    print('@CodeBookDesign_Bin, Codebook Size=', B)

    #defines the array of numbers and the two columns
    number = range(Bin_num)
    col_one = []

    #creates an array that holds the first four
    results = itertools.combinations(number,Bin_Sel_num)

    for x in results:
    #     print(x)
        col_one.append(list(x))

    col_one = np.array(col_one)
    #print(np.shape(col_one))

    Codebook_tmp = np.zeros((B,Bin_num), dtype='int')
    Codebook = np.zeros((B,N), dtype='int')

    for b in range(B):    
        for sel in col_one[b,:]:
            stt_pos = sel * Bin_size
            end_pos = (sel+1) * Bin_size
            Codebook_tmp[b,sel] = 1
            Codebook[b,stt_pos:end_pos] = 1
            
    return Codebook

def CodeBookDesign_Partition(N,K):
    B_Partition = int(N/K)

    Codebook_Partition = np.zeros((B_Partition,N), dtype='int')

    print('@CodeBookDesign_Partition, Codebook Size=',B_Partition)

    for b in range(B_Partition):
        stt_pos = b * K
        end_pos = (b+1) * K
        Codebook_Partition[b,stt_pos:end_pos] = 1

    return Codebook_Partition

def test_function(y_hat, test_image):
    
    flag = y_hat - 0.5
    y_label = (abs(np.sign(flag)) + np.sign(flag))/2
    tmp = test_image.reshape(len(test_image),1)
    num_Error = np.sum(abs(y_label - tmp))

    accuracy = 1 - float(num_Error)/float(len(test_image))

    return accuracy*100

def UserSelection_Codebook(P, B):
    CodeBook_size, N = np.shape(B)
#     print(CodeBook_size, N)
    
    P_sum = np.sum(P, axis=0)
#     print(P_sum)
    
    score = np.zeros((CodeBook_size,))
    for i in range(len(B)):
        tmp_code = B[i,:]
        score[i] = np.sum(P_sum * tmp_code)
#         print(tmp_score)
#     print(score)
    
    min_index = np.argmin(score)
    min_score = score[min_index]

    min_index_array = np.where(score == min_score)

    idx_sel = np.random.choice(min_index_array[0], 1, replace=False)
    
#     print(index_array)
    
    return B[idx_sel[0],:]


def ModelDiff_tensor(w1,w2):
    w_avg = copy.deepcopy(w1)
    w1_np = np.zeros((1,1))
    w2_np = np.zeros((1,1))
    for k in w_avg.keys():
        tmp1 = w1[k].cpu().detach().numpy()
        tmp2 = w2[k].cpu().detach().numpy()
        cur_shape = tmp1.shape
        _d = np.prod(cur_shape)
        
        tmp1 = np.reshape(tmp1,(1,_d))
        tmp2 = np.reshape(tmp2,(1,_d))
        
        dist = tmp1 - tmp2
        
        pow1 = np.matmul(tmp1, tmp1.transpose())
        pow2 = np.matmul(tmp2, tmp2.transpose())
        
        dist_l2 = np.matmul(dist, dist.transpose())
        
#         print(k,cur_shape)
#         print(pow1, pow2, dist_l2)
#         print()
        
        w1_np = np.concatenate([w1_np,tmp1], axis=1)
        w2_np = np.concatenate([w2_np,tmp2], axis=1)
    
    w1_l2 = np.matmul(w1_np, w1_np.transpose())
    w2_l2 = np.matmul(w2_np, w2_np.transpose())
    
    dist = w1_np - w2_np
    dist_l2 = np.matmul(dist, dist.transpose())
    
    print(w1_l2, w2_l2, dist_l2)
    
    return dist_l2/w1_l2

def ModelDiff_np(w1,w2):
    cur_shape = w1.shape
    _d = np.prod(cur_shape)
    
    tmp1 = np.reshape(w1,(1,_d))
    tmp2 = np.reshape(w2,(1,_d))
        
    dist = tmp1 - tmp2
        
    pow1 = np.matmul(tmp1, tmp1.transpose())
    pow2 = np.matmul(tmp2, tmp2.transpose())
        
    dist_l2 = np.matmul(dist, dist.transpose())
    
    print(pow1, pow2, dist_l2)
    
    return dist_l2/pow1

def tensor_dim(w0):
    model_dim = 0
    
    w_avg = copy.deepcopy(w0)

    for k in w_avg.keys():
        tmp1 = w0[k].cpu().detach().numpy()
        cur_shape = tmp1.shape
        _d = np.prod(cur_shape)
        
        model_dim = model_dim + _d
        
    return model_dim