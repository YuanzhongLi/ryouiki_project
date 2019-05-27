# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

# h: 要素の長さ, E: ヤング率, A: 断面積, b: 単位当たりの長さの物体力
factor_data_dic = { 'h': [1.0, 1.0, 1.0], 'E': [2.0, 2.0, 2.0], 'A': [2.0, 1.0, 0.5] }
df = pd.DataFrame(factor_data_dic)
df


# +
# dfのi番目の節点のEA/hを計算する
def calc_node_k(df, i):
    h = df.loc[i, :].h
    E = df.loc[i, :].E
    A = df.loc[i, :].A
    return E * A / h

def solev_kf(k, f):
    gk = csr_matrix(k)
    disg = spsolve(gk, f, use_umfpack=True)
    return disg

# dfのi番目の要素の歪みを計算
def calc_e(df, i, u):
    h = df.loc[i, :].h
    return (u[i + 1] - u[i]) / h

# dfのi番目の要素の応力を計算
def calc_o(df, i, e):
    E = df.loc[i, :].E
    return e[i] * E

def EB1DFixedFreeFEM(df, p):
    # 行数
    line_num = len(df)
    # 節点数
    node_num = line_num + 1
    
    # 節点ごとEA/hの値
    node_k_list = np.zeros(line_num)
    for i in range(line_num):
        node_k_list[i] = calc_node_k(df, i)
    
    # 剛性マトリックス
    k = np.zeros([node_num, node_num])
    for i in range(line_num):
        # 2節点間の剛性マトリックス
        node_k_matrix = node_k_list[i] * np.array([[1, -1], [-1, 1]])
        k[i][i] += node_k_matrix[0][0]
        k[i + 1][i] = node_k_matrix[1][0]
        k[i][i + 1] = node_k_matrix[0][1]
        k[i + 1][i + 1] = node_k_matrix[1][1]

    f = np.zeros(node_num)
    f[0] = -p
    f[-1] = p
    
    # uは変位
    u = np.zeros(1)
    
    # k_matrixからfixdedの節点を取り除いた剛性マトリックス
    k_matrix_without_fixed_point = np.delete(k, 0, 1)
    k_matrix_without_fixed_point = np.delete(k_matrix_without_fixed_point, 0, 0)
    
    u = np.append(u, solev_kf(k_matrix_without_fixed_point, f[1:]))
    
    # 歪み
    e = np.zeros(line_num)
    for i in range(line_num):
        e[i] = calc_e(df, i, u)
    
    # 応力
    o = np.zeros(line_num)
    for i in range(line_num):
        o[i] = calc_o(df, i, e)
    
    return {'変位': u, '歪み': e, '応力': o}


# -


EB1DFixedFreeFEM(df, 4)



