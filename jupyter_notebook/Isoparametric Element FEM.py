# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
from copy import copy
import matplotlib
import matplotlib.pyplot as plt

node_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
# connect_infoは要素を作っているnodeのglobal番号, Aは要素面積
element_data = np.array([{"connect_info": [1, 2, 4, 3], "A": 1}])
# 境界条件より既知となる部分
known_part_of_d = { "u1": 0, "v1": 0, "v2": 0, "u3": 0 }
known_part_of_F = { "H2": 5, "V3": 0, "H4": 5, "V4": 0 }
E = 100
PR = (1 / 3)

node_data2 = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=np.float64)
element_data2 = np.array([{"connect_info": [4, 2, 1, 3], "A": 1}])
known_part_of_d2 = { "u3": 0, "v3": 0, "u4": 0, "v4": 0 }
known_part_of_F2 = { "H1": 0, "V1": 0, "H2": 0, "V2": -100 }
E2 = 100
PR2 = 0


# E: Young's module PR: Poisson's ratio
def elastic_matrix(E, PR):
    return (E / (1 - PR ** 2)) * np.array([[1, PR, 0], [PR, 1, 0], [0, 0, (1 - PR) / 2]])


# ガウス積分点指定
def ab(kk):
    if kk == 0:
        a = -0.5773502692
        b = -0.5773502692
    if kk == 1:
        a = 0.5773502692
        b = -0.5773502692
    if kk == 2:
        a = 0.5773502692
        b = 0.5773502692
    if kk == 3:
        a = -0.5773502692
        b = 0.5773502692
    return a,b


def B_matrix(a, b, x1, y1, x2, y2, x3, y3, x4, y4):
    B = np.zeros((3,8),dtype=np.float64)
    #dN/da,dN/db
    dn1a = -0.25 * (1.0 - b)
    dn2a = 0.25 * (1.0 - b)
    dn3a = 0.25 * (1.0 + b)
    dn4a = -0.25 * (1.0 + b)
    dn1b = -0.25 * (1.0 - a)
    dn2b = -0.25 * (1.0 + a)
    dn3b = 0.25 * (1.0 + a)
    dn4b = 0.25 * (1.0 - a)
    #Jacobi matrix and det(J)
    J11 = dn1a * x1 + dn2a * x2 + dn3a * x3 + dn4a * x4
    J12 = dn1a * y1 + dn2a * y2 + dn3a * y3 + dn4a * y4
    J21 = dn1b * x1 + dn2b * x2 + dn3b * x3 + dn4b * x4
    J22 = dn1b * y1 + dn2b * y2 + dn3b * y3 + dn4b * y4
    detJ = J11 * J22 - J12 * J21
    #[B]=[dN/dx][dN/dy]
    B[0,0] = J22 * dn1a - J12 * dn1b
    B[0,2] = J22 * dn2a - J12 * dn2b
    B[0,4] = J22 * dn3a - J12 * dn3b
    B[0,6] = J22 * dn4a - J12 * dn4b
    B[1,1] = -J21 * dn1a + J11 * dn1b
    B[1,3] = -J21 * dn2a + J11 * dn2b
    B[1,5] = -J21 * dn3a + J11 * dn3b
    B[1,7] = -J21 * dn4a + J11 * dn4b
    B[2,0] = -J21 * dn1a + J11 * dn1b
    B[2,1] = J22 * dn1a - J12 * dn1b
    B[2,2] = -J21 * dn2a + J11 * dn2b
    B[2,3] = J22 * dn2a - J12 * dn2b
    B[2,4] = -J21 * dn3a + J11 * dn3b
    B[2,5] = J22 * dn3a - J12 * dn3b
    B[2,6] = -J21 * dn4a + J11 * dn4b
    B[2,7] = J22 * dn4a - J12 * dn4b
    B = B / detJ
    return B, detJ


def K_matrix(element_elastic_matrix, A, x1, y1, x2, y2, x3, y3, x4, y4):
    K = np.zeros((8,8),dtype = np.float64)
    #Stiffness matrix [B]T[D][B]*t*det(J)
    for kk in range(0,4):
        a, b = ab(kk)
        B, detJ = B_matrix(a, b, x1, y1, x2, y2, x3, y3, x4, y4)
        K += np.dot(B.T, np.dot(element_elastic_matrix, B)) * float(A) * float(detJ)
    return K


# dの未知の要素に対応する行、列を削除
def compress_K_F(K, d, F):
    compressed_K = K
    compressed_d = d
    compressed_F = F
#     いくつ削除したかを記憶する
    delete_count = 0
    for index, element_d in enumerate(d):
        if element_d != None:
#             削除する行と列が上から数えて何番目のものか
            delete_num = index + 1 - delete_count
            compressed_K = np.delete(compressed_K, delete_num - 1, 0)
            compressed_K = np.delete(compressed_K, delete_num - 1, 1)
            compressed_d = np.delete(compressed_d, delete_num - 1, 0)
            compressed_F = np.delete(compressed_F, delete_num - 1, 0)
            delete_count += 1
    return { "compressed_K": compressed_K, "compressed_F": compressed_F }


# +
# connect_infoは要素を作っているnode番号, Aは要素面積
# known_part_of_dとknow_part_of_Fは境界条件より既知となる部分
# E: Young's module PR: Poisson's ratio

def ISO_FEM(node_data, element_data, known_part_of_d, known_part_of_F, E, PR):
#     Kを0で初期化
    K = np.zeros([node_data.shape[0] * 2, node_data.shape[0] * 2])
    element_elastic_matrix = elastic_matrix(E, PR)
#     assembling
    for element in element_data:
        info = element['connect_info']
        x1 = node_data[info[0] - 1][0]
        y1 = node_data[info[0] - 1][1]
        x2 = node_data[info[1] - 1][0]
        y2 = node_data[info[1] - 1][1]
        x3 = node_data[info[2] - 1][0]
        y3 = node_data[info[2] - 1][1]
        x4 = node_data[info[3] - 1][0]
        y4 = node_data[info[3] - 1][1]
        A = element['A']

        element_K_matrix = K_matrix(element_elastic_matrix, A, x1, y1, x2, y2, x3, y3, x4, y4)

#       Kに対応する部分を足していく
#       node_i, node_jはnodeのglobal番号
        for i, node_i in enumerate(element['connect_info']):
            for j, node_j in enumerate(element['connect_info']):
                K[2*(node_i - 1)][2*(node_j - 1)] += element_K_matrix[2 * i][2 * j]
                K[2*(node_i - 1)][2*(node_j - 1) + 1] += element_K_matrix[2 * i][2 * j + 1]
                K[2*(node_i - 1) + 1][2*(node_j - 1)] += element_K_matrix[2 * i + 1][2 * j]
                K[2*(node_i - 1) + 1][2*(node_j - 1) + 1] += element_K_matrix[2 * i + 1][2 * j + 1]

#   dとFをNoneで初期化
    d = [None] * (node_data.shape[0] * 2)
    F = [None] * (node_data.shape[0] * 2)

#     dとFに既知の部分を代入
    for k, v in known_part_of_d.items():
#         vecはuまたはv方向
        vec = k[0]
        node_num = int(k[1])
        if vec == 'u':
            d[(node_num - 1) * 2] = v
        elif vec == 'v':
            d[(node_num - 1) * 2 + 1] = v

    for k, v in known_part_of_F.items():
#         vecはHまたはV方向
        vec = k[0]
        node_num = int(k[1])
        if vec == 'H':
            F[(node_num - 1) * 2] = v
        elif vec == 'V':
            F[(node_num - 1) * 2 + 1] = v
#     uが既知の部分のkd = Fを削除して未知の部分のdを求める
    compressed_K_F_dict = compress_K_F(K, d, F)
    compressed_K = compressed_K_F_dict['compressed_K'].astype(np.float64)
    compressed_F = compressed_K_F_dict['compressed_F'].astype(np.float64)
    compressed_d = np.linalg.solve(compressed_K, compressed_F)
#     dの未知の部分に求めたcompressed_dを代入
#     compressed_dの対応するindex
    compressed_d_index = 0

    for index, element_d in enumerate(d):
        if element_d == None:
            d[index] = compressed_d[compressed_d_index]
            compressed_d_index += 1

    for index, element_F in enumerate(F):
        if element_F == None:
            F[index] = np.dot(K[index], d)

    cordinates = copy(node_data)
    for i in range(0, node_data.shape[0]):
        cordinates[i][0] += d[i * 2]
        cordinates[i][1] += d[i * 2 + 1]
    return { 'd': d, 'F': F, 'cordinates': cordinates }


# -

ISO_FEM(node_data, element_data, known_part_of_d, known_part_of_F, E, PR)

result = ISO_FEM(node_data, element_data, known_part_of_d, known_part_of_F, E, PR)

result = ISO_FEM(node_data2, element_data2, known_part_of_d2, known_part_of_F2, E2, PR2)

result

cordinates = result['cordinates']

cordinates

# +
X = np.zeros(4, dtype=np.float64)
Y = np.zeros(4, dtype=np.float64)

for i in range(0, 4):
    X[i] += node_data2[i][0]
    Y[i] += node_data2[i][1]

# +
x = np.zeros(4, dtype=np.float64)
y = np.zeros(4, dtype=np.float64)

for i in range(0, 4):
    x[i] += cordinates[i][0]
    y[i] += cordinates[i][1]


# +
def main():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, c='red')
    ax.scatter(X, Y, c='blue')
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    main()
# -


