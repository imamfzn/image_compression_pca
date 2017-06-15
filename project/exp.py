from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def pca(X, numpc):
    mean = np.mean(X)
    cov_mat = (X - mean).T.dot((X - mean)) / (X.shape[0] - 1)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real

    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda tup: tup[0])
    eig_pairs.reverse()

    arr_w = ()
    col = len(eig_pairs[0][1])
    for i in range(0, numpc):
        arr_w = arr_w + (eig_pairs[i][1].reshape(col, 1),)

    w = np.hstack(arr_w)
    return mean, w

def compress(img):
    img_com = []
    R = 0; G = 1; B = 2
    NUM_PC = 250

    mean_r, w_r = pca(img[...,R], NUM_PC)
    mean_g, w_g = pca(img[...,G], NUM_PC)
    mean_b, w_b = pca(img[...,B], NUM_PC)

    final_r = (img[...,R] - mean_r).dot(w_r)
    img_com_r = final_r.dot(w_r.T) + mean_r

    final_g = (img[...,G] - mean_g).dot(w_g)
    img_com_g = final_g.dot(w_g.T) + mean_g

    final_b = (img[...,B] - mean_b).dot(w_b)
    img_com_b = final_b.dot(w_b.T) + mean_b

    result = np.dstack((img_com_r, img_com_g, img_com_b))
    result = result.astype(np.uint8)
    return result

__DIR__ = '../'
FILE_NAME = 'lena_square.jpg'

img = misc.imread(__DIR__ + FILE_NAME)
result = compress(img)
misc.imsave(__DIR__ + "compressed_" + FILE_NAME, result)

