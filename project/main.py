from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import sys
<<<<<<< HEAD
import math

def pca(X, numpc):
    mean = np.mean(X, axis = 0)
=======

def pca(X, numpc):
    mean = np.mean(X)
>>>>>>> f990f3c4040fb6b4479ff43674d2a923cf7387a6
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

def compress(img, NUM_PC):
    img_com = []
    R = 0; G = 1; B = 2

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

def mse(img1, img2):
    R = 0; G = 1; B = 2
    mse_r = mse_component(img1[...,R], img2[...,R])
    mse_g = mse_component(img1[...,G], img2[...,G])
    mse_b = mse_component(img1[...,B], img2[...,B])

    return mse_r, mse_g, mse_b

def mse_component(img1, img2):
    result = 0
    r, c = img1.shape
<<<<<<< HEAD
    img1 = img1.astype(np.int)
    img2 = img2.astype(np.int)
    
    for i in range(0, r):
        for j in range(0, c):
            result += math.pow((img1[i][j] - img2[i][j]),2)
=======

    for i in range(0, r):
        for j in range(0, c):
            result += (img1[i][j] - img2[i][j]) * (img1[i][j] - img2[i][j])
>>>>>>> f990f3c4040fb6b4479ff43674d2a923cf7387a6
    return float(result / (r * c))


__DIR__ = '../'
__SAVEDIR__ = __DIR__ + "compressed/"
FILE_NAME = sys.argv[1]

START_PC = 32 ; INC_PC = 32 ;

img = misc.imread(__DIR__ + FILE_NAME)
MAX_PC = img.shape[1] + 1

for numpc in range(START_PC, MAX_PC, INC_PC):
    result = compress(img, numpc)
    mse_r, mse_g, mse_b = mse(img, result)
    print ("numpc : %d, mse : %.2f %.2f %.2f" % (numpc, mse_r, mse_g, mse_b))
<<<<<<< HEAD
    misc.imsave(__SAVEDIR__ + "compressed_" + FILE_NAME + "_" + str(numpc) + "pc.JPG", result)
=======
    misc.imsave(__SAVEDIR__ + "compressed_" + FILE_NAME + "_" + str(numpc) + "pc.JPG", result)


>>>>>>> f990f3c4040fb6b4479ff43674d2a923cf7387a6
