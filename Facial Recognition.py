import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import itertools

# Read in training plots
mydata = []
path_faces = "/Users/Destination/Desktop/Thesis/Facial Recognition/faces"
for filename in glob.glob(os.path.join(path_faces, '*.jpg')):
    file = Image.open(filename, 'r').convert("L")    
    mydata.append(np.asarray(file.getdata(),dtype=np.float64).reshape((file.size[1],file.size[0])))
    file.close()

path_background = "/Users/Destination/Desktop/Thesis/Facial Recognition/background/"
for filename in glob.glob(os.path.join(path_background, '*.jpg')):
    file = Image.open(filename, 'r').convert("L")    
    mydata.append(np.asarray(file.getdata(),dtype=np.float64).reshape((file.size[1],file.size[0])))
    file.close()

mydata = np.array(mydata)
y = np.array([1] * 2000 + [-1] * 2000) # True Classifications

def feature_pairs(data, d):
    # construct the feature indices given the minimum scale of patch based on Haar features
    length = data[0].shape[0]
    result = []
    for x1 in range(0, length, d):
        for y1 in range(0, length, d):
            for x2 in range(x1 + d, length, d):
                for y2 in range(y1 + d, length, d):
                    result.append([x1, y1, x2, y2, (x2 + x1)//2, y1, x2, y2]) # d = 8
                    result.append([x1, y1, x2, y2, x1, (y1 + y2)//2, x2, y2]) # d = 8
                    result.append([x1, y1, x2, y2, (x1 + x2)//4, y1, (x1 + x2)//4 * 3, y2])  # d = 8; 3 rectangle features
                    result.append([x1, y1, x2, y2, (x1 + x2)//2, (y1 + y2)//2]) # d = 6; 1 line feature
    
    return result

# Two kinds of feature in the Haar Features
## dimension = 8
def compute_feature(data, ft):
    black = np.mean(data[:, ft[1]:ft[3], ft[0]:ft[2]], axis=(1,2))
    white = np.mean(data[:, ft[5]:ft[7], ft[4]:ft[6]], axis=(1,2))

    return - black + 2 * white


## dimension = 6
def compute_feature2(data, ft):
    black = np.mean(data[:, ft[1]:ft[3], ft[0]:ft[2]], axis=(1,2))
    white = np.mean(data[:, ft[1]:ft[5], ft[0]:ft[4]], axis=(1,2)) + np.mean(data[:, ft[5]:ft[3], ft[4]:ft[2]], axis=(1,2))

    return - black + 2 * white


# Algorithm discussed in Viola–Jones object detection framework in finding parameters to build weak classifier
def find_pars(data, fv, Dt, label):
    # compute the parameter
    N = len(data)
    sigma = np.argsort(fv) # find the permutation
    Dt_sigma = Dt[sigma]
    label_sigma = label[sigma]

    Tp = sum((label_sigma == 1) * Dt_sigma)
    Tm = sum((label_sigma == -1) * Dt_sigma)
    Sp = np.cumsum((label_sigma == 1) * Dt_sigma)
    Sm = np.cumsum((label_sigma == -1) * Dt_sigma)

    epsilon_left, epsilon_right = Sp + (Tm - Sm), Sm + (Tp - Sp)
    if np.min(epsilon_left) < np.min(epsilon_right):
        j_hat = np.argmin(epsilon_left)
        p_hat = 1
    else:
        j_hat = np.argmin(epsilon_right)
        p_hat = -1

    theta_hat = (fv[sigma[j_hat]] + fv[sigma[j_hat + 1]]) / 2 if j_hat < (N-1) else fv[sigma[j_hat]]
    clsfr_hat = np.sign(p_hat * (fv - theta_hat))
    error = np.sum(Dt * (clsfr_hat != label))

    return theta_hat, p_hat, error, clsfr_hat


def AdaBoost(data, label, d, all_fvs, tol=0):
    print("***Start AdaBoost Procedure***")
    N = len(data)
    Dt = np.array([1 / N] * N) # initialize Dt
    pairs = feature_pairs(mydata, d)
    H = len(all_fvs)
    result = []
    
    t1 = time.time() # timing
    num_weak_learner = 0

    while True:
        min_error = float("inf")
        theta_t, p_t = 0, 0
        min_ftval_index = 0
        clsfr_t = np.empty(N)
        count = 0

        # Obtain the minimal error weak classifier
        for h in range(H):
            fv_h = all_fvs[h]
            theta_h, p_h, error_h, clsfr_h = find_pars(data, fv_h, Dt, label)
            if min_error > error_h:
                min_error = error_h
                theta_t, p_t, clsfr_t = theta_h, p_h, clsfr_h
                min_ftval_index = count

            count += 1 # counting      
            if count % 10000 == 0:
                print(count)
                t2 = time.time()
                print(t2 - t1)

        # Obtain the strong classifier
        alpha_t = 0.5 * np.log((1 - min_error) / min_error)
        Z_t = 2 * np.sqrt((min_error * (1 - min_error)))
        Dt = Dt * np.exp(- alpha_t * label * clsfr_t) / Z_t
        weak_learner_t = (alpha_t, p_t, theta_t, pairs[min_ftval_index])
        result.append(weak_learner_t)
        print(weak_learner_t)  # print the weak classifier at round t
        
        errors_t = errors(data, label, result, all_fvs, tol, Theta=1) # compute errors (in Adaboost, not the prediction error) for current booster
        print("The errors and Theta are:")
        print(errors_t[0:4])
        
        num_weak_learner += 1
        if errors_t[0] <= 0.05 and errors_t[1] <= 0.3: # break the loop when error (false negative) <= 0.05 and false positive rate <= 0.3
            print("This stage of cascade took", t2 - t1, " seconds.")
            break

    return result, errors_t

# Obtain the strong classifer (weighted sum of weak classifiers)
def booster_fit(newdata, booster):
    if len(newdata.shape) == 2:
        newdata = np.array([newdata])
    N = len(newdata)
    T = len(booster)
    fitted = np.zeros(N)
    
    for t in range(T):
        alpha, p, theta, pair = booster[t]
        try:
            fitted += alpha * np.sign((p * (compute_feature(newdata, pair) - theta)))
        except IndexError:
            fitted += alpha * np.sign((p * (compute_feature2(newdata, pair) - theta)))            
    return fitted

def errors(newdata, newlabel, booster, all_fvs, tol, Theta=0):
    fitted = booster_fit(newdata, booster)
    faces_h = fitted[newlabel == 1]
    true_poz_num = len(faces_h)
    
    if Theta != 0:
        tol_num = round(true_poz_num * tol)
        Theta = np.partition(faces_h, tol_num)[tol_num] # 1e-4
    
    fitted = np.sign(fitted - Theta) # Thresholding
    faces = fitted[newlabel == 1]
    backgrounds = fitted[newlabel == -1]
    
    error = np.mean(fitted != newlabel)
    foz_poz = np.mean(backgrounds == 1)
    foz_neg = np.mean(faces == -1)
    
    # Following four values actually are not used
    update_index = (newlabel == 1) + ((fitted == 1) * (newlabel == -1)) 
    update_data = newdata[update_index]
    update_label = newlabel[update_index]
    update_fvs = all_fvs[:,update_index]

    return error, foz_poz, foz_neg, Theta, update_data, update_label, update_fvs

def cascade(data, label, d, all_fvs, tol=0, num=4): # `num` is the times of running adaboost
    newdata = data
    newlabel = label
    new_fvs = all_fvs
    result = []
    for i in range(num):
        booster_i = AdaBoost(newdata, newlabel, d, new_fvs, tol)
        newdata, newlabel, new_fvs = booster_i[1][4], booster_i[1][5], booster_i[1][6]
        result.append(booster_i)
    
    return result


def VJ_classifier(pars, image, Theta):
    B = len(pars)
    for b in range(B):
        if np.sign(booster_fit(image, pars[b]) - Theta[b]) == -1:
            return -1
    return 1


# Create a list containing all feature values on training data
start = time.time()
c = 0
print("***Start Computing All Feature Values***")
feartbl = feature_pairs(mydata, 4)
all_feature_vals = []

for h in feartbl:
    if len(h) == 8:
        all_feature_vals.append(compute_feature(mydata, h))
    elif len(h) ==6:
        all_feature_vals.append(compute_feature2(mydata, h))
    c += 1
    if c % 2000 == 0:
        stop = time.time()
        print(c, "features are computed")
        print(stop - start)
        
all_feature_vals = np.array(all_feature_vals, dtype = np.dtype("float32"))


# Obtain the parameters using the training sets
test2 = cascade(mydata, y, 4, all_feature_vals, tol = 0, num = 4)

parameter_list = [test2[i][0] for i in range(len(test2))]
Theta_list = [test2[i][1][3] for i in range(len(test2))]


# Testing picture
test_file = Image.open("/Users/Destination/Desktop/Thesis/Facial Recognition/class.jpg", 'r')
test_file = test_file.convert("L")
test_image = np.asarray(test_file.getdata(),dtype = np.float64).reshape((test_file.size[1],test_file.size[0]))
test_file.close()

plot_mat_y = np.array(test_image)
y_dim, x_dim = test_image.shape
step = 10
white_line = np.zeros((64,4)) + 256 # Indicator line dimension 64 * 4
train_dim = 64


for x in range(0, x_dim, step):
    for y in range(0, y_dim, step):
        test_block = test_image[y:(y + train_dim), x:(x + train_dim)]
	    ## Avoid overlapping
        if len(plot_mat_y[y:(y + train_dim), x:(x + train_dim)][plot_mat_y[y:(y + train_dim), x:(x + train_dim)] == 256]) < 100:
            
            if test_block.shape == (64,64) and VJ_classifier(parameter_list, test_block, Theta_list) == 1:
                plot_mat_y[y:(y + train_dim), x:(x + 4)] = white_line
                plot_mat_y[y:(y + train_dim), (x + train_dim - 4):(x + train_dim)] = white_line
                plot_mat_y[y:(y + 4), x:(x + train_dim)] = white_line.T
                plot_mat_y[(y + train_dim - 4):(y + train_dim), x:(x + train_dim)] = white_line.T
                
plt.imshow(plot_mat_y, cmap = 'gray')
plt.rcParams["figure.figsize"] = (50,50)
plt.show()


plot_mat_x = np.array(test_image)

for y in range(0, y_dim, step):
    for x in range(0, x_dim, step):
        test_block = test_image[y : (y + train_dim), x : (x + train_dim)]
	    ## Avoid overlapping
        if len(plot_mat_x[y : (y + train_dim), x : (x + train_dim)][plot_mat_x[y : (y + train_dim), x : (x + train_dim)] == 256]) < 180:

            if test_block.shape == (64,64) and VJ_classifier(parameter_list, test_block, Theta_list) == 1:
                plot_mat_x[y:(y + train_dim), x:(x + 4)] = white_line
                plot_mat_x[y:(y + train_dim), (x + train_dim - 4):(x + train_dim)] = white_line
                plot_mat_x[y:(y + 4), x:(x + train_dim)] = white_line.T
                plot_mat_x[(y + train_dim - 4):(y + train_dim), x:(x + train_dim)] = white_line.T
                
plt.imshow(plot_mat_x, cmap = 'gray')
plt.rcParams["figure.figsize"] = (50,50)
plt.show()