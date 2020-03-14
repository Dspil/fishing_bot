import data_handlers
import cnn_model
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import time
from scipy.special import softmax
import os
slide = 120
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load("model.pt").to(device)
model.eval()

def find_bobber(img, slide = 120):
    result = np.zeros((((img.shape[0] - 150) // slide + 1), ((img.shape[1] - 150) // slide + 1)))
    for i in range(0, img.shape[0] - 150, slide):
        batch = np.zeros((((img.shape[1] - 150) // slide + 1), 3, 150, 150))
        for j in range(0, img.shape[1] - 150, slide):
            window = img[i : 150 + i, j : 150 + j]
            batch[j // slide, 0] = window[:,:,0]
            batch[j // slide, 1] = window[:,:,1]
            batch[j // slide, 2] = window[:,:,2]
        out = model(torch.Tensor(batch).float().to(device))
        temp = out.cpu().detach().numpy()
        result[i // slide] = softmax(temp, axis = 1)[:, 0]
    return result

def find_clicking_point(img, base_size = 25, starti = None, startj = None, endi = None, endj = None):
    if starti == None:
        starti, startj = 0,0
        endi, endj = img.shape[:2]
    if endi - starti < base_size and endj - startj < base_size:
        return starti, endi, startj, endj
    img2 = np.zeros(img.shape)
    nninput = np.empty((1, 3, 150, 150))
    while base_size < endi - starti or base_size < endj - startj:
        if endj - startj == endi - starti:
            midi = (endi + starti) // 2
            midj = (endj + startj) // 2
            tests = [(starti, midi, startj, endj), (midi, endi, startj, endj), (starti, endi, startj, midj), (starti, endi, midj, endj), ((starti + endi) // 4, (starti + endi) // 4 + (endi - starti) // 2, startj, endj)]
        elif endj - startj > endi - starti:
            width = endi - starti
            tests = [(starti, endi, startj, startj + width), (starti, endi, endj - width, endj), (starti, endi, (startj + endj) // 4, (startj + endj) // 4 + width)]
        else:
            width = endj - startj
            tests = [(starti, starti + width, startj, endj), (endi - width, endi, startj, endj), ((starti + endi) // 4, (starti + endi) // 4 + width, startj, endj)]
        grades = []
        for i1, i2, j1, j2 in tests:
            img2[i1:i2, j1:j2] = img[i1:i2, j1:j2]
            nninput[0, 0] = img2[:,:,0]
            nninput[0, 1] = img2[:,:,1]
            nninput[0, 2] = img2[:,:,2]
            out = model(torch.Tensor(nninput).float().to(device))
            temp = out.cpu().detach().numpy()[0]
            grades.append(softmax(temp)[0])
            img2[i1:i2, j1:j2] = 0
        starti, endi, startj, endj = tests[np.argmax(grades)]
    img2[starti:endi,startj:endj] = img[starti:endi,startj:endj]
    return (starti + endi) // 2, (startj + endj) // 2
    
    

def find_bobber_pos(img):
    result = find_bobber(img, 120)
    f = np.argmax(result)
    columns = ((img.shape[1] - 150) // slide + 1)
    i,j = f // columns, f % columns
    i1, j1 = find_clicking_point(img[i * 120 : i * 120 + 150, j * 120 : j * 120 + 150])
    return i * 120 + i1, j * 120 + j1

    
if __name__ == "__main__":
    count = 0
    for i in os.listdir("images"):
        count += 1
        img = plt.imread("images/{}".format(i))[:,:,:3]
        plt.imshow(img)
        plt.show()
        result = find_bobber(img, 120)
        plt.imshow(result, cmap='gray')
        plt.show()
        f = np.argmax(result)
        columns = ((img.shape[1] - 150) // slide + 1)
        i,j = f // columns, f % columns
        find_clicking_point(img[i * 120 : i * 120 + 150, j * 120 : j * 120 + 150])
        #plt.imshow(img[i * 120 : i * 120 + 150, j * 120 : j * 120 + 150])
        #plt.show()

    
