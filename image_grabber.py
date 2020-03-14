import mss
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os
import time
import sys



with mss.mss() as sct:
    c = len(os.listdir('dataset_images'))
    monitor = sct.monitors[0]
    while True:
        time.sleep(20)
        img = np.flip(np.array(sct.grab(monitor))[:,:,:3].astype(np.uint8), axis = 2)
        mpimg.imsave('dataset_images/{}.png'.format(c), img)
        c += 1
        print(chr(7))
        
