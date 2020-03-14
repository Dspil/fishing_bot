import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from PIL import Image
from PIL import ImageFilter
from scipy.ndimage.interpolation import rotate
import os
import sys

def dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1/2)

def add_dot(x, base_color=None):
    ab = 0.5
    color = 250
    bubble = np.array([color, color, color])
    r1 = int(np.random.rand() * x.shape[0])
    c1 = int(np.random.rand() * x.shape[1])
    r2 = int((np.random.rand() - 0.5) * x.shape[0] / 5 + r1)
    c2 = int((np.random.rand() - 0.5) * x.shape[1] / 5 + c1)
    r = (np.random.rand() + 1) * dist(r1, c1, r2, c2)
    for i in range(int(max(min(r1,r2) - r, 0)), int(min(max(r1, r2) + r, x.shape[0]))):
        for j in range(int(max(min(c1,c2) - r, 0)), int(min(max(c1, c2) + r, x.shape[1]))):
            if dist(r1, c1, i, j) + dist(r2, c2, i, j) <= r:
                aa = 1-(dist(r1, c1, i, j) + dist(r2, c2, i, j))/r/1.1
                x[i][j] = ((bubble * aa + ab * x[i][j] * (1 - aa)) / (aa + ab * (1 - aa))).astype(np.int64)
    return x

def random_bg(r = 150, c = 150, base_color = np.array([140, 185, 237], dtype = np.uint64), filters = 1, prob = 0.499):
    x = np.zeros((2*r,2*c,3), dtype = np.int64)
    x[0][0] = base_color
    for i in range(2*r):
        for j in range(2*c):
            if j > 0:
                if np.random.rand() < prob:
                    x[i][j] = x[i][j-1]
                    continue
            if i > 0:
                if np.random.rand() < prob / (1 - prob):
                    x[i][j] = x[i-1][j]
                    continue
            x[i][j] = base_color + (np.random.rand(3) - 0.5) * 200
    over = (x > 255)
    x = x - over * x + over * 255
    under = x < 0
    x = (x - under * x).astype(np.uint8)
    ys = rotate(x, angle = np.random.rand() * 360)
    ys = ys[(ys.shape[0] - r) // 2: (ys.shape[0] - r) // 2 + r, (ys.shape[1] - c) // 2: (ys.shape[1] - c) // 2 + c]
    ys = Image.fromarray(ys)
    for i in range(filters):
        ys = ys.filter(ImageFilter.SMOOTH)
    return np.array(ys)


def create_bg(r = 150, c = 150, base_color = np.array([140, 185, 237], dtype = np.uint64), dots_ratio = 0.1, filters = 1):
    prob = 0.5
    x = np.zeros((r,c,3), dtype = np.int64)
    x[0][0] = base_color
    for i in range(r):
        for j in range(c):
            if j > 0:
                if np.random.rand() < 0.7:
                    x[i][j] = x[i][j-1]
                    continue
            if i > 0:
                if np.random.rand() < 0.5:
                    x[i][j] = x[i-1][j]
                    continue
            x[i][j] = base_color + (np.random.rand(3) - 0.5) * 20
    scale1 = (np.arange(r * c * 4).reshape((r * 2,c * 2)) / r / c / 4 * 80).astype(np.int64)
    scale = np.zeros((r * 2, c * 2, 3)).astype(np.int64)
    for i in range(3):
        scale[:,:,i] = scale1
    scale = rotate(scale, angle = np.random.rand() * 360)
    scale = scale[(scale.shape[0] - r) // 2: (scale.shape[0] - r) // 2 + r, (scale.shape[1] - c) // 2: (scale.shape[1] - c) // 2 + c]
    dots = int(r * dots_ratio)
    for i in range(dots):
        x = add_dot(x)
    over = (x > 255)
    x = x - over * x + over * 255
    under = x < 0
    x = (x - under * x).astype(np.uint8)
    y = Image.fromarray(x)
    ys = y
    for i in range(filters):
        ys = ys.filter(ImageFilter.SMOOTH)
    return ys

def add_bobber(img, bobber, scale = 1, theta = 0):
    bobber1 = Image.fromarray(bobber)
    bobber1 = bobber1.resize(tuple(map(int, [scale * bobber.shape[0], scale * bobber.shape[1]])), Image.ANTIALIAS)
    bobber1 = bobber1.rotate(theta)
    bobber2 = np.array(bobber1)
    img2 = np.array(img)
    r = np.random.randint(0, img2.shape[0] - bobber2.shape[0])
    c = np.random.randint(0, img2.shape[1] - bobber2.shape[1])
    bobber3 = bobber2.astype(np.int64) + np.random.randint(-2, 2, bobber2.shape)
    bobber3 += np.random.randint(-30, 30)
    bobber3 *= bobber2 > 10
    over = (bobber3 > 255)
    bobber3 = bobber3 - over * bobber3 + over * 255
    under = bobber3 < 0
    bobber3 = (bobber3 - under * bobber3).astype(np.uint8)
    img2[r:r + bobber3.shape[0], c:c + bobber3.shape[1]] *= (bobber3 < 10)
    img2[r:r + bobber3.shape[0], c:c + bobber3.shape[1]] += bobber3
    return img2

def create_dataset(bobbers, num):
    bar_length = 20
    num //= 3
    print("Creating Dataset...")
    for i in range(num):
        percentage = 100 * i // num + 1
        sys.stdout.write('\r[{}{}] {}%'.format('#' * (bar_length * percentage // 100), ' ' * (bar_length - bar_length * percentage // 100), percentage))
        sys.stdout.flush()
        bg = create_bg(base_color = np.random.randint(0,255,3), dots_ratio = np.random.random() * 0.1, filters = np.random.randint(0, 4))
        if np.random.random() < 0.1:
            bg = np.array(bg)
            r = np.random.randint(0, bg.shape[0], 2)
            c = np.random.randint(0, bg.shape[1], 2)
            i1, i2 = min(r), max(r)
            j1, j2 = min(c), max(c)
            bg[i1:i2, j1:j2] = 0
        plt.imsave(os.path.join('dataset', '{}.png'.format(3*i)), np.array(bg))
        test = add_bobber(bg, bobbers[np.random.randint(len(bobbers))], np.random.rand() + 0.5, (np.random.rand() - 0.5) * 20)
        plt.imsave(os.path.join('dataset', '{}.png'.format(3*i+1)), test)
        bg2 = create_bg(base_color = np.random.randint(0,255,3))
        if np.random.random() < 0.1:
            bg2 = np.array(bg2)
            r = np.random.randint(0, bg2.shape[0], 2)
            c = np.random.randint(0, bg2.shape[1], 2)
            i1, i2 = min(r), max(r)
            j1, j2 = min(c), max(c)
            bg2[i1:i2, j1:j2] = 0
        test2 = add_bobber(bg2, bobbers[np.random.randint(len(bobbers))], np.random.rand() + 0.5, (np.random.rand() - 0.5) * 20)
        plt.imsave(os.path.join('dataset', '{}.png'.format(3*i+2)), test2)
        """randombg = add_bobber(randombg, bobbers[np.random.randint(len(bobbers))], np.random.rand() + 0.5, (np.random.rand() - 0.5) * 20)
        plt.imsave(os.path.join('dataset', '{}.png'.format(4*i+2)), randombg)"""
        
    print("\nCreating targets...")
    with open("dataset/target.txt", 'w') as fhandle:
        for i in range(num):
            fhandle.write("1\n0\n0\n")
    print("Done")

def create_extra(prev, num2):
    num = num2 // len(os.listdir('bgs'))
    count = (prev // 3) * 3
    start = count
    bar_length = 20
    fhandle = open("dataset/target.txt", 'a')
    print("Creating extra bg's...")
    for i in os.listdir('bgs'):
        img = mpimg.imread('bgs/{}'.format(i))
        img = img[:img.shape[0]//3,:,:3]
        for j in range(num):
            percentage = 100 * (count - start) // num2 + 1
            sys.stdout.write('\r[{}{}] {}%'.format('#' * (bar_length * percentage // 100), ' ' * (bar_length - bar_length * percentage // 100), percentage))
            sys.stdout.flush()
            r = np.random.randint(0, img.shape[0] - 150)
            c = np.random.randint(0, img.shape[1] - 150)
            bg = img[r:r+150, c:c+150]
            bg += (np.random.rand(150, 150, 3) - 0.5) * 2 * np.random.random() * 0.078 + (np.random.random() - 0.5) * 0.1
            over = (bg > 1)
            bg = bg - over * bg + over
            under = bg < 0
            bg = (bg - under * bg)
            plt.imsave(os.path.join('dataset', '{}.png'.format(count)), bg)
            count += 1
            fhandle.write("1\n")
    fhandle.close()
    print("\nDone")
        


if __name__ == "__main__":
    bobbers = []
    for i in os.listdir('bobbers'):
        bobbers.append((mpimg.imread(os.path.join('bobbers', i)) * 255).astype(np.uint8)[:,:,:3])
        bobbers[-1] = bobbers[-1] = bobbers[-1] * (bobbers[-1] < 250)
    if len(sys.argv) < 2:
        num = 9000
    else:
        try:
            num = int(sys.argv[1])
        except:
            print("Usage:\npython3 create_dataset.py [<number of samples>] [<number of bg images>]")
    if len(sys.argv) < 3:
        num2 = 3000
    else:
        try:
            num2 = int(sys.argv[2])
        except:
            print("Usage:\npython3 create_dataset.py [<number of samples>] [<number of bg images>]")
    create_dataset(bobbers, num)
    create_extra(num, num2)
