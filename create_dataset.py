import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from PIL import Image
from PIL import ImageFilter
from scipy.ndimage.interpolation import rotate
import os
import sys
from multiprocessing import Pool
import threading
import time

progress_bar_control = True


def add_bobber(img, bobber, scale=1, theta=0):
    bobber1 = Image.fromarray(bobber)
    bobber1 = bobber1.resize(
        tuple(map(int, [scale * bobber.shape[0], scale * bobber.shape[1]])),
        Image.ANTIALIAS,
    )
    bobber1 = bobber1.rotate(theta)
    bobber2 = np.array(bobber1)
    img2 = np.array(img)
    r = np.random.randint(0, img2.shape[0] - bobber2.shape[0])
    c = np.random.randint(0, img2.shape[1] - bobber2.shape[1])
    bobber3 = bobber2.astype(np.int64) + np.random.randint(-2, 2, bobber2.shape)
    bobber3 += np.random.randint(0, max(255 - bobber3.max(), 1))
    bobber3 *= bobber2 > 10
    over = bobber3 > 255
    bobber3 = bobber3 - over * bobber3 + over * 255
    under = bobber3 < 0
    bobber3 = (bobber3 - under * bobber3).astype(np.uint8)
    img2[r : r + bobber3.shape[0], c : c + bobber3.shape[1]] *= bobber3 < 10
    img2[r : r + bobber3.shape[0], c : c + bobber3.shape[1]] += bobber3
    return img2, (r + bobber3.shape[0] // 2) / img.shape[0], (c + bobber3.shape[1] // 2) / img.shape[1]


def create_image_dataset(bobbers, num):
    global progress_bar_control
    num_per_image = num // len(os.listdir("dataset_images")) + 1
    progress_bar = threading.Thread(target=progress_bar_target, args=(num_per_image,))
    progress_bar.start()
    print("Creating Dataset...")
    images = os.listdir("dataset_images")
    with open("dataset/target.txt", 'w') as fhandle:
        with Pool(12) as p:
            fhandle.write(
                "\n".join(
                    p.map(
                        multi_target,
                        [(i, e, num_per_image) for i, e in enumerate(images)],
                    )
                )
            )
    progress_bar_control = False
    progress_bar.join()
    print("\nDone")


def multi_target(args):
    i, img_name, num_per_image = args
    ret = []
    img = (mpimg.imread("dataset_images/{}".format(img_name))[:, :, :3] * 255).astype(
        np.uint8
    )
    for j in range(num_per_image):
        bg = img
        test, row, col = add_bobber(
            bg,
            bobbers[np.random.randint(len(bobbers))],
            np.random.rand() + 0.5,
            (np.random.rand() - 0.5) * 20,
        )
        plt.imsave(os.path.join("dataset", f"{i*num_per_image+j}.png"), test)
        ret.append(f"{row};{col}")
    return "\n".join(ret)


def progress_bar_target(num_per_image):
    global progress_bar_control
    bar_length=40
    num_images = len(os.listdir("dataset_images"))
    total = num_per_image * num_images
    startt = time.time()
    prev_pbar_len = 0
    time_arr = []
    while progress_bar_control:
        percentage = max(0, len(os.listdir("dataset")) - 1) / total
        try :
            eta = int((time.time() - startt) * (1 - percentage) / percentage)
            time_arr.append(eta)
            if len(time_arr) > 7:
                time_arr.pop(0)
            eta = int(np.mean(time_arr))
            etas = eta % 60
            etam = (eta // 60) % 60
            etah = eta // 60 // 60
            eta = ""
            if etah > 0:
                eta += f"{etah}h "
            if etam > 0:
                eta += f"{etam}m "
            if etas > 0:
                eta += f"{etas}s"
        except:
            eta = "NA"
        percentage = int(percentage * 100)
        pbar = '\r[{}{}] {}% [ETA {}]'.format('█' * (bar_length * percentage // 100), ' ' * (bar_length - bar_length * percentage // 100), percentage, eta)
        sys.stdout.write("{}{}".format(pbar, " " * max(0, (prev_pbar_len - len(pbar)))))
        prev_pbar_len = len(pbar)
        sys.stdout.flush()
        time.sleep(0.4)


if __name__ == "__main__":
    bobbers = []
    for i in os.listdir("bobbers"):
        bobbers.append(
            (mpimg.imread(os.path.join("bobbers", i)) * 255).astype(np.uint8)[:, :, :3]
        )
        bobbers[-1] = bobbers[-1] = bobbers[-1] * (bobbers[-1] < 250)
    if len(sys.argv) < 2:
        num = 10
    else:
        try:
            num = int(sys.argv[1])
        except:
            print(
                "Usage:\npython3 create_dataset.py [<number of samples>] [<number of bg images>]"
            )
    create_image_dataset(bobbers, num)
