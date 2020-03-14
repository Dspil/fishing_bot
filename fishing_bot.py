from time import time,sleep
from pynput import keyboard, mouse
import threading
import mss
import find_bobber
import numpy as np
from matplotlib import pyplot as plt

control = True
mymouse = mouse.Controller()
mykeyboard = keyboard.Controller()
time_left = 1000
sleep_time = 1.5
nn_time = 0.5
brightness_threshold = 0.1
frame_width = 60

def main_thread():
    global control, mymouse, mykeyboard, time_left, sleep_time, nn_time
    sleep(10)
    start = time()
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        while time() - start < time_left and control:
            mymouse.position = (monitor['width'], monitor['height'])
            sleep(0.1)
            mykeyboard.type('/')
            sleep(sleep_time)
            img = (np.flip(np.array(sct.grab(monitor))[:,:,:3].astype(np.uint8), axis = 2) / 255)
            y, x = find_bobber.find_bobber_pos(img)
            bobber_monitor = {'left' : max(x - frame_width // 2, 0), 'top' : max(y - frame_width // 2, 0), 'width' : min(frame_width, monitor['width'] - max(x - frame_width // 2, 0)), 'height' : min(frame_width, monitor['height'] - max(y - frame_width // 2, 0))}
            observe_time = time()
            frame = img[bobber_monitor['top'] : bobber_monitor['top'] + bobber_monitor['height'], bobber_monitor['left'] : bobber_monitor['left'] + bobber_monitor['width']]
            base_brightness = frame.sum()
            while time() - observe_time < 17 - nn_time - sleep_time and control:
                frame = (np.flip(np.array(sct.grab(bobber_monitor))[:,:,:3].astype(np.uint8), axis = 2) / 255)
                brightness = frame.sum()
                if (brightness - base_brightness) / base_brightness > brightness_threshold:
                    break
            sleep(0.8)
            if control:
                mymouse.position = (x, y)
                mymouse.click(mouse.Button.right, 1)
            sleep(sleep_time * 2)
            
        


def on_press(key):
    global control
    try:
        if key.char in ['m', 'M']:
            control = False
            return False
    except:
        pass

def main():
    t1 = keyboard.Listener(on_press=on_press)
    t1.start()
    t2 = threading.Thread(target = main_thread)
    t2.start()
    t1.join()
    t2.join()


main()
