# fishing_bot
Bot for World of Warcraft 3.3.5 to use computer vision in order to perform the in-game fishing mechanic. 

## General Algorithm

The bot sends a keystroke to throw the bobber in the water. Then it takes a screenshot and uses a convolutional neural network to find the bobber location (see detection algorithm). After that, it captures the specific part of the screen constantly until a certain luminosity threshold is exceeded, which means that the bobber has sunken, releasing bright water particles. Then it sends a mouse right click at the location of the bobber to capture the fish. 

## Detection algorithm

The core of the detection algorithm is a neural network able to decide whether a bobber is present in an image or not. Its architecture consists of convolutional layers with max pooling layes in between and followed by some linear layers. It takes an 2d colored image as input and outputs the probability of a bobber being present. 

This neural network was trained by a computer generated dataset. Each positive image in this dataset contains a randomly rotated, translated and scaled instance of a rotoscoped bobber in front of a randomly zoomed background (out of many captured in-game backgrounds) with added noise. Each negative image contains only the noised background. The backgrounds have random rectangles blacked out for reasons that will become obvious later. To detect the position of the bobber using this neural network the following algorithm is used:
1) A sliding window technique with a large window size is used to find the general area of the bobber.
2) Split the window through it's max length axis to two or three parts and completely erase one of them.
3) Decide which piece of the window still contains the bobber using the neural network and use this piece as the new window repeating step 2 until a small enough window is achieved.


## Authors
  
  * Danai Efstathiou ([danaiefst](https://github.com/danaiefst))
  * Dionysios Spiliopoulos ([Dspil](https://github.com/Dspil))
