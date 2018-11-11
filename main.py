from matplotlib import pyplot as plt
import numpy as np
import math


def gaussian_smoothing(img, mask):
    I, J = img.shape[0], img.shape[1]
    # Initiate  a new Img Array
    new_gray = np.zeros([I, J])
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # if out of boundary
            if row - 3 < 0 or col - 3 < 0 or row + 3 > img.shape[0] - 1 or col + 3 > img.shape[1] - 1:
                # if out of boundary
                # set the output be NaN.
                new_gray[row][col] = np.NaN
            else:
                sum, x, y = 0, 0, 0
                for i in range(len(mask)):
                    for j in range(len(mask[0])):
                        if i < 3:
                            x = row - (3 - i)
                        else:
                            x = row + (i - 3)
                        if j < 3:
                            y = col - (3 - j)
                        else:
                            y = col + (j - 3)
                        sum += img[x][y] * mask[i][j]
                # normalization
                new_gray[row][col] = sum / 140
    return new_gray


def gradient_operator(img):
    prewitt_mask = {
        'Gx': ([-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]),
        'Gy': ([1, 1, 1],
               [0, 0, 0],
               [-1, -1, -1])
    }
    # for row in range(img.shape[0]):
    #     for col in range(img.shape[1]):
    #         # if out of boundary
    #         if row - 3 < 0 or col - 3 < 0 or row + 3 > img.shape[0] - 1 or col + 3 > img.shape[1] - 1:
    #             # Since the data type is uint8,  we cannot assign 'undefined' to an element.
    #             # Instead, if mask goes outside of the border, let the output be 0.
    #             new_gray[row][col] = 0

input = plt.imread('zebra-crossing-1.bmp')
gaussian_mask = [[1, 1, 2, 2, 2, 1, 1],
                 [1, 2, 2, 4, 2, 2, 1],
                 [2, 2, 4, 8, 4, 2, 2],
                 [2, 4, 8, 16, 8, 4, 2],
                 [2, 2, 4, 8, 4, 2, 2],
                 [1, 2, 2, 4, 2, 2, 1],
                 [1, 1, 2, 2, 2, 1, 1]]
output = gaussian_smoothing(input, gaussian_mask)
plt.figure(1)
# plt.subplot(221)
plt.imshow(input, cmap='gray')
plt.show()
plt.figure(2)
# plt.subplot(222)
plt.imshow(output, cmap='gray')
plt.show()
