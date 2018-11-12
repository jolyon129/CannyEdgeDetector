import numpy as np
import sys


def gaussian_smoothing(img):
    gaussian_mask = [[1, 1, 2, 2, 2, 1, 1],
                     [1, 2, 2, 4, 2, 2, 1],
                     [2, 2, 4, 8, 4, 2, 2],
                     [2, 4, 8, 16, 8, 4, 2],
                     [2, 2, 4, 8, 4, 2, 2],
                     [1, 2, 2, 4, 2, 2, 1],
                     [1, 1, 2, 2, 2, 1, 1]]
    # Initiate  a new Img Array, img.shape store the number of rows and columns
    new_gray = np.zeros([img.shape[0], img.shape[1]])
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # if out of boundary
            if row - 3 < 0 or col - 3 < 0 or row + 3 > img.shape[0] - 1 or col + 3 > img.shape[1] - 1:
                # if out of boundary
                # set the output be NaN.
                new_gray[row][col] = np.NaN
            else:
                sum, x, y = 0, 0, 0
                # do convolution wth the mask
                for i in range(len(gaussian_mask)):
                    for j in range(len(gaussian_mask[0])):
                        if i < 3:
                            x = row - (3 - i)
                        else:
                            x = row + (i - 3)
                        if j < 3:
                            y = col - (3 - j)
                        else:
                            y = col + (j - 3)
                        sum += img[x][y] * gaussian_mask[i][j]
                # normalization
                new_gray[row][col] = sum / 140
    return new_gray


def gradient_operator(img):
    # Initiate  two Img Arrays, img.shape store the number of rows and columns
    gx = np.zeros([img.shape[0], img.shape[1]])
    gy = np.zeros([img.shape[0], img.shape[1]])
    prewitt_mask = {
        'Gx': ([-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]),
        'Gy': ([1, 1, 1],
               [0, 0, 0],
               [-1, -1, -1])
    }
    # track the max and min of gx and gy
    gx_min, gy_min, gx_max, gy_max = sys.maxsize, sys.maxsize, 0, 0
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # There are total 4 layers of pixels of the original image is out of boundary.
            # Assign 0 to these pixels to indicate that there is no edge
            if row - 4 < 0 or col - 4 < 0 or row + 4 > img.shape[0] - 1 or col + 4 > img.shape[1] - 1:
                img[row][col] = 0
            else:
                sum_gx, sum_gy, x, y = 0, 0, 0, 0
                # Convolution
                for i in range(3):
                    for j in range(3):
                        if i < 1:
                            x = row - (1 - i)
                        else:
                            x = row + (i - 1)
                        if j < 1:
                            y = col - (1 - j)
                        else:
                            y = col + (j - 1)
                        sum_gx += img[x][y] * prewitt_mask['Gx'][i][j]
                        sum_gy += img[x][y] * prewitt_mask['Gy'][i][j]
                gx[row][col] = abs(sum_gx)
                gy[row][col] = abs(sum_gy)
                #  track the max and min of gx and gy, which are used in normalization
                if gx[row][col] > gx_max:
                    gx_max = gx[row][col]
                if gx[row][col] < gx_min:
                    gx_min = gx[row][col]
                if gy[row][col] > gy_max:
                    gy_max = gy[row][col]
                if gy[row][col] < gy_min:
                    gy_min = gy[row][col]

    # normalize Gx and Gy
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gx[i][j] = (gx[i][j] - gx_min) * 255 / (gx_max - gx_min)
            gy[i][j] = (gy[i][j] - gy_min) * 255 / (gy_max - gy_min)
    return gx, gy

# def normalization(nparr):
