import numpy as np
import sys
import imageio


@profile
def gaussian_smoothing(img):
    width = img.shape[0]
    height = img.shape[1]
    gaussian_mask = [[1, 1, 2, 2, 2, 1, 1],
                     [1, 2, 2, 4, 2, 2, 1],
                     [2, 2, 4, 8, 4, 2, 2],
                     [2, 4, 8, 16, 8, 4, 2],
                     [2, 2, 4, 8, 4, 2, 2],
                     [1, 2, 2, 4, 2, 2, 1],
                     [1, 1, 2, 2, 2, 1, 1]]
    # Initiate  a new Img Array, img.shape store the number of rows and columns
    new_gray = np.zeros([width, height])
    # There are 3 layers of pixels are out of boarder. I don't need to iterate those pixels.
    # narrow the range
    width_range = width - 3
    height_range = height - 3
    for row in range(width):
        for col in range(height):
            if 3 <= row < width_range and 3 <= col < height_range:
                # if out of boundary
                sum, x, y = 0, 0, 0
                # do convolution wth the mask
                for i in range(7):
                    for j in range(7):
                        x = row + (i - 3)
                        y = col + (j - 3)
                        sum += img[x][y] * gaussian_mask[i][j]
                # normalization
                new_gray[row][col] = sum / 140
    return new_gray

@profile
def gradient_operator(img):
    width = img.shape[0]
    height = img.shape[1]
    # Initiate three Img Arrays, img.shape store the number of rows and columns
    gx = np.zeros([width, height])
    gy = np.zeros([width, height])
    magnitude_arr = np.zeros([width, height])
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
    for row in range(width):
        for col in range(height):
            # There are total 4 layers of pixels of the original image is out of boundary.
            if 4 <= row < width - 4 and 4 <= col < height - 4:
                sum_gx, sum_gy, x, y = 0, 0, 0, 0
                # Convolution
                for i in range(3):
                    for j in range(3):
                        x = row + (i - 1)
                        y = col + (j - 1)
                        sum_gx += img[x][y] * prewitt_mask['Gx'][i][j]
                        sum_gy += img[x][y] * prewitt_mask['Gy'][i][j]
                # absolute value
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

    # track the max and the min of magnitude
    mag_max, mag_min = 0, sys.maxsize
    # normalize Gx and Gy and generate magnitude array
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gx[i][j] = (gx[i][j] - gx_min) * 255 / (gx_max - gx_min)
            gy[i][j] = (gy[i][j] - gy_min) * 255 / (gy_max - gy_min)
            magnitude_arr[i][j] = np.sqrt(np.power(gx[i][j], 2) + np.power(gy[i][j], 2))
            if magnitude_arr[i][j] > mag_max:
                mag_max = magnitude_arr[i][j]
            if magnitude_arr[i][j] < mag_min:
                mag_min = magnitude_arr[i][j]

    # normalize magnitude
    for i in range(magnitude_arr.shape[0]):
        for j in range(magnitude_arr.shape[1]):
            magnitude_arr[i][j] = (magnitude_arr[i][j] - mag_min) * 255 / (mag_max - mag_min)

    return gx, gy, magnitude_arr

@profile
def non_maxima_suppression(gx, gy, magnitude):
    width = magnitude.shape[0]
    height = magnitude.shape[1]
    magnitude_sup = np.zeros([width, height])

    pi = np.pi
    # the smallest section of pi, pi/8
    sec = pi / 8
    for i in range(width):
        for j in range(height):
            # There are overall 5 layers of pixels are out of boarder. I don't need to iterate those pixels.
            # narrow the range
            if 5 <= i < width - 5 and 5 <= j < height - 5:
                angle = np.arctan(gy[i][j] / gx[i][j])
                if angle < 0:
                    angle = angle + 2 * pi
                # if angle belongs to sector 0
                if (7 * sec <= angle < 9 * sec) or (0 <= angle < sec) or (
                        2 * pi - sec <= angle < 2 * pi):
                    if magnitude[i][j] > magnitude[i - 1][j] and magnitude[i][j] > magnitude[i + 1][j]:
                        magnitude_sup[i][j] = magnitude[i][j]
                    else:
                        magnitude_sup[i][j] = 0
                # if angle belongs to sector 1
                elif (sec <= angle < 3 * sec) or (9 * sec <= angle < 11 * sec):
                    if magnitude[i][j] > magnitude[i + 1][j + 1] and magnitude[i][j] > magnitude[i - 1][j - 1]:
                        magnitude_sup[i][j] = magnitude[i][j]
                    else:
                        magnitude_sup[i][j] = 0
                elif (3 * sec <= angle < 5 * sec) or (11 * sec <= angle < 13 * sec):
                    if magnitude[i][j] > magnitude[i][j + 1] and magnitude[i][j] > magnitude[i][j - 1]:
                        magnitude_sup[i][j] = magnitude[i][j]
                    else:
                        magnitude_sup[i][j] = 0
                # if angle belongs to sector 3
                else:
                    if magnitude[i][j] > magnitude[i - 1][j + 1] and magnitude[i][j] > magnitude[i + 1][j - 1]:
                        magnitude_sup[i][j] = magnitude[i][j]
                    else:
                        magnitude_sup[i][j] = 0
    return magnitude_sup

# input = imageio.imread('zebra-crossing-1.bmp')
# output_after_gaussian = gaussian_smoothing(input)
# np.save('output_after_gaussian', output_after_gaussian)
