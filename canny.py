import numpy as np
import sys
import imageio


# @profile
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
    new_gray = np.zeros([width, height], dtype=np.uint8)
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
                        # offset the current index
                        x = row + (i - 3)
                        y = col + (j - 3)
                        sum += img[x][y] * gaussian_mask[i][j]
                # normalization
                new_gray[row][col] = sum / 140
    return new_gray


# @profile
def gradient_operator(img):
    width = img.shape[0]
    height = img.shape[1]
    # Initiate three Img Arrays, img.shape store the number of rows and columns
    gx = np.zeros([width, height], dtype=np.uint8)
    gy = np.zeros([width, height], dtype=np.uint8)
    magnitude_arr = np.zeros([width, height], dtype=np.uint8)
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
                sum_gx, sum_gy, new_i, new_j = 0, 0, 0, 0
                # Convolution
                for i in range(3):
                    for j in range(3):
                        # offset the current index
                        new_i = row + (i - 1)
                        new_j = col + (j - 1)
                        sum_gx += img[new_i][new_j] * prewitt_mask['Gx'][i][j]
                        sum_gy += img[new_i][new_j] * prewitt_mask['Gy'][i][j]
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


# @profile
def non_maxima_suppression(gx, gy, magnitude):
    width = magnitude.shape[0]
    height = magnitude.shape[1]
    magnitude_after_sup = np.zeros([width, height], dtype=np.uint8)

    pi = np.pi
    # the smallest section of pi, pi/8
    sec = pi / 8
    for i in range(width):
        for j in range(height):
            # There are overall 5 layers of pixels are out of boarder. I don't need to iterate those pixels.
            # narrow the range
            if 5 <= i < width - 5 and 5 <= j < height - 5:
                # if gx is 0, then the angle is 90 degree, pi/2. The angle belongs to sector 2.
                if gx[i][j] == 0:
                    if magnitude[i][j] > magnitude[i + 1][j] and magnitude[i][j] > magnitude[i - 1][j]:
                        magnitude_after_sup[i][j] = magnitude[i][j]
                    else:
                        magnitude_after_sup[i][j] = 0
                else:
                    angle = np.arctan(gy[i][j] / gx[i][j])
                    if angle < 0:
                        angle = angle + 2 * pi
                    # if angle belongs to sector 0，  7*pi/8 <= angle < 9*pi/8,
                    # or 0 <= angle< pi/8 or, or 2*pi-pi/8 <= angle <= 2*pi
                    if (7 * sec <= angle < 9 * sec) or (0 <= angle < sec) or (
                            2 * pi - sec <= angle <= 2 * pi):
                        if magnitude[i][j] > magnitude[i][j - 1] and magnitude[i][j] > magnitude[i][j + 1]:
                            magnitude_after_sup[i][j] = magnitude[i][j]
                        else:
                            magnitude_after_sup[i][j] = 0
                    # if angle belongs to sector 1, pi/8<= angle < 3*pi/8 or 9* pi/8 <= angle< 11* pi/8
                    elif (sec <= angle < 3 * sec) or (9 * sec <= angle < 11 * sec):
                        if magnitude[i][j] > magnitude[i - 1][j + 1] and magnitude[i][j] > magnitude[i + 1][j - 1]:
                            magnitude_after_sup[i][j] = magnitude[i][j]
                        else:
                            magnitude_after_sup[i][j] = 0
                    # if angle belongs to sector 2
                    elif (3 * sec <= angle < 5 * sec) or (11 * sec <= angle < 13 * sec):
                        if magnitude[i][j] > magnitude[i + 1][j] and magnitude[i][j] > magnitude[i - 1][j]:
                            magnitude_after_sup[i][j] = magnitude[i][j]
                        else:
                            magnitude_after_sup[i][j] = 0
                    # if angle belongs to sector 3
                    else:
                        if magnitude[i][j] > magnitude[i - 1][j - 1] and magnitude[i][j] > magnitude[i + 1][j + 1]:
                            magnitude_after_sup[i][j] = magnitude[i][j]
                        else:
                            magnitude_after_sup[i][j] = 0
    return magnitude_after_sup


def thresholding(magnitude_after_sup, p):
    width = magnitude_after_sup.shape[0]
    height = magnitude_after_sup.shape[1]
    output = np.zeros([width, height], dtype=np.uint8)
    # initialize an arr to count the pixels of each gray level from 0 to 255
    gray_level_count = [0 for i in range(256)]
    for i in range(width):
        for j in range(height):
            gray_level_count[magnitude_after_sup[i][j]] += 1
    #  The original total number of pixels(including 0)
    total_pixels = width * height
    # The number of edge pixels (The total number minus the number of pixels with gray value of 0)
    num_edge_pixels = total_pixels - gray_level_count[0]
    num_edge_pixels_after_threshold = None
    # we need to find out a gray level T such that
    # approximately target_num number of pixels have gray level value > T
    target_num = p * num_edge_pixels
    current_num = 0
    min_difference = sys.maxsize
    T = 0
    # Iterate the gray level histogram find a gray level T such that the min difference is minimum,
    # starting from the last one(gray_level_count).
    # Since we only count edge pixels which has gray value greater than 0, we don't need to check gray_level_count[0]
    for i in range(255, 1, -1):
        current_num += gray_level_count[i]
        current_difference = target_num - current_num
        # When current_difference is less than 0, either T=i or T=i+1
        if current_difference < 0:
            # find the minimum difference in this 2 cases
            if abs(current_difference) < min_difference:
                T = i
                num_edge_pixels_after_threshold = current_num
            else:
                T = i + 1
                num_edge_pixels_after_threshold = current_num - gray_level_count[i]
                break
        # store the difference
        min_difference = current_difference

    for row in range(width):
        for col in range(height):
            if magnitude_after_sup[row][col] > T:
                output[row][col] = magnitude_after_sup[row][col]

    return output, T, num_edge_pixels_after_threshold
