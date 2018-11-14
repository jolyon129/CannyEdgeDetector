from matplotlib import pyplot as plt
import canny
import numpy as np
import imageio
from os import path

# plt.figure(1)
# # plt.subplot(221)
# plt.imshow(input, cmap='gray')
# plt.show()
# plt.figure(2)
# # plt.subplot(222)
# plt.imshow(output_after_gaussian, cmap='gray')
# plt.show()
# np.save('output_after_gaussian', output_after_gaussian)


# input = np.load('output_after_gaussian.npy')
# (gradient_x, gradient_y, magnitude) = canny.grad                                                                              ient_operator(input)
# plt.figure(3)
# plt.title('Gradient Responses')
# plt.subplot(311)
# plt.imshow(gradient_x, cmap='gray')
# plt.subplot(312)
# plt.imshow(gradient_y, cmap='gray')
# plt.subplot(313)
# plt.imshow(magnitude, cmap='gray')
# plt.show()
#
# np.save('gradient_x', gradient_x)
# np.save('gradient_y', gradient_y)
# np.save('magnitude', magnitude)

output_path = path.join(path.dirname(__file__), 'output')
npy_path = path.join(path.dirname(__file__), 'npy')

# input = imageio.imread('zebra-crossing-1.bmp')
# output_after_gaussian = canny.gaussian_smoothing(input)
# imageio.imsave(path.join(output_path, 'after_gaussian.bmp'), output_after_gaussian)
# np.save(path.join(npy_path, 'output_after_gaussian'), output_after_gaussian)

# after_gaussian = np.load(path.join(npy_path, 'output_after_gaussian.npy'))
# (gx, gy, magnitude) = canny.gradient_operator(after_gaussian)
# imageio.imsave(path.join(output_path, 'gx.bmp'), gx)
# imageio.imsave(path.join(output_path, 'gy.bmp'), gy)
# imageio.imsave(path.join(output_path, 'magnitude.bmp'), magnitude)
# np.save(path.join(npy_path, 'gx'), gx)
# np.save(path.join(npy_path, 'gy'), gy)
# np.save(path.join(npy_path, 'magnitude'), magnitude)


# magnitude = np.load(path.join(npy_path, 'magnitude.npy'))
# gx = np.load(path.join(npy_path, 'gx.npy'))
# gy = np.load(path.join(npy_path, 'gy.npy'))
# magnitude_after_sup = canny.non_maxima_suppression(gx, gy, magnitude)
# np.save(path.join(npy_path, 'magnitude_after_sup'), magnitude_after_sup)
# imageio.imsave(path.join(output_path, 'magnitude_after_sup.bmp'), magnitude_after_sup)

gx = np.load(path.join(npy_path, 'gx.npy'))
gy = np.load(path.join(npy_path, 'gy.npy'))
magnitude = np.load(path.join(npy_path, 'magnitude.npy'))
magnitude_after_sup = np.load(path.join(npy_path, 'magnitude_after_sup.npy'))
output_1 = canny.thresholding(magnitude_after_sup, 0.1)
output_2 = canny.thresholding(magnitude_after_sup, 0.3)
output_3 = canny.thresholding(magnitude_after_sup, 0.5)
imageio.imsave(path.join(output_path, 'output_1.bmp'), output_1[0])
imageio.imsave(path.join(output_path, 'output_2.bmp'), output_2[0])
imageio.imsave(path.join(output_path, 'output_3.bmp'), output_3[0])

#
# plt.figure(3, figsize = (8,8))
# plt.subplot(311)
# plt.imshow(gx, cmap='gray')
# plt.subplot(312)
# plt.imshow(gy, cmap='gray')
# plt.subplot(313)
# plt.imshow(magnitude, cmap='gray')
# plt.show()
#
# magnitude_after_sup = canny.non_maxima_suppression(gx, gy, magnitude)
# plt.figure(4)
# plt.imshow(magnitude_after_sup, cmap='gray')
# plt.show()
