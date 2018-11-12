from matplotlib import pyplot as plt
import canny
import numpy as np

# output_after_gaussian = gaussian_smoothing(input)

# plt.figure(1)
# # plt.subplot(221)
# plt.imshow(input, cmap='gray')
# plt.show()
# plt.figure(2)
# # plt.subplot(222)
# plt.imshow(output_after_gaussian, cmap='gray')
# plt.show()
# np.save('output_after_gaussian', output_after_gaussian)

input = np.load('output_after_gaussian.npy')

(gradient_x, gradient_y) = canny.gradient_operator(input)
#
plt.figure(3)
plt.title('Gradient Responses')
plt.subplot(211)
plt.imshow(gradient_x, cmap='gray')
plt.subplot(212)
plt.imshow(gradient_y, cmap='gray')
plt.show()
