import canny
import imageio
from os import path
from os import mkdir

# file_name = 'zebra-crossing-1.bmp'
file_name = 'Lena_test.bmp'
file_name_without_extension = file_name[:-4]
file_path = path.join(path.dirname(__file__), 'input', file_name)

output_path = path.join(path.dirname(__file__), 'output', file_name_without_extension)
npy_path = path.join(path.dirname(__file__), 'npy', file_name_without_extension)
if not path.exists(output_path):
    mkdir(output_path)
if not path.exists(npy_path):
    mkdir(npy_path)
input_img = imageio.imread(file_path)
output_after_gaussian = canny.gaussian_smoothing(input_img)
imageio.imwrite(path.join(output_path, 'after_gaussian.bmp'), output_after_gaussian)
# np.save(path.join(npy_path, 'output_after_gaussian'), output_after_gaussian)

# output_after_gaussian = np.load(path.join(npy_path, 'output_after_gaussian.npy'))
(gx, gy, magnitude) = canny.gradient_operator(output_after_gaussian)
imageio.imwrite(path.join(output_path, 'gx.bmp'), gx)
imageio.imwrite(path.join(output_path, 'gy.bmp'), gy)
imageio.imwrite(path.join(output_path, 'magnitude.bmp'), magnitude)
# np.save(path.join(npy_path, 'gx'), gx)
# np.save(path.join(npy_path, 'gy'), gy)
# np.save(path.join(npy_path, 'magnitude'), magnitude)

# magnitude = np.load(path.join(npy_path, 'magnitude.npy'))
# gx = np.load(path.join(npy_path, 'gx.npy'))
# gy = np.load(path.join(npy_path, 'gy.npy'))
magnitude_after_sup = canny.non_maxima_suppression(gx, gy, magnitude)
imageio.imwrite(path.join(output_path, 'magnitude_after_sup.bmp'), magnitude_after_sup)
# np.save(path.join(npy_path, 'magnitude_after_sup'), magnitude_after_sup)
# magnitude_after_sup = np.load(path.join(npy_path, 'magnitude_after_sup.npy'))


output_1 = canny.thresholding(magnitude_after_sup, 0.1)
output_2 = canny.thresholding(magnitude_after_sup, 0.3)
output_3 = canny.thresholding(magnitude_after_sup, 0.5)
imageio.imwrite(path.join(output_path, 'output_1.bmp'), output_1[0])
imageio.imwrite(path.join(output_path, 'output_2.bmp'), output_2[0])
imageio.imwrite(path.join(output_path, 'output_3.bmp'), output_3[0])
