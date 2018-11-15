import canny_detector
import imageio
import os

# file_name = 'zebra-crossing-1.bmp'
file_name = 'Lena256.bmp'
script_path = os.path.dirname(os.path.realpath(__file__))
print('Processing the image: ' + file_name)

file_name_without_extension = file_name[:-4]
file_path = os.path.join(script_path, 'input', file_name)
output_path = os.path.join(script_path, 'output', file_name_without_extension)
npy_path = os.path.join(script_path, 'npy', file_name_without_extension)
if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists(npy_path):
    os.mkdir(npy_path)
input_img = imageio.imread(file_path)
output_after_gaussian = canny_detector.gaussian_smoothing(input_img)
imageio.imwrite(os.path.join(output_path, 'after_gaussian.bmp'), output_after_gaussian)
# np.save(os.path.join(npy_path, 'output_after_gaussian'), output_after_gaussian)

# output_after_gaussian = np.load(os.path.join(npy_path, 'output_after_gaussian.npy'))
(gx, gy, magnitude) = canny_detector.gradient_operator(output_after_gaussian)
imageio.imwrite(os.path.join(output_path, 'gx.bmp'), gx)
imageio.imwrite(os.path.join(output_path, 'gy.bmp'), gy)
imageio.imwrite(os.path.join(output_path, 'magnitude_before_sup.bmp'), magnitude)
# np.save(os.path.join(npy_path, 'gx'), gx)
# np.save(os.path.join(npy_path, 'gy'), gy)
# np.save(os.path.join(npy_path, 'magnitude'), magnitude)

# magnitude = np.load(os.path.join(npy_path, 'magnitude.npy'))
# gx = np.load(os.path.join(npy_path, 'gx.npy'))
# gy = np.load(os.path.join(npy_path, 'gy.npy'))
magnitude_after_sup = canny_detector.non_maxima_suppression(gx, gy, magnitude)
imageio.imwrite(os.path.join(output_path, 'magnitude_after_sup.bmp'), magnitude_after_sup)
# np.save(os.path.join(npy_path, 'magnitude_after_sup'), magnitude_after_sup)
# magnitude_after_sup = np.load(os.path.join(npy_path, 'magnitude_after_sup.npy'))


output_1, T_1, num_1 = canny_detector.thresholding(magnitude_after_sup, 0.1)
output_3, T_3, num_3 = canny_detector.thresholding(magnitude_after_sup, 0.3)
output_5, T_5, num_5 = canny_detector.thresholding(magnitude_after_sup, 0.5)

results = open(os.path.join(output_path, 'results.txt'), 'w')
imageio.imwrite(os.path.join(output_path, 'output_1.bmp'), output_1)
imageio.imwrite(os.path.join(output_path, 'output_3.bmp'), output_3)
imageio.imwrite(os.path.join(output_path, 'output_5.bmp'), output_5)
results.write('P    ' + 'Threshold' + '  ' + 'Number of Edge Pixels' + '\n')
results.write('0.1  ' + str(T_1) + '      ' + str(num_1) + '\n')
results.write('0.3  ' + str(T_3) + '      ' + str(num_3) + '\n')
results.write('0.5  ' + str(T_5) + '      ' + str(num_5) + '\n')
results.close()

