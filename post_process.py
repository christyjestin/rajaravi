import numpy as np
from numba import njit

# applies a gaussian kernel to an image i.e. iterate over an image
# and replace each pixel with a weighted average of pixels in its
# neighborhood where closer neighbors are given more weight
# N.B. this process has been parallelized so that the for loops are over
# the kernel instead of over the image
@njit
def post_process_gaussian(img: np.ndarray, kernel_size: int):
    assert isinstance(kernel_size, int) and (kernel_size % 2 == 1), \
        f"Kernel size {kernel_size} must be an odd integer"

    half = kernel_size // 2
    output = np.zeros_like(img, dtype=np.float32)
    output_scale = np.zeros_like(img, dtype=np.float32)
    img = img.astype(np.float32)
    m, n, _ = img.shape
    temperature = half ** 2
    for i in range(half + 1):
        for j in range(half + 1):
            # the scale is the weighting in the weighted average
            scale = np.exp(-1 * (i ** 2 + j ** 2) / temperature)
            scaled_img = img * scale
            # add scaled and shifted versions of the image to itself
            output[i:, j:] += scaled_img[:m - i, :n - j]
            output[:m - i, j:] += scaled_img[i:, :n - j]
            output[i:, :n - j] += scaled_img[:m - i, j:]
            output[:m - i, j:] += scaled_img[i:, j:]

            # do the same for the scale values so that we can normalize later
            output_scale[i:, j:] += scale
            output_scale[:m - i, j:] += scale
            output_scale[i:, :n - j] += scale
            output_scale[:m - i, j:] += scale

    return output / output_scale

# similar to the gaussian version but instead of taking a fixed weighted average,
# you sample from your neighborhood of pixels with closer neighbors having a higher
# probability of being included
# N.B. the temperature parameter controls how quick the drop off in probability is
# as you get further away from the current pixel
# N.B. it is now a conventional average but with a varying # of terms
def post_process_random(img: np.ndarray, kernel_size: int, temperature: float):
    assert isinstance(kernel_size, int) and (kernel_size % 2 == 1), \
        f"Kernel size {kernel_size} must be an odd integer"

    half = kernel_size // 2
    output = np.zeros_like(img, dtype=np.float32)
    output_scale = np.zeros_like(img, dtype=np.float32)
    img = img.astype(np.float32)
    m, n, _ = img.shape
    for i in range(half + 1):
        for j in range(half + 1):
            # only consider neighbors within the circle centered at the current pixel
            if np.sqrt(i ** 2 + j ** 2) > half ** 2:
                break
            scale = np.exp(-1 * np.sqrt(i ** 2 + j ** 2) / temperature)
            random_scale = lambda: np.random.rand(m, n, 1) < scale
            # add scaled and shifted versions of the image to itself
            # do the same for the scaling values so that we can normalize later
            left_up_scale = random_scale()
            left_down_scale = random_scale()
            right_up_scale = random_scale()
            right_down_scale = random_scale()

            output[i:, j:] += (img * right_down_scale)[:m - i, :n - j]
            output[:m - i, j:] += (img * right_up_scale)[i:, :n - j]
            output[i:, :n - j] += (img * left_down_scale)[:m - i, j:]
            output[:m - i, :n - j] += (img * left_up_scale)[i:, j:]

            
            output_scale[i:, j:] += right_down_scale[:m - i, :n - j]
            output_scale[:m - i, j:] += right_up_scale[i:, :n - j]
            output_scale[i:, :n - j] += left_down_scale[:m - i, j:]
            output_scale[:m - i, :n - j] += left_up_scale[i:, j:]

    return output / output_scale