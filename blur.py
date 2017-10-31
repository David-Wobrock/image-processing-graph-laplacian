import matplotlib.pyplot as plt
from scipy import misc
from scipy.ndimage import gaussian_filter


def blur(y):
    return gaussian_filter(y, sigma=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Blur an image by gaussian blur')
    parser.add_argument(
        'image_name',
        type=str,
        nargs=1)

    args = parser.parse_args()
    img_name = args.image_name[0]

    y = misc.imread(img_name)
    z = blur(y)
    cmap = 'gray' if len(z.shape) == 2 else None
    plt.imsave('results/blurred.png', z, cmap='gray')
