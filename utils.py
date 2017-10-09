import numpy as np

# https://stackoverflow.com/a/30110497
def im2col(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape[:2]
    s0, s1 = A.strides[:2]
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]

# https://stackoverflow.com/a/17201686
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):

    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# From https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py
yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                         [-0.14714119, -0.28886916,  0.43601035 ],
                         [ 0.61497538, -0.51496512, -0.10001026 ]])

rgb_from_yuv = np.linalg.inv(yuv_from_rgb)


def rgb2ycc(im):
    return np.dot(im, yuv_from_rgb)

def ycc2rgb(im):
    return np.dot(im, rgb_from_yuv)

def xy2num(x, y, M, N):
    return int(x*N + y)

def num2xy(num, M, N):
    y = int(np.floor(num / M))
    x = int(num - (y)*M)
    return x, y
