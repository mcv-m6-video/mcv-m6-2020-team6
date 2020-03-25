import cv2
import numpy as np
from imageio import imwrite
import os
import numpy as np
from struct import pack, unpack
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from Optical_flow_metrics import flow_read


def Flow_read(flow_dir):
    # cv2 imread ---> BGR  need to converted in RGB format

    im = cv2.imread(flow_dir, cv2.IMREAD_UNCHANGED)
    im_kitti = np.flip(im, axis=2).astype(np.double)

    u_f = (im_kitti[:, :, 0] - 2. ** 15) / 64
    v_f = (im_kitti[:, :, 1] - 2. ** 15) / 64

    f_valid = im_kitti[:, :, 2]
    f_valid[f_valid > 1] = 1

    u_f[f_valid == 0] = 0
    v_f[f_valid == 0] = 0

    flow = np.dstack([u_f, v_f])

    flow_1 = np.dstack([u_f, v_f, f_valid])

    return flow, flow_1


def OF_quiver_visualization(image, flow_dir,step, fname_output='flow_quiver.png'):

    flow, flow_1 = Flow_read(flow_dir)

    u = flow_1[:, :, 0]
    v = flow_1[:, :, 1]
    (h, w) = flow.shape[0:2]
    valid = flow_1[:, :, 2]

    U = u * valid
    V = v * valid
    M = np.hypot(U, V)

    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))

    x = x[::step, ::step]
    y = y[::step, ::step]
    U = U[::step, ::step]
    V = V[::step, ::step]
    M = M[::step, ::step]

    plt.figure()
    img = mpimg.imread(image)
    img = cv2.equalizeHist(img)
    plt.imshow(img, cmap='gray')
    plt.quiver(x, y, U, V, M, scale_units='xy', angles='xy', scale=.5)
    plt.axis('off')
    plt.savefig(fname_output)


def OF_quiver_visualization_flow(image, flow,step, fname_output='flow_quiver.png'):

    u = flow[:, :, 0]
    v = flow[:, :, 1]
    (h, w) = flow.shape[0:2]
    valid = flow[:, :, 2]

    U = u * valid
    V = v * valid
    M = np.hypot(U, V)

    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))

    x = x[::step, ::step]
    y = y[::step, ::step]
    U = U[::step, ::step]
    V = V[::step, ::step]
    M = M[::step, ::step]

    plt.figure()
    img = mpimg.imread(image)
    plt.imshow(img, cmap='gray')
    plt.quiver(x, y, U, V, M, scale_units='xy', angles='xy', scale=.5)
    plt.axis('off')
    plt.savefig(fname_output)

def OF_quiver_visualization_flow2d(image, flow,step, fname_output='flow_quiver.png'):
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    (h, w) = flow.shape[0:2]

    U = u
    V = v
    M = np.hypot(U, V)

    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))

    x = x[::step, ::step]
    y = y[::step, ::step]
    U = U[::step, ::step]
    V = V[::step, ::step]
    M = M[::step, ::step]

    plt.figure()
    img = mpimg.imread(image)
    plt.imshow(img, cmap='gray')
    plt.quiver(x, y, U, V, M, scale_units='xy', angles='xy', scale=.5)
    plt.axis('off')
    plt.savefig(fname_output)

'''
python by:  youngjung uh, Clova ML, Naver
contact:    youngjung.uh@navercorp.com
date:       17 Dec 2018

-------------------------------------------------------------------
----- below comment came from the original (flowToColor.m) --------
-------------------------------------------------------------------
flowToColor(flow, maxFlow) flowToColor color codes flow field, normalize
based on specified value,

flowToColor(flow) flowToColor color codes flow field, normalize
based on maximum flow present otherwise

According to the c++ source code of Daniel Scharstein
Contact: schar@middlebury.edu

Author: Deqing Sun, Department of Computer Science, Brown University
Contact: dqsun@cs.brown.edu
$Date: 2007-10-31 18:33:30 (Wed, 31 Oct 2006) $

Copyright 2007, Deqing Sun.

                        All Rights Reserved

Permission to use, copy, modify, and distribute this software and its
documentation for any purpose other than its incorporation into a
commercial product is hereby granted without fee, provided that the
above copyright notice appear in all copies and that both that
copyright notice and this permission notice appear in supporting
documentation, and that the name of the author and Brown University not be used in
advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

THE AUTHOR AND BROWN UNIVERSITY DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
PARTICULAR PURPOSE.  IN NO EVENT SHALL THE AUTHOR OR BROWN UNIVERSITY BE LIABLE FOR
ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
'''


def computeColor(u, v, cast_uint8=True):
    '''
    args
        u (numpy array) height x width
        v (numpy array) height x width
        cast_uint8 (bool) set False to have image range 0-1 (np.float32)
    return
        img_color (numpy array) height x width x 3
    '''

    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = makeColorwheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1)  # -1~1 maped to 1~ncols

    k0 = np.floor(fk).astype(int)  # 1, 2, ..., ncols

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1

    f = fk - k0

    height, width = u.shape
    img = np.zeros((height, width, 3), np.float32)
    nrows = colorwheel.shape[1]
    for i in range(nrows):
        tmp = colorwheel[:, i]
        col0 = tmp[k0.reshape(-1)] / 255
        col1 = tmp[k1.reshape(-1)] / 255
        col = col0.reshape(height, width) * (1 - f) + \
              col1.reshape(height, width) * f

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])  # increase saturation with radius

        col[np.logical_not(idx)] *= 0.75  # out of range

        img[:, :, i] = col * (1 - nanIdx)

    if cast_uint8:
        img = np.floor(img * 255).astype(np.uint8)
    return img


def makeColorwheel():
    '''
    color encoding scheme
    adapted from the color circle idea described at
    http://members.shaw.ca/quadibloc/other/colint.htm
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros((ncols, 3))  # r g b

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.array(range(RY)) / RY)
    col = col + RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.array(range(YG)) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.array(range(GC)) / GC)
    col = col + GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.array(range(CB)) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.array(range(BM)) / BM)
    col = col + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.array(range(MR)) / MR)
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def flowToColor(flow, maxflow=None, verbose=False):
    '''
    args
        flow (numpy array) height x width x 2
    return
        img_color (numpy array) height x width x 3
    '''

    UNKNOWN_FLOW_THRESH = 5e2
    eps = 1e-6

    height, widht, nBands = flow.shape

    if nBands != 2:
        exit('flowToColor: image must have two bands')

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999
    maxrad = -1

    # fix unknown flow
    idxUnknown = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    maxu = max(maxu, u.max())
    minu = min(minu, u.min())

    maxv = max(maxv, v.max())
    minv = min(minv, v.min())

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(maxrad, rad.max())

    if verbose:
        print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' %
              (maxrad, minu, maxu, minv, maxv))

    if maxflow is not None:
        if maxflow > 0:
            maxrad = maxflow

    u = u / (maxrad + eps)
    v = v / (maxrad + eps)

    img = computeColor(u, v)

    # unknown flow
    # IDX = repmat(idxUnknown, [1, 1, 3])
    img[idxUnknown] = 0

    return img


def OF_visualization(flow_dir, fname_output='flow.png', maxflow=None):
    flow, flow_1 = Flow_read(flow_dir)
    img_result = flowToColor(flow, maxflow)
    imwrite(fname_output, img_result)


def OF_visualization_flow(flow, fname_output='flow.png', maxflow=None):
    img_result = flowToColor(flow, maxflow)
    imwrite(fname_output, img_result)


"""
flow_dir = "D:\\Documents\\Proves\\week1\\datasets\\kitti\\groundtruth\\000045_10.png"
OF_visualization(flow_dir, fname_output='flow.png', maxflow=None)
"""
