#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I. Bloch"""

# %% SECTION 1 inclusion of packages

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

# necessite scikit-image
from skimage import img_as_float
from skimage import io as skio
from skimage.filters import gaussian
from skimage.segmentation import (
    active_contour,
    chan_vese,
    checkerboard_level_set,
    disk_level_set,
)

# %% SECTION 2 - Input image

# im = skio.imread("coeurIRM.bmp")

# im=skio.imread('retineOA.bmp')

# im = skio.imread("brain.bmp")
# im = im[:, :, 1]

im=skio.imread('retina2.png')

# im=skio.imread('brain2.bmp')

# plt.imshow(im, cmap="gray", vmin=0, vmax=255)

# %% SECTION 3a - Segmentation using active contours

# s = np.linspace(0, 2 * np.pi, 100)
# r = 140 + 25 * np.sin(s)
# c = 130 + 25 * np.cos(s)
# init = np.array([r, c]).T

# snake = active_contour(
#     gaussian(im, 0.1), init, alpha=0.1, beta=150, w_edge=50, gamma=0.001
# )

# fig, ax = plt.subplots(figsize=(7, 7))
# ax.imshow(im, cmap=plt.cm.gray)
# ax.plot(init[:, 1], init[:, 0], "--r", lw=3)
# ax.plot(snake[:, 1], snake[:, 0], "-b", lw=3)
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, im.shape[1], im.shape[0], 0])

# plt.show()

# %% SECTION 3b - Open contours

# Use retineOA.bmp

x1 = 185
x2 = 254
y1 = 10
y2 = 72

r = np.linspace(y1, y2, 100)
c = np.linspace(x1, x2, 100)
init = np.array([r, c]).T

snake = active_contour(
    gaussian(im, 1),
    init,
    boundary_condition="fixed",
    alpha=0.05,
    beta=5,
    w_line=10,
    w_edge=2,
    gamma=0.001,
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.imshow(im, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], "--r", lw=3)
ax.plot(snake[:, 1], snake[:, 0], "-b", lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, im.shape[1], im.shape[0], 0])

plt.show()

# %% SECTION 4 - Segmentation using level sets (and region homogeneity)

# image = img_as_float(im)

# # Init avec un damier
# # init_ls = checkerboard_level_set(image.shape, 6)

# # Init avec un cercle
# init_ls = disk_level_set(image.shape)

# # Init avec plusieurs cercles
# # circleNum = 8
# # circleRadius = image.shape[0] / (3 * circleNum)
# # circleStep0 = image.shape[0] / (circleNum + 1)
# # circleStep1 = image.shape[1] / (circleNum + 1)
# # init_ls = np.zeros(image.shape)
# # for i in range(circleNum):
# #     for j in range(circleNum):
# #         init_ls = init_ls + disk_level_set(
# #             image.shape, ((i + 1) * circleStep0, (j + 1) * circleStep1), circleRadius
# #         )


# cv = chan_vese(
#     image,
#     mu=0.1,
#     lambda1=6,
#     lambda2=1,
#     tol=0.00001,
#     max_num_iter=1000,
#     dt=0.1,
#     init_level_set=init_ls,
#     extended_output=True,
# )

# fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ax = axes.flatten()

# ax[0].imshow(image, cmap="gray")
# ax[0].set_axis_off()
# ax[0].set_title("Original Image", fontsize=12)

# ax[1].imshow(cv[0], cmap="gray")
# ax[1].set_axis_off()
# title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
# ax[1].set_title(title, fontsize=12)

# ax[2].imshow(cv[1], cmap="gray")
# ax[2].set_axis_off()
# ax[2].set_title("Final Level Set", fontsize=12)

# ax[3].plot(cv[2])
# ax[3].set_title("Evolution of energy over iterations", fontsize=12)

# fig.tight_layout(pad=2.0, w_pad=3.0, h_pad=3.0)

# plt.show()


# %% END  TP - Deformable Models
