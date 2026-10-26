import itertools
import os
import matplotlib.pyplot as plt
import numpy as np

from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import io as skio

output_dir = "/Users/erwan/kDrive/Master/TADI/TP-DeformableModels/images"
im = skio.imread("coeurIRM.bmp")

# Initialisation du contour
s = np.linspace(0, 2 * np.pi, 100)
r = 140 + 25 * np.sin(s)
c = 130 + 25 * np.cos(s)
init = np.array([r, c]).T


# Valeur fixe du lissage gaussien
gau = 0.1


# Valeurs min / max
alpha_values = [0.001, 10]
beta_values = [1, 500]
w_edge_values = [1, 200]
gamma_values = [0.0001, 0.1]


combinations = list(
    itertools.product(alpha_values, beta_values, w_edge_values, gamma_values)
)


fig, axes = plt.subplots(4, 4, figsize=(16, 16))

axes = axes.flatten()


for idx, (ax, (alpha, beta, w_edge, gamma)) in enumerate(
    zip(axes, combinations), start=1
):

    snake = active_contour(
        gaussian(im, gau), init, alpha=alpha, beta=beta, w_edge=w_edge, gamma=gamma
    )

    ax.imshow(im, cmap=plt.cm.gray)

    # Contour initial
    ax.plot(init[:, 1], init[:, 0], "--r", lw=1.5)

    # Contour final
    ax.plot(snake[:, 1], snake[:, 0], "-b", lw=1.5)

    ax.set_title(
        f"#{idx}\n" f"α={alpha}, β={beta}\n" f"edge={w_edge}, γ={gamma}", fontsize=9
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.axis([0, im.shape[1], im.shape[0], 0])


fig.suptitle("Combinaisons min/max des paramètres", fontsize=18)

fig.tight_layout()

output_path = os.path.join(output_dir, "combinaisons_parametres.png")

fig.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()
