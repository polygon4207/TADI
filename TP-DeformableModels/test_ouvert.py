import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import io as skio

output_dir = "/Users/erwan/kDrive/Master/TADI/TP-DeformableModels/images"
im = skio.imread("coeurIRM.bmp")

# ============================================================
# Initialisation commune
# ============================================================

s = np.linspace(0, 2 * np.pi, 100)
r = 140 + 25 * np.sin(s)
c = 130 + 25 * np.cos(s)
init = np.array([r, c]).T


# ============================================================
# Valeurs de référence
# ============================================================

default_params = {"sigma": 0.1, "alpha": 0.1, "beta": 150, "w_edge": 50, "gamma": 0.001}


# ============================================================
# Fonction de test
# ============================================================


def test_parameter(im, parameter_name, values, output_dir=None, params=default_params):
    """
    Teste plusieurs valeurs d'un paramètre de active_contour
    en gardant tous les autres paramètres constants.

    parameter_name :
        "sigma", "alpha", "beta", "w_edge" ou "gamma"

    values :
        liste des valeurs à tester
    """

    n = len(values)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    # Si une seule valeur est testée
    if n == 1:
        axes = [axes]

    for ax, value in zip(axes, values):

        # Copie des paramètres de référence
        p = params.copy()

        # Modification uniquement du paramètre étudié
        p[parameter_name] = value

        # Calcul du snake
        snake = active_contour(
            gaussian(im, p["sigma"]),
            init,
            boundary_condition="fixed",
            alpha=p["alpha"],
            beta=p["beta"],
            w_edge=p["w_edge"],
            gamma=p["gamma"],
        )

        # Affichage de l'image
        ax.imshow(im, cmap=plt.cm.gray)

        # Contour initial
        ax.plot(init[:, 1], init[:, 0], "--r", lw=2, label="Initial")

        # Contour final
        ax.plot(snake[:, 1], snake[:, 0], "-b", lw=2, label="Final")

        ax.set_title(f"{parameter_name} = {value}")

        ax.set_xticks([])
        ax.set_yticks([])

        ax.axis([0, im.shape[1], im.shape[0], 0])

    fig.suptitle(f"Influence du paramètre {parameter_name}", fontsize=16)

    fig.tight_layout()

    # Sauvegarde automatique
    if output_dir is not None:

        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"test_{parameter_name}_ouvert.png")

        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

        print("Image sauvegardée :", output_path)

    # plt.show()


test_parameter(im, "alpha", [0.001, 0.1, 1, 10], output_dir)

test_parameter(im, "beta", [0.1, 10, 150, 500], output_dir)

test_parameter(im, "w_edge", [0, 1, 50, 100], output_dir)

test_parameter(im, "gamma", [0.0001, 0.001, 0.01, 1], output_dir)

test_parameter(im, "sigma", [0.01, 0.1, 1, 5], output_dir)
