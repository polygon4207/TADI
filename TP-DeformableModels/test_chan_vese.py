import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float
from skimage import io as skio
from skimage.segmentation import (chan_vese, checkerboard_level_set,
                                  disk_level_set)

# ============================================================
# Entrée / sortie
# ============================================================

output_dir = "/Users/erwan/kDrive/Master/TADI/TP-DeformableModels/images"
im = skio.imread("coeurIRM.bmp")
image = img_as_float(im)


# ============================================================
# Initialisation commune : disque
# ============================================================

init_ls = disk_level_set(image.shape)


# ============================================================
# Valeurs de référence (code fourni dans le TP)
# ============================================================

default_params = {
    "mu": 0.1,
    "lambda1": 5.5,
    "lambda2": 1,
    "tol": 1e-3,
    "max_num_iter": 200,
    "dt": 0.5,
}


# ============================================================
# Fonction de test d'un paramètre
# ============================================================


def test_parameter(
    image,
    parameter_name,
    values,
    output_dir=None,
    params=default_params,
    init_level_set=init_ls,
):
    """
    Teste plusieurs valeurs d'un paramètre de chan_vese
    en gardant tous les autres paramètres constants.

    parameter_name :
        "mu", "lambda1", "lambda2", "tol", "max_num_iter" ou "dt"

    values :
        liste des valeurs à tester
    """

    n = len(values)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, value in zip(axes, values):
        # Copie des paramètres de référence
        p = params.copy()

        # Modification uniquement du paramètre étudié
        p[parameter_name] = value

        # Segmentation Chan-Vese
        cv = chan_vese(
            image,
            mu=p["mu"],
            lambda1=p["lambda1"],
            lambda2=p["lambda2"],
            tol=p["tol"],
            max_num_iter=p["max_num_iter"],
            dt=p["dt"],
            init_level_set=init_level_set,
            extended_output=True,
        )

        segmentation = cv[0]
        energies = cv[2]

        # Image originale
        ax.imshow(image, cmap="gray")

        # Level set initial en rouge pointillé
        ax.contour(
            init_level_set,
            levels=[0.5],
            colors="red",
            linestyles="--",
            linewidths=2,
        )

        # Contour final de la segmentation en bleu
        ax.contour(
            segmentation.astype(float),
            levels=[0.5],
            colors="blue",
            linewidths=2,
        )

        ax.set_title(f"{parameter_name} = {value}\n" f"{len(energies)} itérations")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Influence du paramètre {parameter_name}", fontsize=16)
    fig.tight_layout()

    # Sauvegarde automatique
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f"test_{parameter_name}_chan_vese.png",
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

        print("Image sauvegardée :", output_path)

    plt.close(fig)


# ============================================================
# Test des différentes initialisations du level set
# ============================================================


def create_multiple_disks(shape, circle_num=8):
    """Construit une initialisation contenant plusieurs petits disques."""

    circle_radius = shape[0] / (3 * circle_num)
    circle_step0 = shape[0] / (circle_num + 1)
    circle_step1 = shape[1] / (circle_num + 1)

    multiple_disks = np.zeros(shape, dtype=float)

    for i in range(circle_num):
        for j in range(circle_num):
            disk = disk_level_set(
                shape,
                center=((i + 1) * circle_step0, (j + 1) * circle_step1),
                radius=circle_radius,
            )
            multiple_disks = np.maximum(multiple_disks, disk)

    return multiple_disks


def test_initializations(image, output_dir=None, params=default_params):
    """Compare les initialisations proposées dans le code du TP."""

    initializations = {
        "Damier": checkerboard_level_set(image.shape, 6),
        "Disque": disk_level_set(image.shape),
        "Plusieurs disques": create_multiple_disks(image.shape, circle_num=8),
    }

    fig, axes = plt.subplots(1, len(initializations), figsize=(15, 5))

    for ax, (name, initial_level_set) in zip(axes, initializations.items()):
        cv = chan_vese(
            image,
            mu=params["mu"],
            lambda1=params["lambda1"],
            lambda2=params["lambda2"],
            tol=params["tol"],
            max_num_iter=params["max_num_iter"],
            dt=params["dt"],
            init_level_set=initial_level_set,
            extended_output=True,
        )

        segmentation = cv[0]
        energies = cv[2]

        ax.imshow(image, cmap="gray")

        # Initialisation en rouge pointillé
        ax.contour(
            initial_level_set,
            levels=[0.5],
            colors="red",
            linestyles="--",
            linewidths=1.5,
        )

        # Segmentation finale en bleu
        ax.contour(
            segmentation.astype(float),
            levels=[0.5],
            colors="blue",
            linewidths=2,
        )

        ax.set_title(f"{name}\n{len(energies)} itérations")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Influence de l'initialisation du level set", fontsize=16)
    fig.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            "test_initialisation_chan_vese.png",
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )

        print("Image sauvegardée :", output_path)

    plt.close(fig)


# ============================================================
# Tests : un paramètre varie, les autres restent à leur valeur
# de référence.
# ============================================================

# # Régularisation / longueur du contour
# test_parameter(image, "mu", [0.01, 0.1, 0.5, 1], output_dir)

# # Poids d'homogénéité de la région intérieure
# test_parameter(image, "lambda1", [0.1, 1, 5.5, 10], output_dir)

# # Poids d'homogénéité de la région extérieure
# test_parameter(image, "lambda2", [0.1, 1, 5, 10], output_dir)

# # Critère d'arrêt
# test_parameter(image, "tol", [1e-1, 1e-2, 1e-3, 1e-4], output_dir)

# # Nombre maximal d'itérations
# test_parameter(image, "max_num_iter", [10, 50, 200, 500], output_dir)

# # Pas temporel
# test_parameter(image, "dt", [0.1, 0.5, 1, 2], output_dir)

# Comparaison des trois initialisations proposées dans le TP
# test_initializations(image, output_dir)
