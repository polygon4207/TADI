import os

import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage import io as skio
from skimage.segmentation import chan_vese, disk_level_set

output_dir = "/Users/erwan/kDrive/Master/TADI/TP-DeformableModels/images"

im = skio.imread("coeurIRM.bmp")
image = img_as_float(im)


# ============================================================
# Initialisation identique au code de base
# ============================================================

init_ls = disk_level_set(image.shape)


# ============================================================
# Valeurs de référence identiques au code de base
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
# Fonction de test
# ============================================================


def test_parameter(
    image, parameter_name, values, output_dir=None, params=default_params
):
    """
    Teste plusieurs valeurs d'un paramètre de chan_vese
    en gardant tous les autres paramètres à leur valeur de référence.

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

        # Chan-Vese : même appel que dans le code de base
        cv = chan_vese(
            image,
            mu=p["mu"],
            lambda1=p["lambda1"],
            lambda2=p["lambda2"],
            tol=p["tol"],
            max_num_iter=p["max_num_iter"],
            dt=p["dt"],
            init_level_set=init_ls,
            extended_output=True,
        )

        # Affichage de la segmentation finale
        ax.imshow(cv[0], cmap="gray")
        ax.set_axis_off()
        ax.set_title(f"{parameter_name} = {value}", fontsize=12)

    fig.suptitle(f"Influence du paramètre {parameter_name}", fontsize=16)

    fig.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"test_{parameter_name}_chan_vese.png")

        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

        print("Image sauvegardée :", output_path)

    # plt.show()


# ============================================================
# Tests : un seul paramètre varie à la fois
# ============================================================

test_parameter(image, "mu", [0.01, 0.1, 0.5, 1], output_dir)

test_parameter(image, "lambda1", [0.1, 1, 5.5, 10], output_dir)

test_parameter(image, "lambda2", [0.1, 1, 5, 10], output_dir)

test_parameter(image, "tol", [1e-2, 1e-3, 1e-4, 1e-5], output_dir)

test_parameter(image, "max_num_iter", [20, 50, 200, 500], output_dir)

test_parameter(image, "dt", [0.1, 0.5, 1, 2], output_dir)
