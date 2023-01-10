import os
import imageio.v2 as io
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # pour resoudre ce message d'erreur OMP: Error #15: Initializing libiomp5, but found libiomp5md.dll already initialized.
from fonctions_utiles import combinaison, Timing, DimError
import re
import csv
import math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from matplotlib.ticker import MultipleLocator
from skimage import io, color, exposure, transform
from importlib import reload
from sklearn.model_selection import train_test_split

# Declaration des variable
datasets_dir = os.path.join(os.getcwd(), '/datasets-fidle')
batch_size = 64
epochs = 8
fit_verbosity = 1
scale = 0.2  # pourcentage du dataset a traiter


def read_csv_dataset(parent_dataset_name):

    '''
    Extrait les images de plusieurs dossiers fils situe dans un dossier parent  dans le folder du main -> parent_dataset_name
    Recupere des matrice numpy x l'image et y le nombre de panneaux
    Le fichier csv contenant les informations 'name' -> nom de l'image et 'id'-> y sont dans le fichier d'image

    name : nom du fichier ou seront enregistrees les valeurs :
    -   le nom des images : Name
    -   leur nombres de panneaux respectifs : ClassId

    '''

    parent_dataset_path = os.path.join(os.getcwd(), parent_dataset_name)  # Fichier de fichiers d'images

    out_image = [] # list final qui contient les images
    out_target = [] # list final qui contient le nombre de panneaux

    # Lecture de l'ensemble des fichier csv
    for _, son_dataset_name, _ in os.walk(parent_dataset_path):
        for dataset_name in son_dataset_name:
            son_dataset_path = os.path.join(parent_dataset_path, dataset_name)
            nom_fichier_csv = dataset_name + '.csv' # Fichier csv enregistre avec le nom du fichier + csv
            df = pd.read_csv(os.path.join(son_dataset_path, nom_fichier_csv))
            list_image = df['name'].to_list()
            out_target = out_target + (df['id'].to_list())

            # Parcours de toutes les images et ajout de la mtrice de l'image
            for i, name_image in enumerate(list_image):
                _image = io.imread(os.path.join(son_dataset_path, name_image))
                out_image.append(_image)

        return out_image, out_target

def display_image(images,  im_ligne=5, dim_im=(100, 100)):
    '''
    Affiche les images en noir et blanc et les images en couleurs

    :param images: liste de tableau 3D * [RGB] de l'image
    :im_ligne: nbr d'image affiche par ligne
    :dim_im: tuple de dim 2, resolution de l'image en pixel
    :param lx: largeur de l'image
    :param ly: hauteur de l'image
    :return:
    '''

    if target and len(images) != len(target):
        raise DimError("nombre d'image", "cible des images", len(images), len(target))

    dim_x, dim_y = dim_im  # taille des image
    x_meta_, Y_meta_, _, _, _ = read_csv_dataset(f'{datasets_dir}/GTSRB/origine/Meta.csv',
                                                 name_csv='Dataset_meta', r=1)
    # Creation bibliotheque de panneau officiel par numero
    panneau = {}

    x_train_resized = []
    # uniformisation de la taille des images
    [x_train_resized.append(transform.resize(img, (dim_x, dim_y))) for img in images]

    # Si image en RGB
    if len(x_train_resized[-1].shape) == 3:
        m, n, o = x_train_resized[-1].shape
        mat_nul = np.ones(
            shape=(m, n, o))  # Creation de la matrice nulle pour completer les images manquante de la derniere ligne
        mat_finale = np.zeros(shape=(
            1, im_ligne * n, 3))  # initialise la matrice RVB mais 1 seul ligne, puis je concatene sur cette matrice

        # Panneau officiels charges et uniformisation de la taille charge en RGB
        for compteur, img_pan in enumerate(x_meta_):  # L'image de ce ta set est en RGB-A et pas RGB
            img_pan = np.delete(img_pan, (3), axis=2)  # Je supprime donc la 4 valeur de la 3 eme dimension des couleurs
            panneau[Y_meta_[compteur]] = images_enhancement(img_pan, width=dim_x, height=dim_y, mode='RGB')
            # panneau[Y_meta_[compteur]] = transform.resize(img_pan, (dim_x, dim_y))

    # Si image en N et B
    if len(x_train_resized[-1].shape) == 2:
        m, n = x_train_resized[-1].shape
        mat_nul = np.ones(
            shape=(m, n))  # Creation de la matrice nulle pour completer les images manquante de la derniere ligne
        mat_finale = np.zeros(shape=(
            1, im_ligne * n))  # initialise la matrice NB mais 1 seul ligne, puis je concatene sur cette matrice
        # Panneau officiels charges et uniformisation de la taille charge en RGB
        for compteur, img_pan in enumerate(x_meta_):  # L'image de ce ta set est en RGB-A et pas RGB
            img_pan = np.delete(img_pan, (3), axis=2)  # Je supprime donc la 4 valeur de la 3 eme dimension des couleurs
            panneau[Y_meta_[compteur]] = images_enhancement(img_pan, width=dim_x, height=dim_y, mode='L')
            # panneau[Y_meta_[compteur]] = transform.resize(img_pan, (dim_x, dim_y))

    # Determination du nombre de ligne dimage
    nbr_image = len(x_train_resized)
    nbr_ligne = nbr_image // im_ligne  # Nombre de ligne de 5 photos
    nbr_photo_der_ligne = nbr_image % im_ligne  # Nombre de photo su la derniere ligne
    num_image = 0

    # affichage des photos par groupe de im_ligne
    for l in range(nbr_ligne):
        # Initialisation de la premiere photo de chaque ligne
        if target:
            a = np.concatenate((x_train_resized[num_image], panneau[target[num_image]]), axis=0)
        else:
            a = x_train_resized[num_image]
        num_image += 1
        if target:
            for i in range(im_ligne - 1):
                b = np.concatenate((x_train_resized[num_image], panneau[target[num_image]]), axis=0)
                a = np.concatenate((a, b), axis=1)
                num_image += 1
        else:
            for i in range(im_ligne - 1):
                a = np.concatenate((a, x_train_resized[num_image]), axis=1)
                num_image += 1
        mat_finale = np.concatenate((mat_finale, a), axis=0)
    nbr_mat_null = im_ligne - nbr_photo_der_ligne

    # affichage de la derniere ligne
    if nbr_photo_der_ligne > 0:
        if target:
            mat_der_ligne = np.concatenate((x_train_resized[num_image], panneau[target[num_image]]), axis=0)
            num_image += 1
            # Affichage des photos de la derniere ligne
            for i in range(nbr_photo_der_ligne - 1):
                b = np.concatenate((x_train_resized[num_image], panneau[target[num_image]]), axis=0)
                mat_der_ligne = np.concatenate((mat_der_ligne, b), axis=1)
                num_image += 1
            # Je complete par des images blanches pour finir la ligne
            for i in range(nbr_mat_null):
                b = np.concatenate((mat_nul, mat_nul), axis=0)
                mat_der_ligne = np.concatenate((mat_der_ligne, b), axis=1)
            mat_finale = np.concatenate((mat_finale, mat_der_ligne), axis=0)

        else:
            mat_der_ligne = x_train_resized[num_image]
            num_image += 1
            # Affichage des photos de la derniere ligne
            for i in range(nbr_photo_der_ligne - 1):
                mat_der_ligne = np.concatenate((mat_der_ligne, x_train_resized[num_image]), axis=1)
                num_image += 1
            # Je complete par des images blanches pour finir la ligne
            for i in range(nbr_mat_null):
                mat_der_ligne = np.concatenate((mat_der_ligne, mat_nul), axis=1)
            mat_finale = np.concatenate((mat_finale, mat_der_ligne), axis=0)

    # Affichage des images
    if len(x_train_resized[-1].shape) == 3:
        plt.imshow(mat_finale)
    else:
        plt.imshow(mat_finale, cmap='binary')
    plt.show()

if __name__ == '__main__':

    df = pd.DataFrame(columns=['x', 'y'])
    # Importation des datas
    df.x, df.y = read_csv_dataset('150_150_L')
    df.sample()

    # Separation dataset Train/Validation
    x_train, x_val, y_train, y_val = train_test_split(df.x, df.y, train_size=0.8)

    display_image(images=x_train[:10], im_ligne=5, dim_im=(150, 150))
    # Affichage des 10 premieres images


    print('fin')
