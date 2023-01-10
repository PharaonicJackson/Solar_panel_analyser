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

def display_image(images, im_ligne=5, dim_im=(100, 100)):
    '''
    Affiche les images en noir et blanc et les images en couleurs en concatenant les matrices des images
    et en completant les trous par des matrices nulls

    :param images: liste de tableau 3D * [RGB] de l'image
    :im_ligne: nbr d'image affiche par ligne
    :dim_im: tuple de dim 2, resolution de l'image en pixel
    :return:
    '''


    dim_x, dim_y = dim_im  # taille des image

    # Si image en RGB
    if len(images[-1].shape) == 3:
        m, n, o = images[-1].shape
        # Creation de la matrice nulle pour completer les images manquantes de la derniere ligne
        # les 1 de np.ones sont pour afficher des pixels blancs.
        mat_nul = np.ones(shape=(m, n, o))
        mat_finale = np.ones(shape=(1, im_ligne * n, 3))
    # Si image en N et B
    if len(images[-1].shape) == 2:
        m, n = images[-1].shape
        # Creation de la matrice nulle pour completer les images manquantes de la derniere ligne
        # les 1 de np.ones sont pour afficher des pixels blancs.
        mat_nul = np.ones(shape=(m, n))
        mat_finale = np.zeros(shape=(1, im_ligne * n))  # initialise la matrice NB mais 1 seul ligne, puis je concatene sur cette matrice

    # Determination du nombre de ligne dimage
    nbr_image = len(images)
    nbr_ligne = nbr_image // im_ligne  # Nombre de ligne de im_ligne photos
    nbr_photo_der_ligne = nbr_image % im_ligne  # Nombre de photo su la derniere ligne

    num_image = 0
    # affichage des photos par groupe de im_ligne
    for l in range(nbr_ligne):
        # Initialisation de la premiere photo de chaque ligne
        a = images[num_image]
        num_image += 1
        # Affichage des photos le long de la ligne
        for i in range(im_ligne - 1):
            a = np.concatenate((a, images[num_image]), axis=1) # Photos concatenees horizontalement
            num_image += 1
        mat_finale = np.concatenate((mat_finale, a), axis=0)  # En finde ligne, Photos concatenees verticalement
    nbr_mat_null = im_ligne - nbr_photo_der_ligne

    # affichage de la derniere ligne
    if nbr_photo_der_ligne > 0: # Si il y a des photo en fin de ligne
        mat_der_ligne = images[num_image]
        num_image += 1
        # Affichage des photos de la derniere ligne
        for i in range(nbr_photo_der_ligne - 1):
            mat_der_ligne = np.concatenate((mat_der_ligne, images[num_image]), axis=1)
            num_image += 1
        # Je complete par des images blanches pour finir la ligne
        for i in range(nbr_mat_null):
            mat_der_ligne = np.concatenate((mat_der_ligne, mat_nul), axis=1)
        mat_finale = np.concatenate((mat_finale, mat_der_ligne), axis=0)

    # Affichage des images
    if len(images[-1].shape) == 3:
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

    display_image(images=x_train[:10].to_numpy(), im_ligne=5, dim_im=(150, 150))
    # Affichage des 10 premieres images


    print('fin')
