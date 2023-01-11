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

            # Recuperation de la matrice RGB de chaque image
            for i, name_image in enumerate(list_image):
                _image = io.imread(os.path.join(son_dataset_path, name_image))
                out_image.append(_image)

        return out_image, out_target

def get_data(dataset_name, pourc_dataset=1, ratio_test_val=0.8) -> tuple:

    '''
    Lecture du dataset d'image dataset_name
    Extrait le pourcentage indique du dataset (pourc_dataset) : [0 - 1]
    Melange le dataset
    Divise le dataset en donnees de validation et donnees de tests ratio_test_val : [0 - 1]

    retourne x_train, y_train, x_val, y_val
    '''

    df = pd.DataFrame(columns=['x', 'y'])
    # Nombre de valeurs a extraire du dateset :
    # Pourcentage * nbr de valeurs
    rows = len(df.axes[0]) # Nbr de lignes
    df = df.head(math.ceil(pourc_dataset * rows))

    # Importation des datas
    df.x, df.y = read_csv_dataset(dataset_name)
    df.sample() # melande des donnees

    # Separation dataset Train/Validation
    x_train, x_val, y_train, y_val = train_test_split(df.x, df.y, train_size=ratio_test_val)

    return x_train.to_numpy(), x_val.to_numpy(), y_train.to_numpy(), y_val.to_numpy()

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
        # La matrice est compose de valeur entiere entre 0 et 255
        # le dtype des valeurs donc etre uint8
        # Si compose de Reel la valeur devrait etre entre 0 et 1
        plt.imshow(mat_finale.astype('uint8'))
    else:
        plt.imshow(mat_finale.astype('uint8'), cmap='binary')
    plt.show()


def analyse_apprentissage(**kwargs):
    '''
    En fonction des liste de variable passees en argument
    Realise plusieurs apprentissages en utilisant toutes les combinaisons possibles
    Synthetyse l'evolution des differents apprentissage sur un graph : 1 courbe / combinaison d'argument
    Args:
        try_scale: list des pourcentages de taille de donnees du dataset a utiliser
        try_epoch: list du nombre d'epoch a tester
        fct_model : Liste de fonction qui retourne un modele
        try_dataset : Liste de nom de dataset

    Exemple d'appel : analyse_apprentissage(try_epoch = [2],
                  try_scale = [0.1, 0.2],
                  try_batch = [64],
                  fct_model = ['create_model_v1'],
                  try_dataset = [150_150_L] )
    '''

    # Classement des arguments:
    # Chaque argument est soit un label soit un titre
    # La distinction permet d'annoter correctement le graphique final avec le titre d'une part
    # et les labels des courbes d'autre part

    arg_list = ['try_scale', 'try_epoch', 'try_batch', 'fct_model']
    dict_titre = {}
    dict_label = {}

    # Si j'etudie differente version de ce parametre c'est un label
    assert 'try_scale' in kwargs
    value = kwargs['try_scale']
    if len(value) > 1: # Test si label
        dict_label['try_scale'] = value # Si oui : Valeur enregistree dans le dict de label.
    else:
        dict_titre['try_scale'] = value # Si non : Valeur enregistree dans le dict de titre.

    assert 'try_epoch' in kwargs
    value = kwargs['try_epoch']
    if len(value) > 1: # Test si label
        dict_label['try_epoch'] = value # Si oui : Valeur enregistree dans le dict de label.
    else:
        dict_titre['try_epoch'] = value # Si non : Valeur enregistree dans le dict de titre.

    assert 'try_batch' in kwargs
    value = kwargs['try_batch']
    if len(value) > 1: # Test si label
        dict_label['try_batch'] = value # Si oui : Valeur enregistree dans le dict de label.
    else:
        dict_titre['try_batch'] = value # Si non : Valeur enregistree dans le dict de titre.

    assert 'fct_model' in kwargs
    value = kwargs['fct_model']
    if len(value) > 1:  # Test si label
        dict_label['fct_model'] = value  # Si oui : Valeur enregistree dans le dict de label.
    else:
        dict_titre['fct_model'] = value  # Si non : Valeur enregistree dans le dict de titre.

    assert 'try_dataset' in kwargs
    value = kwargs['try_dataset']
    if len(value) > 1:  # Test si label
        dict_label['try_dataset'] = value  # Si oui : Valeur enregistree dans le dict de label.
    else:
        dict_titre['try_dataset'] = value  # Si non : Valeur enregistree dans le dict de titre.



    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

    # Je parcours les parametre a etudier
    # Combinaison renvoie une liste de dictionnaire contenant chaqu'un une combinaison d'argument unique.
    for each_comb in combinaison(**kwargs):
        # Creation du titre unique base sur les labels
        titre_label = ''
        # Creation du nom ou sera enregistre le fichier log
        titre_fichier_csv = os.path.join(os.getcwd(), ("/learning_log/")) # Adresse enregistrement des fichiers log de l'apprentissage
        for each_label in dict_label.keys():
            titre_label =f"{each_label} : {each_comb[each_label]}   " # Titre du label, compose de tout les elements du dictionnaire label
            titre_fichier_csv = f"{titre_fichier_csv}_{each_label}_{each_comb[each_label]}" # Titre du fichier csv, compose de tout les elements du dictionnaire label
        titre_fichier_csv = titre_fichier_csv + 'training.log'

        # Modification de la taille du dataset
        x_train, Y_train, _, _, _ = read_csv_dataset(f'{datasets_dir}{address_lin_to_win_or_lin("/GTSRB/origine/Train.csv")}',
                                                             name_csv='Dataset_train', r=each_comb['try_scale'])


        x_dataset_train = []
        [x_dataset_train.append(images_enhancement(x, width=24, height=24, mode='RGB')) for x in x_train]
        x_dataset_train = np.asarray(x_dataset_train)

        # Reinitialisation du modele a chaque test
        (lx, ly, lz) = x_dataset_train[0].shape
        a = each_comb['fct_model'] + "(lx, ly, lz)"
        # _model = 0
        _model = eval(a) # Creation du modele
        _model.summary()
        _model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        learning_data = keras.callbacks.CSVLogger(titre_fichier_csv, separator=',')

        # Entrainement du modele
        history = _model.fit(x_dataset_train, Y_train,
                               batch_size=each_comb['try_batch'],
                               epochs=each_comb['try_epoch'],
                               verbose=fit_verbosity,
                               callbacks=[learning_data],
                               validation_data=(x_dataset_test, Y_test))


        # Affichage de la courbe associee

        df = pd.read_csv(titre_fichier_csv)
        y = list(df.val_accuracy)
        x = range(len(y))
        ax1.plot(x, y, label=titre_label)

    # Parametre des courbes
    ax1.set_title("Evolution de l'apprentissage")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation rate')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.grid(axis='y')
    titre_figure = "Annalyse Apprentissage :"
    for key, value in dict_titre.items():
        titre_figure = titre_figure + f"{key}: {value}  "
    ax1.set_title(titre_figure)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x_train, y_train , _, _ = get_data('150_150_RGB', pourc_dataset=0.1)

    display_image(images=x_train, im_ligne=5, dim_im=(150, 150))
    # Affichage des 10 premieres images

    print(y_train[:10])
