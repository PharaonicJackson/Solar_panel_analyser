'''
Le but de ce programme est d'importer des images situes dans un fichier a passer en argument
- Normaliser la taille des photos (size)
- Creer des variance d'affichage RGB | RGB-HE | L | L-HE | L-LHE | L-CLAHE
- Enregistrer en format csv dans le fichier output dataset
'''
import datetime
import os, sys, shutil
from datetime import datetime
import pandas as pd
from fonctions_utiles import combinaison
from skimage import io, color, exposure, transform
import matplotlib.pyplot as plt

class NameError(Exception):
    def __init__(self, key_word):
        message = f"{key_word} n'est pas un mot clef compatible. Veuillew choisir parmi les suivants RGB | RGB-HE | L | L-HE | L-LHE | L-CLAHE "
        super().__init__(message)


def create_uniformized_dataset(dataset_dir, csv_name, r=1, header=0, **kwargs):
    '''
    Creer different dataset composes des combinaison des parametres en argument (parmetre multi ci-dessous, appeler em liste, cf exemple).
    Les images et le fichier csv sont enregistre dans le meme dossier d'adresse.

    dataset_dir : Nom du fichier source des datasets situe dans le meme dossier que le main.
    csv_name : Nom du fichier csv en relation avec les images, ce fichier doit etre dans le dossier des images.
    csv_name contient le chemin depuis le main de cette fonction vers chacune des images.
    r : % des images importes
    traitement : Niveau de couleur de l'image desire (parmetre multi)
    size : largeur / hauteur uniformisees en pixel des images du dataset (parmetre multi)
    header : Presence de header dans le dataset -> 1 sinon 0

    '''

    traitement_possible = ['RGB', 'RGB-HE', 'L', 'L-HE', 'L-LHE', 'L-CLAHE']

    if 'traitement' in kwargs:
        traitement = kwargs['traitement']
    else:
        traitement = 'RGB'

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = (20, 20)

    for each_comb in combinaison(dataset_dir=dataset_dir, csv_name=[csv_name], size=size, r=[r],
                                 traitement=traitement, header=[header]):  # Creation de chaque dataset unique

        # Recuperation des parametres uniques
        dataset_dir = each_comb['dataset_dir']
        csv_name = each_comb['csv_name']
        size = each_comb['size']
        traitement = each_comb['traitement']
        r = each_comb['r']
        header = each_comb['header']

        if traitement not in traitement_possible:
            raise NameError(traitement)

        width, height = size

        folder_to_save = f"{dataset_dir}_{width}_{height}_{traitement}"  # Nom du dossier de sauvegarde des nouvelles photos
        directory_source = os.getcwd()  # .
        nom_fichier_csv = f'/{dataset_dir}/{csv_name}' # Fichier csv ou recuperer les infos
        adr_csv = os.path.join(directory_source + nom_fichier_csv)
        df = pd.read_csv(adr_csv, header=header)  # Informations relatives aux photos a extraire

        # Creation du fichier de sauvegarde du dataset
        # Si le dossier existe deja, je le supprime
        [shutil.rmtree(f) for f in os.listdir(directory_source) if f == (f"{folder_to_save}")]

        # Dossier ou seront enregistrees les images
        os.mkdir(f"./{folder_to_save}")

        nbr_row = df.iloc[:, 0].count()  # Recupearion d'un pourcentage du nombre de photo total du fichier
        df = df.head(int(r * nbr_row))
        filenames = df['Name'].to_list()  # Recuperation du nom de l'image
        y = df['ClassId'].to_list()  # Recuperation des classes des images

        # ajout des images dans la colone x
        x = []

        for i, filename in enumerate(filenames):
            filename = os.path.join(directory_source, dataset_dir, filename)
            image = io.imread(address_lin_to_win_or_lin(filename))
            image = image_augmentation(image, width=width, height=height, mode=traitement)
            io.imsave(f"{directory_source}/{folder_to_save}/{folder_to_save}_image_{i}.png",
                      image)  # Enregistrement de la photo
            x.append(f"{folder_to_save}_image_{i}.png")

        # Sauvegarde du fichier csv
        df_out = pd.DataFrame(list(zip(x, y)), columns=['name', 'id'])
        name_csv_out = f"{directory_source}/{folder_to_save}/{folder_to_save}.csv"
        df_out.to_csv(name_csv_out)


def image_augmentation(image, width=25, height=25, mode='RGB'):
    '''
    image : natrice
    modifie la taille selon valeur width et height
    Les images d'entrees sont soit NB soit RGB soit RGBA.
    Si RGBA, l'image est redimensionne en 3 dim comme RGB

    args:
        images :         liste d'images
        width,height :   nouvelle taille des images
        mode :           RGB | RGB-HE | L | L-HE | L-LHE | L-CLAHE
    return:
        numpy array des images modifiees
    '''
    mode = {'RGB': 3, 'RGB-HE': 3, 'L': 1, 'L-HE': 1, 'L-LHE': 1, 'L-CLAHE': 1}

    # Si RGBA, conversion en RGB
    if image.shape[2] == 4:
        image = color.rgba2rgb(image)

    # -Modification de la taille
    image = transform.resize(image, (width, height))

    # RGB + histogram Egalisation
    if mode == 'RGB-HE':
        hsv = color.rgb2hsv(image.reshape(width, height, 3))
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        image = color.hsv2rgb(hsv)

    # Niveau de gris
    if mode == 'L':
        image = color.rgb2gray(image)

    # Niveau de gris + histogram Egalisation
    if mode == 'L-HE':
        image = color.rgb2gray(image)
        image = exposure.equalize_hist(image)

    # Niveau de gris + egalisatiom
    if mode == 'L-LHE':
        image = color.rgb2gray(image)
        image = img_as_ubyte(image)
        image = rank.equalize(image, disk(10)) / 255.

    # Niveau de gris (CLAHE)
    if mode == 'L-CLAHE':
        image = color.rgb2gray(image)
        image = exposure.equalize_adapthist(image)

    return image


def address_lin_to_win_or_lin(adresse: str) -> str:
    '''
    Convertie une chaine de caractere d'adresse (linux -> windows ou linux) = sep / -> / ou \\

    sur windows : '/home/linux' -> \\home\\linux
    '''

    if '/' in adresse:
        ancetres = adresse.split('/')

    if '\\' in adresse:
        ancetres = adresse.split('\\')

    if sys.platform == "win32":
        out = '\\'.join(ancetres)  # + '\\\\'
    else:
        out = '/'.join(ancetres)  # + '/'
    return out


def create_fichier_csv_dataset(dataset_dir):
    '''
    Pour chaque photo du fichier cible, la photo est affiche.
    L'utilsateur entre la valeur de la cible
    Une fois valide, la photo est stochee dans un fichier et le fichier.csv est update
    :return:
    '''

    dir_to_save = f"New Dataset_{str(datetime.now().today()).replace(':','.')}"
    directory_source = os.getcwd()

    # Creation du directory ou seront les datasets
    _ = False
    for f in os.listdir(directory_source + '/' + dataset_dir):
        if f == dir_to_save:
            _ = True
            break

    if _ == False:
        os.mkdir(f"./{dataset_dir}/{dir_to_save}")
        # Creation du fichier csv
        df_out = pd.DataFrame(columns=['Name', 'ClassId'])
        df_out.to_csv(f"./{dataset_dir}/{dir_to_save}/fichier.csv", index=False)

        # Liste toute les photo du dossier
    fichiers = [f for f in os.listdir(f"./{dataset_dir}") if os.path.isfile(os.path.join(f"./{dataset_dir}/", f))]

    for i, name_image in enumerate(fichiers):
        fig, ax1 = plt.subplots(figsize=(5, 5))
        # Modification du nom de l'image
        new_name = f"image_{str(datetime.now().today()).replace(':','.')}.png"
        os.rename(f"./{dataset_dir}/{name_image}", f"./{dataset_dir}/{new_name}")

        # Chemin de l'image
        ad_image =  f"./{dataset_dir}/{new_name}"
        # Affichage de l'image
        image = io.imread(ad_image)
        ax1.imshow(image)
        plt.show()
        valeur = input('Entrer une valeur : \n')

        # Ouverture du fichier CSV
        df = pd.read_csv(f"./{dataset_dir}/{dir_to_save}/fichier.csv")

        # Mise a jour du fichier CSV
        df = df.append({'Name': new_name, 'ClassId': valeur}, ignore_index=True)

        df.to_csv(f"./{dataset_dir}/{dir_to_save}/fichier.csv",index=False)
        # Je deplace la photo dans le nouveau repertoir
        shutil.move(ad_image, f"./{dataset_dir}/{dir_to_save}")


if __name__ == "__main__":
    # create_fichier_csv_dataset('Test2')
    create_uniformized_dataset(dataset_dir=['Dataset 7'], csv_name='fichier.csv', traitement=['RGB', 'L'], size=[(150, 150)], r=1)
