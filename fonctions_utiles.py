import random, os, sys
from datetime import datetime

def combinaison(**kwargs) -> list:
    '''
    retourne un dictionnaire du dictionnaire passe en entree de toutes les combinaisons uniques
    sur la base de dictionnaire de liste
    par exemple


    liste_1 = [1,2]
    liste_2 = ['a','b']
    liste_3 = ['A','B']

    => [[1, 'b', 'B'], [1, 'b', 'A'], [2, 'b', 'B'], [1, 'a', 'A'], [1, 'a', 'B'], [2, 'a', 'A'], [2, 'b', 'A'], [2, 'a', 'B']]

    Args:
        _list: dictionnaire de combinaison

    Returns:

    '''
    n_fois = 0
    out = []
    list_all = [] # liste de tuple des tout les elements ainsi que de leurs origines
    # Creation d'une liste de tout les elements identifies
    for key, value_list in kwargs.items():
        [list_all.append((value_list[i], key)) for i in range(len(value_list))] # Lecture sur une ligne de tout les dictionnaire
        n_fois += len(value_list)
    n_fois = n_fois ** 3

    for i in range(n_fois): # Creation de combinaisons aleatoire,
    # Mais suffisament pour etre sur de creer toutes les combinaisons possibles
        _origine = dict()
        for clef in kwargs.keys():
            _origine[clef] = 'NaN' # Initialisation des valeur

        while 'NaN' in _origine.values():
            i_rand = random.randint(0, len(list_all) - 1)
            value, clef = list_all[i_rand]
            if _origine[clef] == 'NaN':
                _origine[clef] = value

        if _origine not in out:
            out.append(_origine)
    return out


class Timing:
    '''
    Decorateur du temps d'excution d'une photo
    '''

    def __init__(self, f):
        self.f = f
        self.x1 = datetime.now()

    def __call__(self, *t, **d):
        out = self.f(*t, **d)
        print(f"Temps d'execution de la fonction > {self.f.__name__} < :{(datetime.now() - self.x1)} secondes.")
        return out

class DimError(Exception):
    def __init__(self, x, y, value1, value2):
        print(f'Les datas {x} et {y} ne comportent pas le meme nombre de lignes : {value1} different de {value2}')


#
# print(address_lin_to_win_or_lin('/home/linux'))
