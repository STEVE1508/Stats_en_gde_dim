#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 06:40:48 2021

@author: steve
"""

import numpy as np
import threading,random
from collections import Counter, namedtuple
XY = namedtuple('vect_and_rep',['x','y'])
ind = namedtuple('ind', ['x', 'y', 'dist'])
res = namedtuple('response', ['quanti','quali'])
echantillons = namedtuple('echantillons',['echantillon_entrainement','echantillon_test'])
donnees = namedtuple('donnees',['donnees_classification','donnees_regression'])
from sklearn.datasets import load_iris


'''
    Je crée trois types d'objets : ind , res et sous_ech
    
    ind a 3 instances : 
        - x : le vecteur
        - y : la réponse associée
        - dist : la distance du vecteur avec celui dont on veut prédire la réponse
        
    res a 2 instances:
        - quanti : si la réponse est quantitative
        - quali : si la réponse est qualitative
        
    sous_ech a 2 instances :
        - bag : le sous-échantillon du bagging
        - num : le numéro du sous-échantillon
'''

def chargement():
    '''
    Returns
    -------
    Deux bases de données iris:
        1. une qu'on utilisera pour faire la classification des fleurs dans ['setosa', 'versicolor', 'virginica']
        2. une autre base de données contenant les largeurs et longueurs des sépales et pétales. Celle base de données
           sera utilisée pour faire la régression et prédire la largeur des pétales. 

    '''
    d = load_iris()
    data_classification = list()
    data_regression = list()
    for i in range(len(d['target'])):
        x = d['data'][i]
        y = d['target'][i]
        data_classification.append(XY(x, y))
        data_regression.append(XY(x[:-1], x[-1]))
        
    return donnees(data_classification,data_regression)


def bagging(sample,B,m):
    
    bags = list()
    '''
    

    Parameters
    ----------
    sample : échantillon mère
    B : Nombre de tirages avec remise pour générer les sous-échantillons .
    m : cardinal des sous-échantillons .

    Returns
    -------
    Les sous-échantillons formés

    '''
    for i in range(B):
        b = random.sample(sample,m)
        bags.append(b)
        
    return bags

def kppv(bag, vector,k):
    '''
    

    Parameters
    ----------
    bag : un échantillon.
    vector : vecteur dont on veut prédire la réponse par k plus proches voisins.
    k : Nombre de voisins à considérer.

    Returns
    -------
    pred : réponse prédite pour le vecteur passé en argument.

    '''
    d = list()
    for v in bag:
        vf = np.array(v[0])- np.array(vector)
        dist = np.linalg.norm(vf)
        
        d.append(ind(v[0],v[1], dist))
        
    d = sorted(d, key = lambda c : c.dist)[:k]
    '''
        On sélectionne les k vecteurs qui ont les plus petites distances avec le vecteur
        dont on veut prédire la réponse.
    '''
    
    '''
    print(f"Voici les {min(self.k,len(d))} voisins de {knn.vector} dans le bagging {Bag.num}: ", end = '\n')
    for i in d : print(i)
    print(end = '\n')
    '''
    
    pred = res(np.mean([i.y for i in d]),Counter([i.y for i in d]).most_common(1)[0][0])
    return pred


        
class knn(threading.Thread):
    
    verrou = threading.Lock()
    donnees = None
    
    y = None
    regOUclass = None
    B,k,card = None,None,None
    for_plot = list()
    
    def __init__(self,num):
        self.num = num
        threading.Thread.__init__(self)
        
    def run(self):
        while len(knn.donnees)>0:
            knn.verrou.acquire()
            if len(knn.donnees)>0:
                sample = knn.donnees.pop(0)
            knn.verrou.release()
            erreur_de_prediction = 0
            bags = bagging(sample.echantillon_entrainement, knn.B, knn.card)
            reponses = list()
            n = len(sample.echantillon_test)
            for vect in sample.echantillon_test:
                for bag in bags :
                    reponse = kppv(bag, vect.x, knn.k)
                    reponses.append(reponse)
                if knn.regOUclass in 'Rr' :
                    knn.y = np.mean([val.quanti for val in reponses])
                    erreur_de_prediction += pow(knn.y-vect.y,2)
                    knn.for_plot.append(pow(knn.y-vect.y,2))
                else:
                    knn.y = Counter([val.quali for val in reponses]).most_common(1)[0][0]
                    erreur_de_prediction += int(knn.y != vect.y)
                    knn.for_plot.append(int(knn.y != vect.y))
                    
                print("La réponse prédite pour {} est {} contre {}".format( tuple(vect.x), knn.y, vect.y))
            print(end = '\n')
            print("Cette itération a une erreur de prédiction de {:.2%} ".format(erreur_de_prediction/n))
            print(end = '\n')

def params():
    print(end = '\n')
    knn.regOUclass = input("Vous faites de la régression ou de la classification (r/c) ?  ")
    print(end = '\n')
    while knn.regOUclass not in ['R','r','C','c']:
        print(end = '\n')
        print("Caractère non reconnu, entrer r ou c selon que vous faites de la régression ou de la classification", end = '\n')
        knn.regOUclass = input("Vous faites de la régression ou de la classification (r/c) ?  ")
        print(end = '\n')
        
    confirm = 'ok'
    try:
        knn.B = int(input("Combien de sous-échantillons voulez-vous dans votre bagging ?  "))
        print(end = '\n')
    except Exception:
        print("Veuillez entrer un nombre entier s'il vous plaît ...")
        print(end = '\n')
        confirm = None
    while confirm == None:
        confirm = 'ok'
        try:
            print(end = '\n')
            knn.B = int(input("Combien de sous-échantillons voulez-vous dans votre bagging ?  "))
            print(end = '\n')
        except Exception:
            print("Veuillez entrer un nombre entier s'il vous plaît ...")
            print(end = '\n')
            confirm = None
        
    knn.card = int(input(f"Quelle est la taille des sous-échantillons du bagging ( au plus {len(knn.donnees[0].echantillon_entrainement)}) ? "))
    while knn.card > len(knn.donnees[0].echantillon_entrainement):
        print(end = '\n')
        print("La taille des sous-échantillons ne peut pas être plus grande que l'échantillon mère...")
        knn.card = int(input(f"Quelle est la taille des sous-échantillons du bagging ( au plus {len(knn.donnees[0].echantillon_entrainement)}) ? "))
        print(end = '\n')
    knn.k = int(input(f"Combien de voisins considérez-vous dans votre algorithme des k-ppv( au plus {knn.card}) ? "))
    print(end = '\n')
    while knn.k > knn.card:
        print(end = '\n')
        print("On ne peut pas avoir plus de voisins que le cardinal du bagging...")
        knn.k = int(input(f"Combien de voisins considérez-vous dans votre algorithme des k-ppv( au plus {knn.card}) ? "))
        print(end = '\n')
