#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 01:28:11 2020

@author: steve
"""
from knnBagge import knn,params,echantillons,random,chargement



def division_echantillon(sample,prop):
    '''
    Cette fonction prend en entrée un échantillon et, par tirages aléatoires avec remise, en retourne deux  :
        un échantillon d'apprentissage et un échantillon de test , et ceci dans la proportion prop

    Parameters
    ----------
    sample : échantillon de départ.
    prop : la proportion de repartion.

    Returns
    -------
    train : échantillon d'apprentissage.
    test : échantillon de test.

    '''
    train,test = list(), list()
    n = len(sample)
    indexes = [i for i in range(n)]
    indexes_test = random.sample(indexes, int(n*prop))
    indexes_train = indexes
    for i in indexes_test:
        indexes_train.remove(i)
    
    for i in indexes_test:
        test.append(sample[i])
    for i in indexes_train :
        train.append(sample[i])
        
    return train,test


def cross_validation():
    '''

    Parameters
    ----------
    echantillon : de type liste, contient des objets de type(input, output)

    Returns
    -------
    L'erreur associée à la validation croisée
    
    Principe :
    ---------
    Faire K fois l'opération suivante :
        -simuler un sous-échantillon de l'échantillon mère et en construire un prédicteur
        - le reste de l'échantillon est utilisé pour le test dont on stocke l'erreur de prédiction 
        dans une structure
    Moyenniser les erreurs .

    '''
    donnees = chargement()
    proportion = 0.2
    N = 10
    '''
        La validation est faite de façon parallèle. Étant donné qu'en général le nombre d'itérations
        varie entre 5 et 10, nous mettons en place 10 processeurs pour exécuter simultanément les opérations
        de lavalidation croisée.
    '''
    knn.donnees = list()
    print(end = '\n')
    
    confirm = 'ok'
    try:
        K = int(input("Enter le nombre K d'itérations de la validation croisée :  "))
        print(end = '\n')
    except Exception:
        print("Veuillez entrer un nombre entier s'il vous plaît ...")
        print(end = '\n')
        confirm = None
    while confirm == None:
        confirm = 'ok'
        try:
            print(end = '\n')
            K = int(input("Enter le nombre K d'itérations de la validation croisée :  "))
            print(end = '\n')
        except Exception:
            print("Veuillez entrer un nombre entier s'il vous plaît ...")
            print(end = '\n')
            confirm = None
            
    data = donnees.donnees_classification
        
    for k in range(K) :
        entrainement, test = division_echantillon(data,proportion)
        knn.donnees.append(echantillons(entrainement,test))
        
    params()
    
    if knn.regOUclass in 'Rr' :
        del knn.donnees[:]
        data = donnees.donnees_regression
        for k in range(K) :
            entrainement, test = division_echantillon(data,proportion)
            knn.donnees.append(echantillons(entrainement,test))

    reponses = list()
    for i in range(N):
        reponses.append(knn(i))
        
    for i in range(N):
        reponses[i].start()
        reponses[i].join()
        
if __name__ == '__main__' :

    cross_validation()