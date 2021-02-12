#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:25:59 2021

@author: steve
"""
from knnBagge import knn,params,echantillons,chargement
from knn_validation_croisee import division_echantillon
import matplotlib.pyplot as plt

def prediction_echantillon_test():

    N = 1
    '''
        Cette parallélisation est prévue pour la validation croisée. N processeurs sont disposés pour ça.
        Dans le cas de la prédiction d'un échantillon test simple, un seul processeur suffit. D'où N = 1
    '''

    knn.donnees = list()
    donnees = chargement()
    proportion = 0.5
    data = donnees.donnees_classification
    entrainement, test = division_echantillon(data,proportion)
    knn.donnees.append(echantillons(entrainement,test))
    
    params()
    
    if knn.regOUclass in 'Rr' :
        del knn.donnees[:]
        data = donnees.donnees_regression
        entrainement, test = division_echantillon(data,proportion)
        knn.donnees.append(echantillons(entrainement,test))
        
    print(end = '\n')
    print("Voici la prédictions de l'échantillon test : ", end = '\n')
    print(end = '\n')
    
    reponsesPredites = list()
    for i in range(N):
        reponsesPredites.append(knn(i))

    for i in range(N) :
        reponsesPredites[i].start()
        reponsesPredites[i].join()
    print(end = '\n')
    if knn.regOUclass in 'Rr' :
        i = 0
        for err in knn.for_plot:
            plt.plot([i, i],[0, err], color = 'r')
            plt.scatter(i,err, color = 'b')
            i += 1
        plt.title('Les erreurs commises sur les prédictions des largeurs des pétales')
        del(i)
    else:
        i = 0
        for err in knn.for_plot:
            if err == 0:
                plt.scatter(i,err, color = 'b')
            else:
                plt.scatter(i,err, color = 'r')
            i += 1
        plt.title('Les erreurs commises sur les prédictions')
        del(i)
if __name__ == '__main__':
    prediction_echantillon_test()
