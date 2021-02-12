#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:54:06 2021

@author: steve
"""
import threading
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,plot_confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv("donnees.csv")


class load(threading.Thread):
    colonnes = None
    donnees = list()
    verrou = threading.Lock()
    
    def __init__(self, num):
        self.num = num
        threading.Thread.__init__(self)
        
    def run(self):
        while len(load.colonnes) > 0:
            load.verrou.acquire()
            if len(load.colonnes)>0:
                col = load.colonnes.pop(0)
            load.verrou.release()
            load.donnees.append(list(data[col][1:]))
            
def main(N):
    load.colonnes = list(data.columns)
    #load.colonnes.remove('Sample_geo_accession')
    mes_taches = list()
    
    for i in range(N):
        mes_taches.append(load(i))
        
    for i in range(N):
        mes_taches[i].start()
        mes_taches[i].join()
        
    d = pd.DataFrame(np.array(load.donnees), columns = data['Sample_geo_accession'][1:])
    d.drop(index = [0],columns=['tissue'], inplace = True)
    
    y  =  d['treatment_response']
    d = pd.get_dummies(d,columns=['ethnicity','PR_status: ','MAQC_Distribution_Status',
                              'treatment code','ER_status: ','histology','her2_status'])
    del d['treatment_response']
    '''del d['ethnicity']
    del d['PR_status: ']
    del d['ER_status: ']'''
    del d['ID_REF']
    '''del d['her2_status']
    del d['histology']
    del d['treatment code']
    del d['MAQC_Distribution_Status']'''
    d.drop(index = [143,162,163,165,166,167,168,169,170,171,172,174,175], inplace = True)
    y.drop(index = [143,162,163,165,166,167,168,169,170,171,172,174,175], inplace = True)
    d.index = np.arange(1,266)
    y.index = np.arange(1,266)
    y.apply(lambda x: 0 if x == 'RD' else 1)
    #del d['treatment_response']
    return d,y


X,y = main(100)
    
    
    
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

'''
Je construis plusieurs modèles un noyau linéaire standard,un noyau linéaire pénalisé,  
un noyau gaussien et un noyaupolynomial de degré 2, puis je compare leurs précisions prédictives.
'''
def sVm():
    my_models = [svm.SVC(kernel ='linear', C=1,class_weight = 'balanced'),
              svm.LinearSVC(C = 1, max_iter=10000,class_weight = 'balanced'),
              svm.SVC(kernel='rbf', gamma=0.7, C=1,class_weight = 'balanced'),
              svm.SVC(kernel='poly', degree=2, gamma='auto', C=1,class_weight = 'balanced')
              ]
    
    my_models = [my_classifier.fit(X_train,y_train) for my_classifier in my_models]
    names = 'Noyau linéaire','Noyau linéaire pénalisé','Noyau gaussien','Noyau polynomial'
    k = 0
    for classifieur in my_models:
        mes_predictions = classifieur.predict(X_test)
        print("La précision de prédiction est de {:.2%}".format(accuracy_score(y_test,mes_predictions)))
        plot_confusion_matrix(classifieur,X_test,y_test)
        plt.title('{} : taux de bonnes prédictions = {:.2%}'.format(names[k],accuracy_score(y_test,mes_predictions)))
        k+= 1


def random_forest():
    my_models2 = [RandomForestClassifier(max_depth=10,max_features = 'auto',class_weight = 'balanced'),
                  RandomForestClassifier(max_depth=10,max_features = 'sqrt',class_weight = 'balanced'),
                  RandomForestClassifier(max_depth=10,max_features = 'log2',class_weight = 'balanced')
        ]
    
    
    
    names2 = 'arbre1','arbre2','arbre3'
    my_models2 = [my_classifier.fit(X_train,y_train) for my_classifier in my_models2]
    k = 0
    for classifieur in my_models2:
        mes_predictions = classifieur.predict(X_test)
        print("La précision de prédiction est de {:.2%}".format(accuracy_score(y_test,mes_predictions)))
        plot_confusion_matrix(classifieur,X_test,y_test)
        plt.title('{} : taux de bonnes prédictions = {:.2%}'.format(names2[k],accuracy_score(y_test,mes_predictions)))
        k+= 1
