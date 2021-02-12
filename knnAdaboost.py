#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 22:50:35 2021

@author: steve
"""
from knnBagge import kppv,np,chargement,namedtuple,threading
from knn_validation_croisee import division_echantillon
import matplotlib.pyplot as plt
donnees = chargement()
f_et_alpha = namedtuple('f_et_alpha',['f_chap','alpha'])


        
data = donnees.donnees_classification
proportion = 0.5

class predictors(threading.Thread):
    ehantillon_entrainement, ehantillon_test = division_echantillon(data, proportion)
    p = list()
    def __init__(self,k):
        self.k = k
        threading.Thread.__init__(self)
        
    def run(self):
        pred_courant = list()
        for vecteur in predictors.ehantillon_test:
            pred_courant.append(kppv(predictors.ehantillon_entrainement, vecteur.x, self.k))
        predictors.p.append(pred_courant)
            
def chargement_des_predicteurs():
    '''
    Je mets en place 20 prédicteurs : 1-ppv, 2-ppv, ..., 20-ppv
    Returns
    -------
    Les 20 prédicteurs fabriqués et qui seront utilisés dans le boosting.

    '''
    predicteurs = list()
    for k in range(1,21):
        predicteurs.append(predictors(k))
    for tache in predicteurs:
        tache.start()
        tache.join()
    return predictors.p,predictors.ehantillon_test
    
    
def adaboost(M):
    p,ehantillon_test = chargement_des_predicteurs()
    n = len(p[0])
    print(end = '\n')
    print(end = '\n')
    f_m_chapeaux = list()
    w = pow(n,-1)*np.ones(n)
    y = np.array([i.y for i in ehantillon_test ])
    
    for m in range(M):
        '''
            Prédicteur optimal à l'étape m
            y est la vraie réponse qui se trouve dans la base de données
        '''
        f_chapeau_m = np.array([i.quali for i in p[0]])
        for j in range(1,20):
            f_chapeau_m_courant = np.array([i.quali for i in p[j]])
            if sum(w*(f_chapeau_m != y)) > sum(w*(f_chapeau_m_courant != y)):
                f_chapeau_m = f_chapeau_m_courant
                del(f_chapeau_m_courant)
                
                
        e_m = sum(w*(f_chapeau_m != y))/sum(w)
        if e_m != 0:
            alpha_m = max(pow(2,-1)*np.log((1-e_m)/e_m), 0)
        else:
            alpha_m = 0
            
        w = w*np.exp(alpha_m* (f_chapeau_m != y))
        f_m_chapeaux.append(f_et_alpha(f_chapeau_m,alpha_m))
    del(p)
    '''
        f_m_chapeaux contient les 20 prédicteurs f_chapeau1, f_chapeau2, ..., f_chapeau20
    '''
            
    return f_m_chapeaux, n,ehantillon_test


def f_optimal(M):
    '''
    
    Returns
    -------
    f_chapeau : Le prédicteur construit par le boosting.

    '''
    print(end = '\n')
    predicteurs_optimaux, n,ehantillon_test = adaboost(M)
    f_chapeau = list()
    M = list()
    for i in range(n):
        mod = list()
        for f_and_alpha in predicteurs_optimaux:
            mod.append((f_and_alpha.f_chap[i],f_and_alpha.alpha))
        
        M.append(mod)
        del(mod)
        
    for mod in M:
        mod0 = [i[1] for i in mod if i[0]==0]
        mod1 = [i[1] for i in mod if i[0]==1]
        mod2 = [i[1] for i in mod if i[0]==2]
        
        f_chapeau.append([mod0, mod1,mod2].index(max([mod0, mod1,mod2])))
        del(mod0,mod1,mod2)
    del(M)
    erreur_de_prediction = 0
    j = 0
    for i in range(n):
         print("La réponse prédite pour {} est {} contre {}".format( tuple(ehantillon_test[i].x), f_chapeau[i], ehantillon_test[i].y))
         erreur_de_prediction += int(ehantillon_test[i].y != f_chapeau[i])
         err = int(ehantillon_test[i].y != f_chapeau[i])
         if err == 0:
                plt.scatter(j,err, color = 'b')
         else:
            plt.scatter(j,err, color = 'r')
         j += 1
    plt.title('Les erreurs commises sur les prédictions')
    del(j)
    print(end = '\n') 
    print("L'erreur de prédiction est de {:.2%} ".format(erreur_de_prediction/n))
    print(end = '\n')

if __name__ == '__main__':
    f_optimal(int(input("Nombre M d'itérations du boosting :  ")))
     

    
    
