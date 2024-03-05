# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:28:32 2024

@author: Poyraz
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score
from MCMSTStream import MCMSTStream # import class MCMSTStream from the same directory
import time
from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')
get_ipython().magic('clear all -sf')




# Test on ExclaStar dataset
dataset = np.loadtxt("Datasets/ExclaStar.txt", dtype=float,delimiter=',')
X=dataset[:,1:3]
labels_true=dataset[:,3]
dataset_name="2_ExclaStar"
# Obtained best parameters for ExclaStar dataset
W=235
N=2
r=0.0330
n_micro=5



####MinMaxNormalization#######################################################
scaler = MinMaxScaler()
scaler.fit(X)
MinMaxScaler()
X=scaler.transform(X[:,:])


plotFigure=0 # 1 for plotting clusters 
start = time.time()    
kds=MCMSTStream(X,N,W,r,n_micro,X.shape[1],plotFigure)
end = time.time()
print("Elapsed Time=",end - start)


labels=np.hstack((kds.deleted_data[:,2],kds.buffered_data[:,2]))
ARI=adjusted_rand_score(labels_true.reshape(-1), labels)
Purity=kds.purity_score(labels_true.reshape(-1), labels)
 

print("\n\n##### The Best Results #############3")
print("Purity=",Purity)
print("ARI=",ARI)
kds.plotGraph(str("MCMSTStream =>ARI="+str(ARI)))
    
    