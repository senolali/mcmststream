# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:28:32 2024

@author: Poyraz
"""
import matplotlib.pyplot as plt
import numpy as np
import kdtree
from scipy.spatial import KDTree
import scipy
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')
get_ipython().magic('clear all -sf')



# Defining class KDMCStream
class MCMSTStream:
    N=int()
    W=int()
    r=float()    
    d=int()
    MC_Num=int
    MacroC_Num=int
    colors=np.empty((0,4),int)
    def __init__(self,X,N,W,r,n_micro,d,plotFigure):
        #algorithm parameters###########################################
        self.X=X
        self.N = N #minimum number of data to define a MC 
        self.W = W #sliding window size
        self.r = r # radius of MC
        self.n_micro=n_micro # minimum number of MC to define a Macro Cluster
        self.plotFigure=plotFigure
        ##################################################################
        self.d=d
        self.MC_Num=0
        self.MacroC_Num=0
        self.buffered_data=np.empty((0,d+3),float) #[index | MC No | isActive | features={d1,d2,d3...}]
        self.MCs=np.empty((0,d+3),float) #[MC No | #of data it has | centerCoordinates={d1,d2,d3,...}]
        self.MacroClusters=np.empty((0,4)) #[MacroClusterNo | #of data it has | isActive ]
        self.deleted_data=np.empty((0,d+3),float) #[index | features={d1,d2,d3...} | predictedclusterLabel]
        for i in range(len(self.X)):
            self.AddNode(self.X[i,:])
            self.DefineMC()
            self.AddtoMC()
            self.DefineMacroC()
            self.AddMCtoMacroC()
            self.UpdateInfo()
            self.UpdateMacroCs()
            self.KillMCs()
            self.KillMacroC()
            if self.plotFigure==1 and self.d==2:
                self.plotGraph("KDMCStream") 
    		
    def AddNode(self,X): # add new data to buffered_data, and delete old ones 
       if(self.buffered_data.shape[0]==0):
           self.buffered_data=np.vstack((self.buffered_data,np.hstack((np.array([1,0,0]),X))))
       else:
           self.buffered_data=np.vstack((self.buffered_data,np.hstack((np.array([self.buffered_data[self.buffered_data.shape[0]-1,0]+1,0,0]),X))))
       if(self.buffered_data.shape[0]>self.W):
           self.deleted_data=np.vstack((self.deleted_data,np.split(self.buffered_data, [self.buffered_data.shape[0]-self.W])[0]))
           self.buffered_data=np.split(self.buffered_data, [self.buffered_data.shape[0]-self.W])[1]
           
    def DefineMC(self):
        X=self.buffered_data[self.buffered_data[:,1]==0,:]#data that do not belong to any cluster
        if(X.shape[0]>=self.N): # if # of data that do not belong to any cluster greater than N
            tree=kdtree.create(X[:,3:].tolist()) #construct kdtree
            for i in range(X.shape[0]): # for each data of tree do reangeserach
                points=tree.search_nn_dist(X[i,3:], self.r) #rangesearch
                if(len(points)>=self.N):  
                    center=np.mean(np.array(points),axis=0) #calculate the center of candidate MC
                    kdtree2 = KDTree(self.MCs[self.MCs[:,2]!=0,3:])
                    d2, ind2 = kdtree2.query(center)
                    if(d2>self.r*0.75):                       
                        self.MC_Num=self.MC_Num+1
                        # print("MC number ",self.MC_Num," is defined")
                        self.MCs=np.vstack((self.MCs,np.hstack((np.array([self.MC_Num,len(points),0]),center)))) # define new MC
                        for j in range(len(points)):                  
                            self.buffered_data[np.where((self.buffered_data[:,3:] == points[j]).all(axis=1))[0],1]= self.MC_Num #assign data to new MC     
                        return
    def AddtoMC(self):
        if(self.MCs.shape[0]>1):
            for i in range(self.buffered_data.shape[0]):
                if(self.buffered_data[i,1]==0):
                    kdtree = KDTree(self.MCs[:,3:])
                    d, ind = kdtree.query(self.buffered_data[i,3:])
                    if(d<=self.r):
                        self.buffered_data[i,1]=self.MCs[ind,0] 
    def UpdateInfo(self):
        for i in range(self.MCs.shape[0]):
            self.MCs[i,1]=self.buffered_data[self.buffered_data[:,1]==self.MCs[i,0],:].shape[0]
            if(self.MCs[i,1]>0):
                self.MCs[i,3:]=np.mean(self.buffered_data[self.buffered_data[:,1]==self.MCs[i,0],3:],axis=0) #calculate the center of MC            
            if(self.MCs[i,2]>0):
                self.buffered_data[self.buffered_data[:,1]==self.MCs[i,0],2]=self.MCs[i,2]
    def KillMCs(self):
        for i in range(self.MCs.shape[0]):
            if (int(self.MCs[i,1])==0):                
                # print("MC number ",int(self.MCs[i,0])," is killed")
                if(int(self.MCs[i,2])!=0):
                    MacroCluster=self.MCs[i,2]
                    self.buffered_data[self.buffered_data[:,1]==self.MCs[i,0],1]=0
                    self.MCs=np.delete(self.MCs,i,axis=0)
                    self.UpdateMacroC(MacroCluster)
                else:
                    self.buffered_data[self.buffered_data[:,1]==self.MCs[i,0],1]=0
                    self.MCs=np.delete(self.MCs,i,axis=0)
                
                return
    def UpdateMacroCs(self):
        for i in range(self.MacroClusters.shape[0]):
            self.UpdateMacroC(self.MacroClusters[i,0])
            self.MacroClusters[i,1]=len(np.unique(self.MacroClusters[i,2]))
    def UpdateMacroC(self,macroCluster):
            if(macroCluster!=0):
                P=self.MCs[self.MCs[:,2]==macroCluster,]
                self.MCs[self.MCs[:,2]==macroCluster,2]=0  
                X=squareform(pdist(P[:,3:]))
                edge_lists = self.minimum_spanning_tree(X)
                edge_list=np.empty((0,2),int)
                for index in edge_lists:
                    ii,jj=index
                    edge_list=np.vstack((edge_list,(int(P[ii,0]),int(P[jj,0]))))
                self.MacroClusters[self.MacroClusters[:,0]==macroCluster,2]=[edge_list]
                for j in np.unique(edge_list):
                    self.MCs[self.MCs[:,0]==j,2]=macroCluster
                return 
    def minimum_spanning_tree(self,X, copy_X=True):
        """X are edge weights of fully connected graph"""
        if copy_X:
            X = X.copy()
        if X.shape[0] != X.shape[1]:
            raise ValueError("X needs to be square matrix of edge weights")
        n_vertices = X.shape[0]
        spanning_edges = []  
        # initialize with node 0:                                                                                         
        visited_vertices = [0]                                                                                            
        num_visited = 1
        # exclude self connections:
        diag_indices = np.arange(n_vertices)
        # print(diag_indices)
        X[diag_indices, diag_indices] = np.inf
        X[X>2*self.r]=np.inf
        # print(X,1.5*self.r)   
        while num_visited != n_vertices:
            new_edge = np.argmin(X[visited_vertices], axis=None)
            # print(new_edge)
            # 2d encoding of new_edge from flat, get correct indices                                                      
            new_edge = divmod(new_edge, n_vertices)
            # print(visited_vertices[new_edge[0]], new_edge[1])
            new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                       
            # add edge to tree
            if (new_edge[0] != new_edge[1]):
                spanning_edges.append(new_edge)
                visited_vertices.append(new_edge[1])
            # remove all edges inside current tree
            X[visited_vertices, new_edge[1]] = np.inf
            X[new_edge[1], visited_vertices] = np.inf                                                                
            num_visited += 1
        # print("edges=",spanning_edges)
        if(len(spanning_edges)==0):
            return spanning_edges
        else:
            return np.vstack(spanning_edges)
    def DefineMacroC(self):
        if (len(self.MCs[self.MCs[:,2]==0])>=self.n_micro):
            P=self.MCs[self.MCs[:,2]==0,]
            X=squareform(pdist(P[:,3:]))
            edge_lists = self.minimum_spanning_tree(X)
            # print(edge_lists)
            edge_list=np.empty((0,2),int)
            for index in edge_lists:
                i,j=index
                edge_list=np.vstack((edge_list,(int(P[i,0]),int(P[j,0]))))
            summ=0
            edges=np.unique(edge_list)
            for e in edges:
                summ=summ+self.MCs[self.MCs[:,0]==e,1]
            if(summ>=self.n_micro*self.N or len(np.unique(edge_list))>=self.n_micro):
                self.MacroC_Num=self.MacroC_Num+1
                self.colors = np.array([plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, self.MacroC_Num+1)]) 
                # print(self.colors)
                # print(self.colors[-2,:])
                self.MacroClusters=np.vstack((self.MacroClusters,np.array([self.MacroC_Num,len(np.unique(edge_list)),edge_list,self.colors[-2,:]])))
                print("----------Macro Cluster #",self.MacroC_Num," is defined----------")
                for i in np.unique(edge_list):
                    self.MCs[self.MCs[:,0]==i,2]=self.MacroC_Num    
                return 
    def AddMCtoMacroC(self):#Assign any MC that enoughly close to MC that was assigned MacroC
        if(self.MacroClusters.shape[0]!=0):
            for i in range(self.MCs.shape[0]):
                if(self.MCs[i,2]==0 and self.MCs[i,1]>=self.N):  
                    A=self.MCs[self.MCs[:,2]!=0,:]
                    if(A.shape[0]>0):
                        kdtree = KDTree(A[:,3:])
                        d, ind = kdtree.query(self.MCs[i,3:])
                        if(d<=2*self.r):
                            self.MCs[i,2]=A[ind,2]
                            MacroC=int(A[ind,2])
                            # print("MC #",int(self.MCs[i,0])," is assigned to MacroC #",MacroC," over MC #",int(A[ind,0]))
                            # self.MacroClusters[self.MacroClusters[:,0]==MacroC,2]=[np.vstack((    self.MacroClusters[self.MacroClusters[:,0]==MacroC,2][0],[int(self.MCs[i,0]),int(A[ind,0])]   ))]
                            self.UpdateMacroC(self.MCs[i,2])  
                            return
    def KillMacroC(self):
        for i in range(self.MacroClusters.shape[0]):
            edge_list=self.MacroClusters[i,2]
            summ=0
            edges=np.unique(edge_list)
            for e in edges:
                summ=summ+self.MCs[self.MCs[:,0]==e,1]
            if(summ<self.n_micro*self.N and len(edges)<self.n_micro):
            # if(self.MacroClusters[i,1]<self.n_micro):
                # for j in range(len(np.unique(self.MacroClusters[i,2]))):
                    # print("Before ",self.MCs[self.MCs[:,0]==j,:])
                self.MCs[self.MCs[:,2]==self.MacroClusters[i,0],2]=0;
                    # print("After ",self.MCs[self.MCs[:,0]==j,:])
                print("----------Macro Cluster #",self.MacroClusters[i,0]," is killed----------") 
                self.MacroClusters = np.delete(self.MacroClusters, i, axis=0)                 
                return;
    def plotGraph(self,title,dpi=70):
        ax = plt.gca()
        ax.cla() # clear things for fresh plot 
        plt.rcParams['figure.dpi'] = dpi
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)
        plt.rcParams["figure.figsize"] = (4,4)
        for i in range(len(self.buffered_data)):
            plt.plot(self.buffered_data[i, 3], self.buffered_data[i, 4],'bo',markeredgecolor='k',alpha=.15, markersize=5)
        
        for i in range(len(self.MCs)):
            if(self.MCs[i, 2]==0):
                col=(0,0,1,1)
            else:
                # print(self.MCs[i,2])
                # print(self.MacroClusters[self.MacroClusters[:,0]==self.MCs[i,2],3])
                col=self.MacroClusters[self.MacroClusters[:,0]==self.MCs[i,2],3][0].tolist()

            plt.plot(self.MCs[i, 3], self.MCs[i, 4],'rd',markeredgecolor='k', markersize=5)
            circle1=plt.Circle((self.MCs[i,3],self.MCs[i,4]),self.r,color=col, clip_on=False,fill=False)
            # plt.text(self.MCs[i, 3], self.MCs[i, 4],int(self.MCs[i, 0]),horizontalalignment='right')
            ax.add_patch(circle1)         
        for edge in self.MacroClusters[:,2]:
            for e in edge:
                i, j = e
                plt.plot([self.MCs[self.MCs[:,0]==i, 3], self.MCs[self.MCs[:,0]==j, 3]], [self.MCs[self.MCs[:,0]==i, 4], self.MCs[self.MCs[:,0]==j, 4]], c=self.MacroClusters[self.MacroClusters[:,0]==self.MCs[self.MCs[:,0]==i,2],3][0].tolist(),markersize=500)     
        plt.title(title) 
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.grid()
        plt.xlim([-.25,1.25]) 
        plt.ylim([-.25,1.25])            
        plt.show()
    def purity_score(self,y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


    
    