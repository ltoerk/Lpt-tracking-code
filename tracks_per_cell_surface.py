# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:08:01 2020

@author: lisa
"""
import numpy as np

## import traj data and cell surface data
Cells_treated_1_s=np.loadtxt(".txt") #this is mask form fiji
Cells_treated_1_t=np.load(".npy") # this is corresponding trajectory file from filter_trajectories_ic_cells

Cells_treated_2_s=np.loadtxt(".txt") #this is mask form fiji
Cells_treated_2_t=np.load(".npy") # this is corresponding trajectory file from filter_trajectories_ic_cells

Cells_wt_1_s=np.loadtxt(".txt") #this is mask form fiji
Cells_wt_1_t=np.load(".npy") # this is corresponding trajectory file from filter_trajectories_ic_cells

Cells_wt_2_s=np.loadtxt(".txt") #this is mask form fiji
Cells_wt_2_t=np.load(".npy") # this is corresponding trajectory file from filter_trajectories_ic_cells


L_treated_surface=[Cells_treated_1_s,Cells_treated_2_s]
L_treated_tracks=[Cells_treated_1_t,Cells_treated_2_t]

L_wt_surface=[Cells_wt_1_s,Cells_wt_2_s]
L_wt_tracks=[Cells_wt_1_t,Cells_wt_2_t]

def make_bool(L):
    L_bool=[]
    for i in L:
        L_bool.append(i!=0)
    return L_bool
    
def summ(L):
    L_sum=[]
    for i in L:
        L_sum.append(np.sum(i))
    return L_sum
    
def divide(L_surface,L_tracks):
    L_result=[]
    for i,j in zip(L_surface,L_tracks):
        L_result.append(len(j)/float(i))
    return L_result
    
def plot_tracks(L_tracks):
    for i in L_tracks:
        plt.plot(i.T[0]*6.25,i.T[1]*6.25)
def plot_phase(L_phase):
    pylab.imshow(L_phase)
   
L_treated_bool = make_bool(L_treated_surface)
L_wt_bool=make_bool(L_wt_surface)
L_treated_sum=summ(L_treated_bool)
L_wt_sum=summ(L_wt_bool)

L_treated_result=divide(L_treated_sum,L_treated_tracks)
L_no_result=divide(L_wt_sum,L_wt_tracks)



    

