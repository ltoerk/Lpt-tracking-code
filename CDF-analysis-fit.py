# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:06:50 2017

@author: lisa
"""

import numpy
from pylab import plot, show, grid, axis, xlabel, ylabel, title
import matplotlib.pyplot as plt
import seaborn as sns 
import math
from scipy.spatial import distance
from math import sqrt
from scipy.stats import norm
import numpy as np
from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import  Model, Parameters

import matplotlib.pyplot as plt



    
def gen_list_real_tracks(data_group):
    L=[]
    for key, data in data_group:
        L.append(data[['POSITION_X','POSITION_Y']].values.T)
    return L

def gen_list_tracks_from_array(track_array):
    L_tracks = track_array.tolist()
    L_tracks_transp = []
    for i in range(len(L_tracks)):
        L_tracks_transp.append(L_tracks[i].T)
    return L_tracks_transp
    
def calc_r_sqaure(L_tr,lag_time):
    L = []
    n=lag_time
    for i in range(len(L_tr)):
        delta_x = L_tr[i][0][n:]-L_tr[i][0][:-n]
        delta_y = L_tr[i][1][n:]-L_tr[i][1][:-n]
        r_square = delta_x**2 + delta_y**2
        L = L + r_square.tolist()
    return np.array(L)
    

def plot_cdf_rsquare(r_sq_ar,string):
    x_val = np.sort(r_sq_ar)
    y_val = np.arange(1,len(x_val)+1)/float(len(x_val))
    return x_val, y_val
    
def filter_trajectories_by_length(L_data,cutoff_low,cutoff_up):
    '''
    takes in a 1d numpy of trajectories (each element is a trajectorie
    with rows corresponding to different time steps) and a cut off value (integer)
    returns a new numpy array of trajectories containing only trajectories longer than the
    cutoff value
    '''
    L_data_new=[]    
    for n in L_data:    
        L_index = []
        print len(n)
        for i in range(len(n)):
            if len(n[i]) < cutoff_low or len(n[i]) > cutoff_up:
                L_index.append(i)
                data_modi = np.delete(n,L_index)
        if len(L_index)!=0:
            L_data_new.append(data_modi)
        else:
            L_data_new.append(n)
    return L_data_new

def get_all_fit_values(L_fit_result,tau):
    L_w = []
    L_r1 = []  
    L_r2 = [] 
    L_xi = []
    L_sigma=[]
    for i in L_fit_result:
        L_w.append(i[0].best_values['w'])
        L_r1.append(i[0].best_values['r_1_square']/(4*tau))
        L_r2.append(i[0].best_values['r_2_square']/(4*tau))
#        L_sigma.append(sqrt(i.best_values['four_sigma_square']/4))
#        L_sigma.append(sqrt(i.best_values['four_sigma_square']/4))
    return np.array(L_w), np.array(L_r1),np.array(L_r2)
    
def calc(large_array, tau):
    l_tracks = gen_list_tracks_from_array(large_array)
    
    x, y = plot_cdf_rsquare(calc_r_sqaure(l_tracks,tau),'yo')
    model = Model(two_comp_diffusion_err)
    pars = model.make_params()
    pars = Parameters()
    pars.add('w', value=0.6, min=0, max = 1)
    pars.add('r_1_square', value=0.005, min=0)
    pars.add('r_2_square', value=0.025, min=0)
#    pars.add('four_sigma_square', value=0.0016, min=0)
    result = model.fit(y,pars,x=x)    
    
    sns.set(font_scale = 1.5)
#    plt.plot(x, y,         'bo', label='lag time = 200ms')
#    #plt.plot(x, result.init_fit, 'k--',label='initial fit')
#    plt.plot(x, result.best_fit, 'r-')
#    plt.legend(loc=4)
#    plt.show()
    return result, x, y

def two_comp_diffusion(x,w,r_1_square,r_2_square):
#    w = pars['w']
#    r_1_square = pars['r_1_square']
#    r_2_square = pars['r_2_square']
    return 1-(w*np.exp(-x/(r_1_square+0.0064))+(1-w)*np.exp(-x/(r_2_square+0.0064)))
    
def two_comp_diffusion_err(x,w,r_1_square,r_2_square):
#    w = pars['w']
#    r_1_square = pars['r_1_square']
#    r_2_square = pars['r_2_square']
    return 1-(w*np.exp(-x/(r_1_square+0.0064))+(1-w)*np.exp(-x/(r_2_square+0.0064)))

    
def one_comp_diffusion(x,r_zero_square):
    return 1-np.exp(-x/r_zero_square)
    
def three_comp_diffusion(x,w,b,r_1_square,r_2_square,r_3_square):
    return 1-(w*np.exp(-x/r_1_square)+b*np.exp(-x/r_2_square)+(1-w-b)*np.exp(-x/r_3_square))

    
"""
load tracks and make them to one array
"""
# insert the numpy file names which were exported from filter tracks in cells

L_data_con_1= np.array(np.load('.npy').tolist()+np.load('.npy').tolist()+np.load('.npy').tolist()+np.load('.npy').tolist()+np.load('.npy').tolist())
L_data_con_2= np.array(np.load('.npy').tolist()+np.load('.npy').tolist()+np.load('.npy').tolist()+np.load('.npy').tolist()+np.load('.npy').tolist())


L_data = [L_data_con_1,L_data_con_2]

L_data_filter = filter_trajectories_by_length(L_data,3,150)


def do_fit(L_data,tau):
    L_result = []    
    for i in L_data:
        L_result.append(calc(i,tau))
    return L_result
    
    
L_tau = [1,2,3,4]
L_result_taus = []
for i in L_tau:
    L_result_taus.append(do_fit(L_data_filter,i))
#    
w_1_580, r1_1_580,r2_1_580= get_all_fit_values(L_result_taus[0],0.1)
w_2_580, r1_2_580,r2_2_580 = get_all_fit_values(L_result_taus[1],0.2)
w_3_580, r1_3_580,r2_3_580= get_all_fit_values(L_result_taus[2],0.3)
w_4_580, r1_4_580,r2_4_580 = get_all_fit_values(L_result_taus[3],0.4)

#### plot the fit ####
#took out plot in functions and changed the return form calc function
fit_result,x,z=calc(L_data[1],2)

### print best fit values for tau=2
print "alpha value"
for i in w_2_580:
    print i
print "Diffusion constant 1"    
for i in r1_2_580:
    print i
print "Diffusion constant 2"
for i in r2_2_580:
    print i








         



#https://lmfit.github.io/lmfit-py/model.html