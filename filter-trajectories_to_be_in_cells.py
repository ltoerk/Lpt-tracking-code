# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 17:24:54 2017

@author: lisa
"""
#######################
### Import packages ###
#######################
import numpy as np
import pylab
import skimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math

##################
### Set values ###
##################

pixel_size = 6.25 #pixels per micron
time_steps = 0.1 #time in s

###################
### import data ###
###################

#read data that were exported by trackmate
data = pd.read_csv('14022022_LT16_FG_1uL_of_0_02_per_ara_tirf_003_07_7_02_306.csv')

#import the threshod mask of phase image before timelaps
phase_1 = np.loadtxt('mask_1.txt')
#import the threshold mask of phase image after timelaps
phase_2 = np.loadtxt('mask_2.txt')
#make mask combining before and after phase image
phase_combine =  phase_1+ phase_2
#change "255 value from Fiji to 1 (all values > 1 = 1)
phase_combine[phase_combine > 1] = 1
#function to plot phase image

#################
### Functions ###
#################
def plot_phase(image):
    '''
    takes in numpy array and plot as image
    '''
    pylab.imshow(phase_combine)

def check_in_cell(something, percentage):
    '''
    takes in boolean numpy array and a percentage value 0-1
    return boolean weather number of TRUE in array higher than percentage value
    '''
    cut_off = len(something) * percentage
    return something.sum() >= cut_off
    
def test_if_xy_true(xy):
    '''
    takes in series of x and y values
    (coordinates) returns array with booleans identifying
    wether the coordinates are at 1 or 0 in phase_combine array
    '''
    xy_arr = xy.values
    L=[]
    for i in range(len(xy_arr)): 
        L.append(phase_combine[int(xy_arr[i,0]),int(xy_arr[i,1])])
        value = np.array(L)
    return value
    
def plot_if_in_cell(grouped, percentage):
    '''
    takes in grouped df and percentage, uses percentage to check for each key
    wether track in cell or not, plottes tracks colored dependent on wether in cell or not
    '''
    for key,datas in grouped:
        if check_in_cell(grouped.get_group(key)['Incell'],percentage) == 1:
            plt.plot(datas['POSITION_X']*pixel_size, datas['POSITION_Y']*pixel_size, color='red')
        elif check_in_cell(grouped.get_group(key)['Incell'],percentage) == 0:
            plt.plot(datas['POSITION_X']*pixel_size, datas['POSITION_Y']*pixel_size, color='blue')
        else:
            plt.plot(datas['POSITION_X']*pixel_size, datas['POSITION_Y']*pixel_size, color='green')
            
def make_new_df_with_incell_tracks(data,percentage):
    '''
    takes in raw data and cut off percentage
    checks if tracks in cell or not using percentage cut off
    returns new dataframe only containing the tracks in cells
    '''
    data_new =data   
    grouped = data.groupby('TRACK_ID')    
    for key,datas in data_group:
        if check_in_cell(grouped.get_group(key)['Incell'],percentage) == 0:
            data_new= data_new.drop(grouped.get_group(key).index)
    return data_new 

def delete_tracks(data, track_id):
    '''
    takes in data, and list if track_ids to delete, delets the tracks and returns data 
    only containing non deleted tracks
    '''
    data_new = data
    grouped = data.groupby('TRACK_ID')
    for i in track_id:
        data_new = data_new.drop(grouped.get_group(i).index)
    return data_new     
def delete_tracks_in_area(data,x,y):
    '''
    takes in data, and list if track_ids to delete, delets the tracks and returns data 
    only containing non deleted tracks
    '''
    L=[]
    data_group = data.groupby('TRACK_ID')
    for key, data in data_group:
        if x-2 <= data['POSITION_X'].values[1] <= x+2 and y-2 <= data['POSITION_Y'].values[1] <= y+2:            
            L.append(key)  
    return L      

def plot_tracks(grouped):
    '''
    takes in data grouped by track_id
    plot xy values of the tracks in red
    '''    
    for key,datas in grouped:
        plt.plot(datas['POSITION_X']*pixel_size, datas['POSITION_Y']*pixel_size, color='red')
        
def plot_single_track(track):
    plt.plot(track['POSITION_X']*pixel_size, track['POSITION_Y']*pixel_size, color='red')
    

def calculate_track_length_time(grouped):
    '''
    takes in a grouped panda
    returns array with track legths
    '''
    track_length_ser = grouped.size()
    return track_length_ser.values

def plot_histogram(array):
    '''
    takes in array and plots it as histogram
    '''
    fig = plt.figure()
    ax= fig.add_subplot(1,1,1)
    sns.set(font_scale = 1.5)


    ax.hist(array, bins=40)
    plt.ylabel('number of tracks')
    plt.xlabel('time in s')
    start, end = ax.get_xlim()
#    FacetGrid.set(xticks=np.arange(start,end,1))
    ax.xaxis.set_ticks(np.arange(start, end, 5))
    
def get_tracks_per_squaremicron():
    '''
    return number of tracks per square micron
    '''
    area_cells = np.count_nonzero(phase_1)/pixel_size**2
    number_tracks =  len(real_tracks_df_grouped)      
    return number_tracks/area_cells  
    
def calc_vel(df):
    dist = df[['POSITION_T','POSITION_X','POSITION_Y']].diff().fillna(0.)
    dist['Distance'] = np.sqrt(dist.POSITION_X**2 + dist.POSITION_Y**2)
    dist['Velocity'] = dist.Distance/dist.POSITION_T
    df['Distance'] = dist.Distance
    df['Velocity'] = dist.Velocity
    return df
    
def euklidean((a,b),(c,d)):
    return  math.sqrt((a-c)**2+(b-d)**2)
    
def plot_trackdisplacement(data_group):
    L_x = []
    L_y = []
    for key, data in data_group:
        L_x.append(len(data)*time_steps)
        L_y.append(euklidean((data['POSITION_X'].values[0], data['POSITION_Y'].values[0]),(data['POSITION_X'].values[-1], data['POSITION_Y'].values[-1])))

    x_v =np.array(L_x)
    y_v = np.array(L_y)
    plt.plot(x_v,y_v,'ro')
    plt.ylabel('Track displacement uM')
    plt.xlabel('time in s')
    plt.title('Track displacement')

def calc_trackdisplacement(data_group):
    L_y = []
    for key, data in data_group:
        L_y.append(euklidean((data['POSITION_X'].values[0], data['POSITION_Y'].values[0]),(data['POSITION_X'].values[-1], data['POSITION_Y'].values[-1])))
    y_v = np.array(L_y)
    return y_v     
    
def calc_total_dist_moved(data_group):
    L_v = []
    for key,data in data_group:
        L_v.append(data['Distance'][1:].values.sum())
    return np.array(L_v)
    


############
### Code ###
############  

#group data by TRACK_ID     
data = calc_vel(data)   
data_group= data.groupby('TRACK_ID')
#a = delete_tracks_in_area(data, 60,28)
#b = delete_tracks_in_area(data,18,40)
#delete_list = a+b#+c#+d+e#+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w+x+z#+aa+bb
#data = delete_tracks(data,delete_list)

data_group = data.groupby('TRACK_ID')
##add new columns to data with XY positions in pixel 
data['POSITION_X_in_pixel']=data['POSITION_X']*pixel_size
data['POSITION_Y_in_pixel']=data['POSITION_Y']*pixel_size
##
###add new column with 1/0 for each coordinate depending on if in cell or not
data['Incell'] = test_if_xy_true(data[['POSITION_Y_in_pixel','POSITION_X_in_pixel']])
######
#####make new df only containing tracks in cells !!!GIVE PERCENTAGE
real_tracks_df = make_new_df_with_incell_tracks(data, 0.7)    
#####
#####group new df by tracks
real_tracks_df_grouped = real_tracks_df.groupby('TRACK_ID')
##
#####print number of tracks out of cells and in cells
print str(len(data_group)-len(real_tracks_df_grouped)) + " " + "tracks are considered to be out of cells"
print str(len(real_tracks_df_grouped)) + " " + "tracks are counted in cells"

plot_phase(phase_combine)
plot_tracks(real_tracks_df_grouped)

def safe_traj_np(data_group):
    '''
    takes in a trajectory grouped data frame. saves a 1D numpy array to the disk
    with each element beeing a trajectory (2D array with lanes X and Y position
    and rows time steps)
    '''
    L =[]
    for key, data in data_group:
        element = np.vstack((data['POSITION_X'], data['POSITION_Y']))
        L.append(element.T) 
    all_array = np.array(L)
    np.save('14022022_LT16_FG_1uL_of_0_02_per_ara_tirf_003_07_7_02_306.npy',all_array)

safe_traj_np(real_tracks_df_grouped)
