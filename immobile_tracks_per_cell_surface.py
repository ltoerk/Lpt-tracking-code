# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:28:45 2020

@author: lisa
"""

### plot multiple tracks ####
def plot_tracks_green(L_tracks):
    for i in L_tracks:
        plt.plot(i.T[0]*6.25,i.T[1]*6.25,'g')
def plot_tracks_red(L_tracks):
    for i in L_tracks:
        plt.plot(i.T[0]*6.25,i.T[1]*6.25,'r')
### plot single tracks in different colors ###       
def plot_track_green(track):
    plt.plot(track.T[0]*6.25,track.T[1]*6.25,'g')
def plot_track_red(track):
    plt.plot(track.T[0]*6.25,track.T[1]*6.25,'r')
def plot_phase(L_phase):
    pylab.imshow(L_phase)
def make_bool(L):
    L_bool=[]
    for i in L:
        L_bool.append(i!=0)
    return L_bool
    
def get_imm_tracks(track,conf_radius,surface,radius):
    """
    takes in the tracks for one movie, the corresponding confinement radius results, the phase image txt file, 
    and an integer number radius which is the maximum radius possible to still be considered imobile trajectorie
    returns: number of cell surface, number of immobile tracks, numer of all tracks
    """
    plt.figure()
    ## take car of phase ##
    plot_phase(surface)
    surface_bool = make_bool(surface)
    surface_sum=np.sum(surface_bool)
    
    ## count immobile tracks ##
    L_immo_tracks=[]
    for (i,j) in zip(track,conf_radius):
        if j <=radius:
            plot_track_green(i)
            L_immo_tracks.append(i)
        else:
            plot_track_red(i)
    return surface_sum, len(L_immo_tracks), len(track)
###################################################################################################

#### get the confinement radius results for the data imported with tracks per cell surface tracks_per_cell_surface.py ####
## ENTER DATA HERE ##
data_tracks = L_wt_tracks
data_surface=L_wt_surface
L_dtc, L_conf_rad =run_radius_all(data_tracks) #this function is in confinement_radius_calculation.py

L_result=[]
for i in range(len(data_tracks)):
    L_result.append(get_imm_tracks(data_tracks[i],L_conf_rad[i],data_surface[i],0.07))
L_immo_tracks_per_sur=[]
L_immo_tracks_per_all=[]   
for i in L_result:
    print "immobile tracks/surface"
    itps=i[1]/np.float(i[0])
    L_immo_tracks_per_sur.append(itps)
    print itps
    print "immobile tracks/all tracks"
    itpt=i[1]/np.float(i[2])
    L_immo_tracks_per_all.append(itpt)
    print itpt
    print "  "
    
np_itps=np.mean(np.array(L_immo_tracks_per_sur))
np_itpt=np.mean(np.array(L_immo_tracks_per_all))
np_itps_std=np.std(np.array(L_immo_tracks_per_sur))
np_itpt_std=np.std(np.array(L_immo_tracks_per_all))

print np_itps
print np_itpt
print np_itps_std
print np_itpt_std