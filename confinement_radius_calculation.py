# -*- coding: utf-8 -*-
"""
Created on Tue Feb 05 12:58:33 2019

@author: lisa
"""
import math

pixel_size = 6.25


def get_dist_per_traj(L_tra):
    '''
    takes in a list of tranjectories, traj are 2D numpy arrays([0]=x,[1]=y)
    calculates for each trajectoris the r_sqaures and eukedian distances for each step
    returns two lists: one of r_sqaures and other list of eukledian distances for each traj
    '''
    L_dist_per_tr=[]
    L_r_sqaure_per_tr = []   
    for i in range (len(L_tra)):
        delta_x = L_tra[i][0][1:]-L_tra[i][0][:-1]
        delta_y = L_tra[i][1][1:]-L_tra[i][1][:-1]
        r_sqaure = delta_x**2 + delta_y**2
        L_dist_per_tr.append(np.sqrt(r_sqaure))
        L_r_sqaure_per_tr.append(r_sqaure)
    return L_dist_per_tr, L_r_sqaure_per_tr

def slice_traj_step_size(traj,dist,max_d):
    '''
    takes in trajectory 2D numpy traj[0]=x, traj[1]=y. takes in 1d numpy with
    corresponding distances to the trajectorie, and the max distance
    returns array of sliced trajectories that all have distances < max distance
    '''
    a = np.where(dist<max_d)
    if len(a[0]) > 0:
        b = np.split(a[0],  np.where(a[0][1:] - a[0][:-1] > 1)[0] +1)
        L_small=[]    
        for i in range(len(b)):
            c=traj[b[i][0]:b[i][-1]+2] 
            if len(c) < 40:
                L_small.append(c)
        return L_small
    
def get_centroid(L_sm_d_t):
    L_centroids=[]
    for i in L_sm_d_t:
        ce_x = i.T[0].mean()
        ce_y = i.T[1].mean()
        L_centroids.append(np.array([ce_x,ce_y]))
    return L_centroids

def calc_dist_two_point(p,q):
    delta_x = q[0]-p[0]
    delta_y = q[1]-p[1]
    r_square = delta_x**2+delta_y**2
    dist = math.sqrt(r_square)
    return dist

def get_dist_to_centroid(L_small_dis_tr, L_centroids):
    L_dist_cetr = []
    for (i,j) in zip(L_small_dis_tr,L_centroids):
        L_d_t = []    
        for n in range(len(i)):
            L_d_t.append(calc_dist_two_point(i[n],j))
        L_dist_cetr.extend(L_d_t)
    return L_dist_cetr
def get_dist_to_centroid_sep(L_small_dis_tr, L_centroids):
    L_dist_cetr = []
    for (i,j) in zip(L_small_dis_tr,L_centroids):
        L_d_t = []    
        for n in range(len(i)):
            L_d_t.append(calc_dist_two_point(i[n],j))
        L_dist_cetr.append(L_d_t)
    return L_dist_cetr

def plot_trajectorie_red(L_tr):
    for i in  L_tr:
        lL
        plt.plot(i.T[0]*pixel_size,i.T[1]*pixel_size,color="red")
        
def plot_trajectorie_green(L_tr):
    for i in  L_tr:
        plt.plot(i.T[0]*pixel_size,i.T[1]*pixel_size,color="green")
def plot_centroids(L_centroids):
        for i in L_centroids:
            plt.plot(i[0]*pixel_size,i[1]*pixel_size,'bo')
def plot_radius_around_centroid(L_centroids,rad):
    fig = plt.gcf()
    ax = fig.gca()
    for i in L_centroids:
        circle = plt.Circle(i*pixel_size,rad,color='b',fill=False)
        ax.add_artist(circle)
    plt.axis("equal")
def get_small_dist_traj(L_t_t_data,L_dist_per_tr_input):
    L_small_dis_tr_f =[]
    for (i,j) in zip(L_t_t_data,L_dist_per_tr_input):
        L_sliced_traj = slice_traj_step_size(i,j,0.09)
        if L_sliced_traj!=None:
            L_small_dis_tr_f.extend(L_sliced_traj)
    L_lon_sm_tr=[]
    for i in L_small_dis_tr_f:
        if len(i) > 2:
            L_lon_sm_tr.append(i)
    return L_lon_sm_tr
def get_radius_stat(L_dist_to_centr_f):
    m_radius_f = np.array(L_dist_to_centr_f).mean()
    m_radius_std_f = np.array(L_dist_to_centr_f).std()
    m_radius_variance_f = np.array(L_dist_to_centr_f).var()
    m_radius_variance_dd_f = np.array(L_dist_to_centr_f).var(ddof=1)
    return [m_radius_f, m_radius_std_f, m_radius_variance_f, m_radius_variance_dd_f]
def get_dist_arr(L_dist_per_tr_f):
    L_tr_dist_f=[]
    for i in L_dist_per_tr_f:
        L_tr_dist_f.extend(i)
    return np.array(L_tr_dist_f)
def get_len_sm_tr_ar(L_small_dis_tr_f):
    L_len_sm_tr_f=[]
    for i in L_small_dis_tr_f:
        L_len_sm_tr_f.append(len(i))
    return np.array(L_len_sm_tr_f)
def filter_data_tr_len(data_f,t_l):
    L_data_f=[]    
    for i in data_f:
        if len(i) > t_l:
            L_data_f.append(i)
    return np.array(L_data_f)
def cut_traj_small(data_f):
    L_cut =[]
    for i in data_f:
        if len(i) > 4:
            l = len(i)
            div = l/4
            L_cut.append(i[0:div])
            L_cut.append(i[div:div*2])
            L_cut.append(i[div*2:div*3])
            L_cut.append(i[div*3:])
        else:
            L_cut.append(i)    
    return L_cut


def run_r(data):
    data = cut_traj_small(data)
    L_t =[]
    for i in data:
        L_t.append(i.T)
    L_t_t = []
    for i in data:
        L_t_t.append(i)
        
    ## get a list of 1D numpyes that contain the euklidean distances for the traj ##
    L_dist_per_tr, L_r_square_per_tr = get_dist_per_traj(L_t)
    ## slice the trajecories so that only trajectory pieces with distances < max_dist retain
    L_small_dis_tr = get_small_dist_traj(L_t_t,L_dist_per_tr)  
    ## get list of centroids ##
    L_centroids = get_centroid(L_t_t)
    ## get list of distances from the coordinates to the centroids ""
    L_dist_to_centr = get_dist_to_centroid(L_t_t,L_centroids)
    ## get the mean of the distances and take it as radius ##
    L_radius_stat = get_radius_stat(L_dist_to_centr)
    ## get an array with all the distances from all trajectories ##
    dis_ca3 = get_dist_arr(L_dist_per_tr)
    ## get an array with the len of the small trajectory pieces ##
    len_ca3 = get_len_sm_tr_ar(L_t_t)
    
    cent_ca3=np.array(L_dist_to_centr)
    return dis_ca3,len_ca3,cent_ca3
    
def get_radius(data):
    L=data
    L_dis_a=[]  
    L_len_a=[] 
    L_cent_a=[] 
    for i in L:
        d,l,c=run_r(i)
        L_dis_a.append(d)
        L_len_a.append(l)
        L_cent_a.append(c)
    return L_cent_a

def run_radius_all(all_data_unpacked):
    L_dtc_all=[]
    L_conf_rad_all=[]
    for b in all_data_unpacked:
        data=b         # make two types of data ###
        L_t=[]
        for i in data:
            L_t.append(i)
        L_t_t=[]
        for i in data:
            L_t_t.append(i.T)        
                        
                    ## get a list of 1D numpyes that contain the euklidean distances for the traj ##
        L_dist_per_tr, L_r_square_per_tr = get_dist_per_traj(L_t_t)
                    ## slice the trajecories so that only trajectory pieces with distances < max_dist retain
        L_small_dis_tr = get_small_dist_traj(L_t_t,L_dist_per_tr)  
                    ## get list of centroids ##
        L_centroids = get_centroid(L_t)
                    ## get list of distances from the coordinates to the centroids ""
        L_dist_to_centr = get_dist_to_centroid(L_t,L_centroids)
                    ## get the mean of the distances and take it as radius ##
        L_radius_stat = get_radius_stat(L_dist_to_centr)
                    ## get an array with all the distances from all trajectories ##
        dis_c = get_dist_arr(L_dist_per_tr)
                    ## get an array with the len of the small trajectory pieces ##
        len_c = get_len_sm_tr_ar(L_t_t)
                    
        cent_c=np.array(L_dist_to_centr)
        dist_centr_sep = get_dist_to_centroid_sep(L_t,L_centroids)
        L_radi=[]
        for i in dist_centr_sep:
            L_radi.append(np.mean(i))
        L_dtc_all.append(cent_c)
        L_conf_rad_all.append(np.array(L_radi))
    return L_dtc_all, L_conf_rad_all
    
def plot_hist_new(array,st,b,c1,c2,mark):
#    (n,bins,patches)=plt.hist(array,bins=20)
    n, bins = np.histogram(array, bins=np.arange(min(array), max(array) + b, b))#b=0.01 used for the radius files
    L_new = []
    for i in range(len(bins)-1):
        L_new.append((bins[i]+bins[i+1])/float(2))
    bin_new = np.array(L_new)
    n_new=n/float(len(array))
    print np.log(n_new)
    plt.plot(bin_new,n_new,marker=mark, label=st,linewidth=3,ms=8,color=c1,markeredgewidth=0.0)
    plt.plot(bin_new,n_new, ms=8,color=c2,markeredgewidth=0.0)
    plt.legend(loc='best')
    sns.set_style("ticks")
    plt.xlim([0,0.2])
    sns.set_style("white")
    plt.legend(frameon=False)
    sns.despine()
    plt.show()
    return [bin_new,n_new]
def filter_specific_frame_number(L_tr,num):
    L_filtered=[]    
    for i in L_tr:
        if len(i)==num:
            L_filtered.append(i)
    return L_filtered
    
def filter_and_cut_to_frame_number(L_tr,num):
    L_filtered=[]
    for i in L_tr:
        if len(i)>=num:
            L_filtered.append(i[:num])
    return L_filtered
    
"""
Import list of numpy array containing trajectories
"""    
L_data=[]   

L_data_filter=[]
for i in L_data:
    L_data_filter.append(filter_and_cut_to_frame_number(i,5))
 
L_dtc_all_10,L_conf_rad = run_radius_all(L_data_filter)   
fig_1 = plt.figure()
sns.set_style("ticks")

#
plot_hist_new(L_conf_rad[0],'18_0',0.01,'#fc9e35','#fc9e35','o')
plot_hist_new(L_conf_rad[1],'18_1',0.01,'#fc9e35','#fc9e35','s')
#plot_hist_new(L_conf_rad[2],'18_5',0.01,'#fc9e35','#fc9e35','P')
#plot_hist_new(L_conf_rad[3],'18_15',0.01,'#fc9e35','#fc9e35','d')
#plot_hist_new(L_conf_rad[4],'18_30',0.01,'#fc9e35','#fc9e35','v')
sns.set_style("ticks")