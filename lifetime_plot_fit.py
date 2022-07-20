# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:01:21 2021

@author: lisa
"""
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression
def filter_data(data):
    data_ns=cut_small_values(data,1.5)
    data_ns_nl=cut_large_values(data_ns,60)
    return data_ns_nl
# Function to calculate the exponential with constants a and b
def exponential(x, a, b):
    return a*np.exp(b*x)
    

def cdf(x,k):
    return 1-np.exp(-k*x)
    
def cdf_2_order(x,k,b):
    return 1-0.5*np.exp(-(x-k)/b)  ##laplace distribution for x>=k

def exponential_dist_2(x,k):
    return k*np.exp(-k*x)
def exponential_dist(x,k):
    return np.exp(-k*x)
def bi_expo(x,k,l):
    return np.exp(-k*x)+np.exp(-l*x)
    
def bi_expo_with_alpha(x,k,l,a):
    return a*np.exp(-k*x)+(1-a)*np.exp(-l*x)
    
def bi_expo_with_alpha_fixed_l(x,a,k):
    return a*np.exp(-0.08*x)+(1-a)*np.exp(-k*x)
    
def bi_expo_fixed_l(x,k,l):
    return np.exp(-k*x)+np.exp(-l*x)
    
def bi_expo_fixed_l_multi(x,l):
    return np.exp(-0.2*x)*np.exp(-l*x)
#    return np.exp(-k*x)*np.exp(-0.08*x)
    
def calc(x,y,func,c,lab,tit):
    model = Model(func)
    pars = model.make_params()
    pars = Parameters()
    pars.add('k', value=0.02, min=0, max = 0.5)
    result = model.fit(y,pars,x=x) 
#    plt.figure()
    sns.set(font_scale = 1.5)
    sns.set_style('white')
    plt.plot(x, y,'o',markersize=3,color=c,label=lab)
    plt.plot(x, result.best_fit, '-',linewidth=2,color=c)
    plt.title(tit)
    plt.legend(loc='best')
    plt.show()       
    return result
    
def calc_2(x,y,func,c,lab,tit):
    model = Model(func)
    pars = model.make_params()
    pars = Parameters()
    pars.add('k', value=0.1, min=0)
#    pars.add('l', value=0.01, min=0)
    pars.add('a', value=0.5, min=0,max=1)
#    pars.add('b', value=0.2, min=0,max=1)
    result = model.fit(y,pars,x=x) 
#    plt.figure()
    sns.set(font_scale = 1.5)
    plt.plot(x,y,'o',markersize=1.5,color=c,label=lab)
    plt.plot(x, result.best_fit, '-',linewidth=4,color=c)
    plt.title('biex')
    plt.legend(loc=1)
    plt.show()
    return result
    
def calc_3(x,y,func,c,lab,tit):
    model = Model(func)
    pars = model.make_params()
    pars = Parameters()
    pars.add('k', value=0.1, min=0)
    pars.add('l', value=0.01, min=0)
    pars.add('a', value=0.5, min=0,max=1)
#    pars.add('b', value=0.2, min=0,max=1)
    result = model.fit(y,pars,x=x) 
#    plt.figure()
    sns.set(font_scale = 1.5)
    plt.plot(x,y,'o',markersize=1.5,color=c,label=lab)
    plt.plot(x, result.best_fit, '-',linewidth=4,color=c)
    plt.title('biex')
    plt.legend(loc=1)
    plt.show()
    return result
    
def plot_hist_new(array,st,b,c):
    n, bins = np.histogram(array, bins=np.arange(min(array), max(array) + b, b))#b=0.01 used for the radius files
    L_new = []
    for i in range(len(bins)-1):
        L_new.append((bins[i]+bins[i+1])/float(2))
    bin_new = np.array(L_new)
    n_new=n/float(len(array))
    print np.log(n_new)
    plt.plot(bin_new,n_new,marker="o",label=st, linewidth=2,ms=6,color=c)
    plt.legend(loc='best')
    plt.xlim([0,60])
    sns.set_style("white")
    plt.legend(frameon=False)
    sns.despine()
    plt.show()
    return [bin_new,n_new]
    
def linear_fit(x_data,y_data,c,lab,tit):
    x_f=x_data
    y=np.log(y_data)
    
    x=x_f.reshape((-1, 1))
    model = LinearRegression(fit_intercept=False).fit(x, y)
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    
    print('slope:', model.coef_)
    y_pred = model.predict(x)
#    plt.figure()
    plt.plot(x,y,'o',markersize=1.5,color=c,label=lab)
    plt.plot(x,y_pred,'-',color=c)
    plt.legend(loc=3, prop={'size': 8})
    plt.title(tit)
    plt.legend(frameon=False)
    return [r_sq,model.intercept_,model.coef_[0],len(x)]
    
def linear_fit_second_order(x_data,y_data,c,lab,tit):
    x_f=x_data
    y_t=1/(y_data)
    y=y_t-1   ####this way the intercept is forced to 1
    
    x=x_f.reshape((-1, 1))
    model = LinearRegression(fit_intercept=False).fit(x, y)
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    
    print('slope:', model.coef_)
    y_pred = model.predict(x)
#    plt.figure()
    plt.plot(x,y,'o',markersize=1.5,color=c,label=lab)
    plt.plot(x,y_pred,'-',color=c)
    plt.title(tit)
    plt.legend(loc=2, prop={'size': 8})
    plt.legend(frameon=False)
    return [r_sq,model.intercept_,model.coef_[0],len(x)]   
    
def get_mid_point(x_steps,x_cdf,y_cdf_rev):  
    L_y_val_new=[]
    for i in x_steps:
        points = np.where(x_cdf==i)
        y_points=y_cdf_rev[points]
        L_y_val_new.append(np.mean(y_points))
    nan_tf=np.isnan(L_y_val_new)
    nan_val=np.where(nan_tf==True)
    y_val_final=np.delete(L_y_val_new,nan_val)
    x_val_fin=np.delete(x_steps,nan_val)
    return [x_val_fin,y_val_final]
    
def survival_probability(lifetime):
    counts=np.arange(min(lifetime),max(lifetime),0.5)
    L=[]
    for i in counts:
        L.append(np.sum(lifetime>i)/float(len(lifetime)))
    return [counts,np.array(L)]
    
def cut_small_values(data,num):
    L_short=[]
    for i in data:
        if i > num:
            L_short.append(i)
    return L_short
def cut_large_values(data,num):
    data_short=[]
    for i in data:
        if i < num:
            data_short.append(i)
    return data_short
def filter_data(data):   
    data_ns=cut_small_values(data,1.5)
    data_ns_nl=cut_large_values(data_ns,60)
    return data_ns_nl
    
L_label_3=['1','2','3','4','5','6']
L_color_3=['#bebada','#bebada','#8dd3c7','#8dd3c7','#fdb462','#fdb462']
L_ind_3=[0,1,2,3,4,5]


L=[]##insert list of numpy arrays containing lifetime data ##]
L_results_wt=[]
for i in L:
    L_results_wt.append(filter_data(i))

sp=[]
for i in range(len(L_results_wt)):
    sp.append(survival_probability(L_results_wt[i]))

  
L_ind_2=[2,0,4,5]

plt.figure()
sns.set_style("ticks")
plt.plot(sp[5][0],sp[5][1],'o',markersize=6,color=L_color_3[5],label=L_label_3[5])
plt.plot(sp[4][0],sp[4][1],'o',markersize=6,color=L_color_3[4],label=L_label_3[4])
plt.plot(sp[2][0],sp[2][1],'o',markersize=6,color=L_color_3[2],label=L_label_3[2])
plt.plot(sp[3][0],sp[3][1],'o',markersize=6,color=L_color_3[3],label=L_label_3[3])
plt.plot(sp[1][0],sp[1][1],'o',markersize=6,color=L_color_3[1],label=L_label_3[1])
plt.plot(sp[0][0],sp[0][1],'o',markersize=6,color=L_color_3[0],label=L_label_3[0])
leg = plt.legend()
leg.get_frame().set_linewidth(0.0)
  
  
fig_1=plt.figure()
L_res_60_biexp_sp=[]
for i in L_ind_3:
    L_res_60_biexp_sp.append(calc_3(sp[i][0],sp[i][1],bi_expo_with_alpha,L_color_3[i],L_label_3[i],"sp exp(-kx)+e(-lx)"))

fig_2=plt.figure()
L_res_60_exp_sp=[]
for i in L_ind_3:
    L_res_60_exp_sp.append(calc(sp[i][0],sp[i][1],exponential_dist,L_color_3[i],L_label_3[i],"sp exp(-kx)"))
 
 
