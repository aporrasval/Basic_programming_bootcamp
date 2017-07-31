

from astropy.io import fits
from scipy import stats
from scipy import spatial as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
#get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set_style('darkgrid')


# In[3]:

load_z0DSage = fits.open('tao.1951.0.fits')
load_z1DSage = fits.open('tao.1959.0.fits')
load_z2DSage= fits.open('tao.1961.0.fits')
load_z3DSage = fits.open('tao.1957.0.fits')

data_z0DSage = load_z0DSage[1].data
data_z1DSage = load_z1DSage[1].data
data_z2DSage = load_z2DSage[1].data
data_z3DSage = load_z3DSage[1].data


# In[4]:

# Dividing sample between centrals and satellites for DARK SAGE
idx_cen_z0DSage = np.where(data_z0DSage[:]['Galaxy_Classification'] == 0) [0]
idx_sat_z0DSage  = np.where(data_z0DSage[:]['Galaxy_Classification'] == 1) [0]
gal_cen_z0DSage = data_z0DSage[idx_cen_z0DSage]
gal_sat_z0DSage = data_z0DSage[idx_sat_z0DSage]

idx_cen_z1DSage = np.where(data_z1DSage[:]['Galaxy_Classification'] == 0) [0]
idx_sat_z1DSage  = np.where(data_z1DSage[:]['Galaxy_Classification'] == 1) [0]
gal_cen_z1DSage = data_z1DSage[idx_cen_z1DSage]
gal_sat_z1DSage = data_z1DSage[idx_sat_z1DSage]

idx_cen_z2DSage = np.where(data_z2DSage[:]['Galaxy_Classification'] == 0) [0]
idx_sat_z2DSage  = np.where(data_z2DSage[:]['Galaxy_Classification'] == 1) [0]
gal_cen_z2DSage = data_z2DSage[idx_cen_z2DSage]
gal_sat_z2DSage = data_z2DSage[idx_sat_z2DSage]

idx_cen_z3DSage = np.where(data_z3DSage[:]['Galaxy_Classification'] == 0) [0]
idx_sat_z3DSage  = np.where(data_z3DSage[:]['Galaxy_Classification'] == 1) [0]
gal_cen_z3DSage = data_z3DSage[idx_cen_z3DSage]
gal_sat_z3DSage = data_z3DSage[idx_sat_z3DSage]


# In[5]:

#### Taking all galaxies with TSM = 0 out of sample
TSM_idx_cen_z0DSage = np.where(gal_cen_z0DSage[:]['Total_Stellar_Mass']!=0)[0]
gal_cen_z0DSage = gal_cen_z0DSage[TSM_idx_cen_z0DSage]

TSM_idx_cen_z1DSage = np.where(gal_cen_z1DSage[:]['Total_Stellar_Mass']!=0)[0]
gal_cen_z1DSage = gal_cen_z1DSage[TSM_idx_cen_z1DSage]

TSM_idx_cen_z2DSage = np.where(gal_cen_z2DSage[:]['Total_Stellar_Mass']!=0)[0]
gal_cen_z2DSage = gal_cen_z2DSage[TSM_idx_cen_z2DSage]

TSM_idx_cen_z3DSage = np.where(gal_cen_z3DSage[:]['Total_Stellar_Mass']!=0)[0]
gal_cen_z3DSage = gal_cen_z3DSage[TSM_idx_cen_z3DSage]


######################################################################  Disk dom
gal_cen = {}
gal_cen_z0 = {}
gal_cen_z1 = {}
gal_cen_z2 = {}
gal_cen_z3 = {}
allgal_cen_DSage = [gal_cen_z0DSage, gal_cen_z1DSage, gal_cen_z2DSage, gal_cen_z3DSage]
gal_dict = [gal_cen_z0, gal_cen_z1, gal_cen_z2, gal_cen_z3]

i=0
for i in range(len(allgal_cen_DSage)):
    gal_cen["keys_z{0}".format(i)] = allgal_cen_DSage[i].dtype.names
    key=0
    for key in gal_cen['keys_z{0}'.format(i)]:
        gal_dict[i][key] = list(allgal_cen_DSage[i][key])

    gal_cen["total_z{0}".format(i)] = pd.DataFrame(gal_dict[i])

    gal_cen["points_z{0}".format(i)] = np.array(zip(gal_cen['total_z{0}'.format(i)]['X'].ravel(),
                                                     gal_cen['total_z{0}'.format(i)]['Y'].ravel(),
                                                     gal_cen['total_z{0}'.format(i)]['Z'].ravel()))
    
    gal_cen["tree_z{0}".format(i)] = sp.cKDTree(gal_cen["points_z{0}".format(i)])

    gal_cen["ck_arr_8_z{0}".format(i)] = gal_cen["tree_z{0}".format(i)].query_ball_point(gal_cen["points_z{0}".format(i)], 8)

    ii=0
    gal_cen["ngal_arr_8_z{0}".format(i)] = np.array([len(gal_cen["ck_arr_8_z{0}".format(i)][ii])-1 for ii in range(len(gal_cen["ck_arr_8_z{0}".format(i)]))]).astype(float)
    
    #allgal_cen_DSage[i]['Ngal_counts_R8'] = gal_cen['ngal_arr_8_z{0}'.format(i)]
    gal_cen["keys_z{0}".format(i)] = gal_cen["ngal_arr_8_z{0}".format(i)].dtype.names
    gal_cen["g_dict_z{0}".format(i)] = list(gal_cen["ngal_arr_8_z{0}".format(i)])
    gal_cen["ngal_tot_z{0}".format(i)] = pd.DataFrame(gal_cen["g_dict_z{0}".format(i)], columns = ['Ngal_counts_R8'])
    gal_cen['tot_z{0}'.format(i)] = pd.concat([gal_cen["total_z{0}".format(i)], gal_cen["ngal_tot_z{0}".format(i)]], axis=1)



# gal_cen_keys = gal_cen['ngal_arr_8_z0'].dtype.names
# g_dict = list(gal_cen['ngal_arr_8_z0'])
# gal_cen['ngal_tot_z0'] = pd.DataFrame(g_dict, columns = ['Ngal_counts_R8'])
# result = pd.concat([gal_cen['total_z0'], gal_cen['ngal_tot_z0']], axis=1)


"""
#### Taking all galaxies with DSM = 0 out of sample
DSM_idx_cen_z0DSage = np.where(gal_cen_z0DSage[:]['Disk_Stellar_Mass']!=0)[0]
gal_cen_z0DSage = gal_cen_z0DSage[DSM_idx_cen_z0DSage]

DSM_idx_cen_z1DSage = np.where(gal_cen_z1DSage[:]['Disk_Stellar_Mass']!=0)[0]
gal_cen_z1DSage = gal_cen_z1DSage[DSM_idx_cen_z1DSage]

DSM_idx_cen_z2DSage = np.where(gal_cen_z2DSage[:]['Disk_Stellar_Mass']!=0)[0]
gal_cen_z2DSage = gal_cen_z2DSage[DSM_idx_cen_z2DSage]

DSM_idx_cen_z3DSage = np.where(gal_cen_z3DSage[:]['Disk_Stellar_Mass']!=0)[0]
gal_cen_z3DSage = gal_cen_z3DSage[DSM_idx_cen_z3DSage]
"""

###################################################################### Bulge dom
bulgedom_cen_DSage = {}
diskdom_cen_DSage = {}
i=0
for i in range(len(allgal_cen_DSage)):
    gal_cen["morph_z{0}".format(i)] = gal_cen['tot_z{0}'.format(i)]['Disk_Stellar_Mass']/gal_cen['tot_z{0}'.format(i)]['Total_Stellar_Mass']
    bulgedom_cen_DSage["idx_z{0}".format(i)] = np.where(gal_cen['morph_z{0}'.format(i)] < 0.5)[0]
    bulgedom_cen_DSage["total_z{0}".format(i)] = gal_cen['tot_z{0}'.format(i)].loc[bulgedom_cen_DSage['idx_z{0}'.format(i)],:]

    diskdom_cen_DSage["idx_z{0}".format(i)] = np.where(gal_cen['morph_z{0}'.format(i)] > 0.5)[0]
    diskdom_cen_DSage["total_z{0}".format(i)] = gal_cen['tot_z{0}'.format(i)].loc[diskdom_cen_DSage['idx_z{0}'.format(i)],:]

# In[13]:

i=0
for i in range(len(allgal_cen_DSage)):
    
    bulgedom_cen_DSage["Mvir_z{0}".format(i)] = 1e10*bulgedom_cen_DSage['total_z{0}'.format(i)]['Mvir']
    bulgedom_cen_DSage["TSM_z{0}".format(i)] = 1e10*bulgedom_cen_DSage['total_z{0}'.format(i)]['Total_Stellar_Mass']    
    bulgedom_cen_DSage["CGMvir_z{0}".format(i)] = 1e10*bulgedom_cen_DSage['total_z{0}'.format(i)]['Central_Galaxy_Mvir']
    bulgedom_cen_DSage["bulgepseudo_z{0}".format(i)] = 1e10*bulgedom_cen_DSage['total_z{0}'.format(i)]['Pseudobulge_Mass']
    bulgedom_cen_DSage["DSM_z{0}".format(i)] = 1e10*bulgedom_cen_DSage['total_z{0}'.format(i)]['Disk_Stellar_Mass']
    bulgedom_cen_DSage["ColdGM_z{0}".format(i)] = 1e10*bulgedom_cen_DSage['total_z{0}'.format(i)]['Cold_Gas_Mass']    
    bulgedom_cen_DSage["HotGM_z{0}".format(i)] = 1e10*bulgedom_cen_DSage['total_z{0}'.format(i)]['Hot_Gas_Mass']
    bulgedom_cen_DSage["EjectedGM_z{0}".format(i)] = 1e10*bulgedom_cen_DSage['total_z{0}'.format(i)]['Ejected_Gas_Mass']     
    bulgedom_cen_DSage["Vvir_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['Vvir']
    bulgedom_cen_DSage["Rvir_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['Rvir']    
    bulgedom_cen_DSage["xpos_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['X']
    bulgedom_cen_DSage["ypos_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['Y']
    bulgedom_cen_DSage["zpos_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['Z']
    bulgedom_cen_DSage["xvel_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['X_Velocity']
    bulgedom_cen_DSage["yvel_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['Y_Velocity']
    bulgedom_cen_DSage["zvel_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['Z_Velocity']
    bulgedom_cen_DSage["xSDspin_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['X_Spin_of_Stellar_Disk']
    bulgedom_cen_DSage["ySDspin_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['Y_Spin_of_Stellar_Disk']
    bulgedom_cen_DSage["zSDspin_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['Z_Spin_of_Stellar_Disk']
    bulgedom_cen_DSage["xGDspin_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['X_Spin_of_Gas_Disk']
    bulgedom_cen_DSage["yGDspin_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['Y_Spin_of_Gas_Disk']
    bulgedom_cen_DSage["zGDspin_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['Z_Spin_of_Gas_Disk']
    bulgedom_cen_DSage["xhaloJ_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['jX_Halo']
    bulgedom_cen_DSage["yhaloJ_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['jY_Halo']
    bulgedom_cen_DSage["zhaloJ_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['jZ_Halo']      
    bulgedom_cen_DSage["JSD_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['j_Stellar_Disk']

    bulgedom_cen_DSage["NgalcountsR8_z{0}".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)]['Ngal_counts_R8'] 
    
    bulgedom_cen_DSage["pos_z{0}".format(i)] = np.sqrt(bulgedom_cen_DSage['xpos_z{0}'.format(i)]**2 + 
                                                    bulgedom_cen_DSage['ypos_z{0}'.format(i)]**2 + 
                                                    bulgedom_cen_DSage['zpos_z{0}'.format(i)]**2)
    
    bulgedom_cen_DSage["vel_z{0}".format(i)] = np.sqrt(bulgedom_cen_DSage['xvel_z{0}'.format(i)]**2 + 
                                                    bulgedom_cen_DSage['yvel_z{0}'.format(i)]**2 + 
                                                    bulgedom_cen_DSage['zvel_z{0}'.format(i)]**2)
    
    bulgedom_cen_DSage["SDspin_z{0}".format(i)] = np.sqrt(bulgedom_cen_DSage['xSDspin_z{0}'.format(i)]**2 + 
                                                    bulgedom_cen_DSage['ySDspin_z{0}'.format(i)]**2 + 
                                                    bulgedom_cen_DSage['zSDspin_z{0}'.format(i)]**2)
    
    bulgedom_cen_DSage["GDspin_z{0}".format(i)] = np.sqrt(bulgedom_cen_DSage['xGDspin_z{0}'.format(i)]**2 + 
                                                    bulgedom_cen_DSage['yGDspin_z{0}'.format(i)]**2 + 
                                                    bulgedom_cen_DSage['zGDspin_z{0}'.format(i)]**2)
    
    
    bulgedom_cen_DSage["haloJ_z{0}".format(i)] = np.sqrt(bulgedom_cen_DSage['xhaloJ_z{0}'.format(i)]**2 + 
                                                    bulgedom_cen_DSage['yhaloJ_z{0}'.format(i)]**2 + 
                                                    bulgedom_cen_DSage['zhaloJ_z{0}'.format(i)]**2) 
    


ii=0
for ii in range(len(allgal_cen_DSage)):
    diskdom_cen_DSage["Mvir_z{0}".format(ii)] = 1e10*diskdom_cen_DSage['total_z{0}'.format(ii)]['Mvir']
    diskdom_cen_DSage["TSM_z{0}".format(ii)] = 1e10*diskdom_cen_DSage['total_z{0}'.format(ii)]['Total_Stellar_Mass']
    diskdom_cen_DSage["CGMvir_z{0}".format(ii)] = 1e10*diskdom_cen_DSage['total_z{0}'.format(ii)]['Central_Galaxy_Mvir']
    diskdom_cen_DSage["bulgepseudo_z{0}".format(ii)] = 1e10*diskdom_cen_DSage['total_z{0}'.format(ii)]['Pseudobulge_Mass']
    diskdom_cen_DSage["DSM_z{0}".format(ii)] = 1e10*diskdom_cen_DSage['total_z{0}'.format(ii)]['Disk_Stellar_Mass'] 
    diskdom_cen_DSage["ColdGM_z{0}".format(ii)] = 1e10*diskdom_cen_DSage['total_z{0}'.format(ii)]['Cold_Gas_Mass']    
    diskdom_cen_DSage["HotGM_z{0}".format(ii)] = 1e10*diskdom_cen_DSage['total_z{0}'.format(ii)]['Hot_Gas_Mass']
    diskdom_cen_DSage["EjectedGM_z{0}".format(ii)] = 1e10*diskdom_cen_DSage['total_z{0}'.format(ii)]['Ejected_Gas_Mass']       
    diskdom_cen_DSage["Vvir_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['Vvir']
    diskdom_cen_DSage["Rvir_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['Rvir']    
    diskdom_cen_DSage["xpos_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['X']
    diskdom_cen_DSage["ypos_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['Y']
    diskdom_cen_DSage["zpos_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['Z']
    diskdom_cen_DSage["xvel_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['X_Velocity']
    diskdom_cen_DSage["yvel_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['Y_Velocity']
    diskdom_cen_DSage["zvel_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['Z_Velocity']
    diskdom_cen_DSage["xSDspin_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['X_Spin_of_Stellar_Disk']
    diskdom_cen_DSage["ySDspin_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['Y_Spin_of_Stellar_Disk']
    diskdom_cen_DSage["zSDspin_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['Z_Spin_of_Stellar_Disk']
    diskdom_cen_DSage["xGDspin_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['X_Spin_of_Gas_Disk']
    diskdom_cen_DSage["yGDspin_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['Y_Spin_of_Gas_Disk']
    diskdom_cen_DSage["zGDspin_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['Z_Spin_of_Gas_Disk']
    diskdom_cen_DSage["xhaloJ_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['jX_Halo']
    diskdom_cen_DSage["yhaloJ_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['jY_Halo']
    diskdom_cen_DSage["zhaloJ_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['jZ_Halo']  
    diskdom_cen_DSage["JSD_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['j_Stellar_Disk']

    diskdom_cen_DSage["NgalcountsR8_z{0}".format(ii)] = diskdom_cen_DSage['total_z{0}'.format(ii)]['Ngal_counts_R8'] 
    
    diskdom_cen_DSage["pos_z{0}".format(ii)] = np.sqrt(diskdom_cen_DSage['xpos_z{0}'.format(ii)]**2 + 
                                                    diskdom_cen_DSage['ypos_z{0}'.format(ii)]**2 + 
                                                    diskdom_cen_DSage['zpos_z{0}'.format(ii)]**2)
    
    diskdom_cen_DSage["vel_z{0}".format(ii)] = np.sqrt(diskdom_cen_DSage['xvel_z{0}'.format(ii)]**2 + 
                                                    diskdom_cen_DSage['yvel_z{0}'.format(ii)]**2 + 
                                                    diskdom_cen_DSage['zvel_z{0}'.format(ii)]**2)

    
    diskdom_cen_DSage["SDspin_z{0}".format(ii)] = np.sqrt(diskdom_cen_DSage['xSDspin_z{0}'.format(ii)]**2 + 
                                                    diskdom_cen_DSage['ySDspin_z{0}'.format(ii)]**2 + 
                                                    diskdom_cen_DSage['zSDspin_z{0}'.format(ii)]**2)
    
    diskdom_cen_DSage["GDspin_z{0}".format(ii)] = np.sqrt(diskdom_cen_DSage['xGDspin_z{0}'.format(ii)]**2 + 
                                                    diskdom_cen_DSage['yGDspin_z{0}'.format(ii)]**2 + 
                                                    diskdom_cen_DSage['zGDspin_z{0}'.format(ii)]**2)
    
    
    diskdom_cen_DSage["haloJ_z{0}".format(ii)] = np.sqrt(diskdom_cen_DSage['xhaloJ_z{0}'.format(ii)]**2 + 
                                                    diskdom_cen_DSage['yhaloJ_z{0}'.format(ii)]**2 + 
                                                    diskdom_cen_DSage['zhaloJ_z{0}'.format(ii)]**2)     


#################################################################################### Disc spin

i = 0
for i in range(len(allgal_cen_DSage)):     
    bulgedom_cen_DSage["slambda_z{0}".format(i)] = bulgedom_cen_DSage["JSD_z{0}".format(i)]/(np.sqrt(2)*bulgedom_cen_DSage["Vvir_z{0}".format(i)]*
                                                                bulgedom_cen_DSage["Rvir_z{0}".format(i)])

    diskdom_cen_DSage["slambda_z{0}".format(i)] = diskdom_cen_DSage["JSD_z{0}".format(i)]/(np.sqrt(2)*diskdom_cen_DSage["Vvir_z{0}".format(i)]*
                                                                diskdom_cen_DSage["Rvir_z{0}".format(i)])


failval = np.nan
# failval = -999
mean_func = np.nanmean
std_func = np.nanstd
Mvir_bins = np.linspace(0, 50, 12)

def meanbin(a, b):
    idx_Mvir = np.digitize(a, Mvir_bins)
    mean_a = np.array([ mean_func(a[idx_Mvir==ii]) if len(a[idx_Mvir==ii]) > 0 else failval for ii in range(1, len(Mvir_bins))])
    mean_b = np.array([ mean_func(b[idx_Mvir==ii]) if len(a[idx_Mvir==ii]) > 0 else failval for ii in range(1, len(Mvir_bins))])
    return mean_a, mean_b



def stdbin(a, b):
    idx_Mvir = np.digitize(a, Mvir_bins)
    std_a = np.array([ std_func(a[idx_Mvir==ii]) if len(a[idx_Mvir==ii]) > 0 else failval for ii in range(1, len(Mvir_bins))])
    std_b = np.array([ std_func(b[idx_Mvir==ii])/np.sqrt(len(b[idx_Mvir==ii])) if len(a[idx_Mvir==ii]) > 0 else failval for ii in range(1, len(Mvir_bins))])
    #std_meanb = np.array(std_b/np.sqrt(len(b[idx_Mvir==k])) for k in range(1, len(Mvir_bins)))
    return std_a, std_b


i = 0
for i in range(len(allgal_cen_DSage)):
    bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z{0}'.format(i)] = meanbin(bulgedom_cen_DSage['slambda_z{0}'.format(i)], bulgedom_cen_DSage['NgalcountsR8_z{0}'.format(i)])
    bulgedom_cen_DSage['slambda_Ngalcounts_stdbin_z{0}'.format(i)] = stdbin(bulgedom_cen_DSage['slambda_z{0}'.format(i)], bulgedom_cen_DSage['NgalcountsR8_z{0}'.format(i)])

    diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z{0}'.format(i)] = meanbin(diskdom_cen_DSage['slambda_z{0}'.format(i)], diskdom_cen_DSage['NgalcountsR8_z{0}'.format(i)])
    diskdom_cen_DSage['slambda_Ngalcounts_stdbin_z{0}'.format(i)] = stdbin(diskdom_cen_DSage['slambda_z{0}'.format(i)], diskdom_cen_DSage['NgalcountsR8_z{0}'.format(i)])




plt.clf()
plt.close()
fig = plt.figure(facecolor='white', figsize=(10, 8))

plt.subplot(221)
plt.errorbar(bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z0'][0],
             bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z0'][1],
            yerr=bulgedom_cen_DSage['slambda_Ngalcounts_stdbin_z0'][1],
            marker='o', color='r', label='Bulge')

plt.errorbar(diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z0'][0],
             diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z0'][1],
            yerr=diskdom_cen_DSage['slambda_Ngalcounts_stdbin_z0'][1],
            marker='o', color='b', label='Disk')

plt.title('Centrals at z=0')
#plt.xlim(0, 1)
#plt.ylim(-1, 21)
plt.xlabel('$\lambda_{Disk Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=2, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])


#plt.xlim(0, 1)
#plt.ylim(-1, 21)
plt.xlabel('$\lambda_{Disk Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=2, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])


plt.subplots_adjust(top=0.92, bottom=0.001, left=0.10, right=0.95, hspace=0.6,
                    wspace=0.35)
plt.style.use('seaborn-notebook')

plt.tight_layout()
#plt.savefig('morphdensity_Centrals_stellarSpin_lambdavsRvir_SMbin10-12_DSAGE_z0.png', dpi=100)

plt.show()


#################################################################################### Halo spin

i = 0
for i in range(len(allgal_cen_DSage)):     
    bulgedom_cen_DSage["hlambda_z{0}".format(i)] = bulgedom_cen_DSage["haloJ_z{0}".format(i)]/(np.sqrt(2)*bulgedom_cen_DSage["Vvir_z{0}".format(i)]*
                                                                bulgedom_cen_DSage["Rvir_z{0}".format(i)])

    diskdom_cen_DSage["hlambda_z{0}".format(i)] = diskdom_cen_DSage["haloJ_z{0}".format(i)]/(np.sqrt(2)*diskdom_cen_DSage["Vvir_z{0}".format(i)]*
                                                                diskdom_cen_DSage["Rvir_z{0}".format(i)])




Mvir_bins = np.linspace(0, 0.175, 10)


i = 0
for i in range(len(allgal_cen_DSage)):
    bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z{0}'.format(i)] = meanbin(bulgedom_cen_DSage['hlambda_z{0}'.format(i)], bulgedom_cen_DSage['NgalcountsR8_z{0}'.format(i)])
    bulgedom_cen_DSage['hlambda_Ngalcounts_stdbin_z{0}'.format(i)] = stdbin(bulgedom_cen_DSage['hlambda_z{0}'.format(i)], bulgedom_cen_DSage['NgalcountsR8_z{0}'.format(i)])

    diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z{0}'.format(i)] = meanbin(diskdom_cen_DSage['hlambda_z{0}'.format(i)], diskdom_cen_DSage['NgalcountsR8_z{0}'.format(i)])
    diskdom_cen_DSage['hlambda_Ngalcounts_stdbin_z{0}'.format(i)] = stdbin(diskdom_cen_DSage['hlambda_z{0}'.format(i)], diskdom_cen_DSage['NgalcountsR8_z{0}'.format(i)])




plt.clf()
plt.close()
fig = plt.figure(facecolor='white', figsize=(10, 8))

plt.subplot(221)
plt.errorbar(bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z0'][0],
             bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z0'][1],
            yerr=bulgedom_cen_DSage['hlambda_Ngalcounts_stdbin_z0'][1],
            marker='o', color='r', label='Bulge')

plt.errorbar(diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z0'][0],
             diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z0'][1],
            yerr=diskdom_cen_DSage['hlambda_Ngalcounts_stdbin_z0'][1],
            marker='o', color='b', label='Disk')

plt.title('Centrals at z=0')
#plt.xlim(0, 1)
#plt.ylim(-1, 21)
plt.xlabel('$\lambda_{Halo Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=2, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])


#plt.xlim(0, 1)
#plt.ylim(-1, 21)
plt.xlabel('$\lambda_{Halo Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=2, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])


plt.subplots_adjust(top=0.92, bottom=0.001, left=0.10, right=0.95, hspace=0.6,
                    wspace=0.35)
plt.style.use('seaborn-notebook')

plt.tight_layout()
#plt.savefig('morphdensity_Centrals_stellarSpin_lambdavsRvir_SMbin10-12_DSAGE_z0.png', dpi=100)

plt.show()






########################################################################## Binning Stellar Mass

# In[109]:

##################################### BULGE MASS #################################
bulgemassbin_cen_DSage = {}
i=0
    
for i in range(len(allgal_cen_DSage)):
   
    bulgemassbin_cen_DSage["bulge_idx_z{0}_SM8_9".format(i)] = np.where(np.logical_and(bulgedom_cen_DSage["TSM_z{0}".format(i)] >= 1e8, 
                                                                                     bulgedom_cen_DSage["TSM_z{0}".format(i)] <= 1e9))[0]
    
    bulgemassbin_cen_DSage["bulge_idx_z{0}_SM9_10".format(i)] = np.where(np.logical_and(bulgedom_cen_DSage["TSM_z{0}".format(i)] > 1e9, 
                                                                                     bulgedom_cen_DSage["TSM_z{0}".format(i)] <= 1e10))[0]
    
    bulgemassbin_cen_DSage["bulge_idx_z{0}_SM10_11".format(i)] = np.where(np.logical_and(bulgedom_cen_DSage["TSM_z{0}".format(i)] > 1e10, 
                                                                                     bulgedom_cen_DSage["TSM_z{0}".format(i)] <= 1e11))[0] 
   
    bulgemassbin_cen_DSage["bulge_idx_z{0}_SM11_12".format(i)] = np.where(np.logical_and(bulgedom_cen_DSage["TSM_z{0}".format(i)] > 1e11, 
                                                                                     bulgedom_cen_DSage["TSM_z{0}".format(i)] <= 1e12))[0]

    
    bulgemassbin_cen_DSage["bulgedom_z{0}_SM8_9".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)].loc[bulgemassbin_cen_DSage["bulge_idx_z{0}_SM8_9".format(i)],:]
    bulgemassbin_cen_DSage["bulgedom_z{0}_SM9_10".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)].loc[bulgemassbin_cen_DSage["bulge_idx_z{0}_SM9_10".format(i)],:]
    bulgemassbin_cen_DSage["bulgedom_z{0}_SM10_11".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)].loc[bulgemassbin_cen_DSage["bulge_idx_z{0}_SM10_11".format(i)],:]  
    bulgemassbin_cen_DSage["bulgedom_z{0}_SM11_12".format(i)] = bulgedom_cen_DSage['total_z{0}'.format(i)].loc[bulgemassbin_cen_DSage["bulge_idx_z{0}_SM11_12".format(i)],:]


# In[110]:

################################################ DISK MASS ####################################################
diskmassbin_cen_DSage = {}
i=0
    
for i in range(len(allgal_cen_DSage)):
   
    diskmassbin_cen_DSage["disk_idx_z{0}_SM8_9".format(i)] = np.where(np.logical_and(diskdom_cen_DSage["TSM_z{0}".format(i)] >= 1e8, 
                                                                                     diskdom_cen_DSage["TSM_z{0}".format(i)] <= 1e9))[0]
    
    diskmassbin_cen_DSage["disk_idx_z{0}_SM9_10".format(i)] = np.where(np.logical_and(diskdom_cen_DSage["TSM_z{0}".format(i)] > 1e9, 
                                                                                     diskdom_cen_DSage["TSM_z{0}".format(i)] <= 1e10))[0]
    
    diskmassbin_cen_DSage["disk_idx_z{0}_SM10_11".format(i)] = np.where(np.logical_and(diskdom_cen_DSage["TSM_z{0}".format(i)] > 1e10, 
                                                                                     diskdom_cen_DSage["TSM_z{0}".format(i)] <= 1e11))[0]
                                                                                   
    
    diskmassbin_cen_DSage["disk_idx_z{0}_SM11_12".format(i)] = np.where(np.logical_and(diskdom_cen_DSage["TSM_z{0}".format(i)] > 1e11, 
                                                                                     diskdom_cen_DSage["TSM_z{0}".format(i)] <= 1e12))[0]

    
    diskmassbin_cen_DSage["diskdom_z{0}_SM8_9".format(i)] = diskdom_cen_DSage['total_z{0}'.format(i)].loc[diskmassbin_cen_DSage["disk_idx_z{0}_SM8_9".format(i)],:]
    diskmassbin_cen_DSage["diskdom_z{0}_SM9_10".format(i)] = diskdom_cen_DSage['total_z{0}'.format(i)].loc[diskmassbin_cen_DSage["disk_idx_z{0}_SM9_10".format(i)],:]
    diskmassbin_cen_DSage["diskdom_z{0}_SM10_11".format(i)] = diskdom_cen_DSage['total_z{0}'.format(i)].loc[diskmassbin_cen_DSage["disk_idx_z{0}_SM10_11".format(i)],:] 
    diskmassbin_cen_DSage["diskdom_z{0}_SM11_12".format(i)] = diskdom_cen_DSage['total_z{0}'.format(i)].loc[diskmassbin_cen_DSage["disk_idx_z{0}_SM11_12".format(i)],:]




i = 0
j = 0
for i in range(len(allgal_cen_DSage)):
    for j in range(4):
        j = j+ 8
        bulgedom_cen_DSage["Mvir_z{0}_SM{1}_{2}".format(i,j,j+1)] = 1e10*bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Mvir']
        bulgedom_cen_DSage["TSM_z{0}_SM{1}_{2}".format(i,j,j+1)] = 1e10*bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Total_Stellar_Mass']
        bulgedom_cen_DSage["CGMvir_z{0}_SM{1}_{2}".format(i,j,j+1)] = 1e10*bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Central_Galaxy_Mvir']
        bulgedom_cen_DSage["bulgepseudo_z{0}_SM{1}_{2}".format(i,j,j+1)] = 1e10*bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Pseudobulge_Mass']
        bulgedom_cen_DSage["DSM_z{0}_SM{1}_{2}".format(i,j,j+1)] = 1e10*bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Disk_Stellar_Mass']            
        bulgedom_cen_DSage["Vvir_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Vvir']
        bulgedom_cen_DSage["Rvir_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Rvir']
        bulgedom_cen_DSage["xpos_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['X']
        bulgedom_cen_DSage["ypos_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Y']
        bulgedom_cen_DSage["zpos_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Z']
        bulgedom_cen_DSage["xvel_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['X_Velocity']
        bulgedom_cen_DSage["yvel_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Y_Velocity']
        bulgedom_cen_DSage["zvel_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Z_Velocity']
        bulgedom_cen_DSage["xSDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['X_Spin_of_Stellar_Disk']
        bulgedom_cen_DSage["ySDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Y_Spin_of_Stellar_Disk']
        bulgedom_cen_DSage["zSDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Z_Spin_of_Stellar_Disk']
        bulgedom_cen_DSage["xGDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['X_Spin_of_Gas_Disk']
        bulgedom_cen_DSage["yGDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Y_Spin_of_Gas_Disk']
        bulgedom_cen_DSage["zGDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Z_Spin_of_Gas_Disk']
        bulgedom_cen_DSage["xhaloJ_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['jX_Halo']
        bulgedom_cen_DSage["yhaloJ_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['jY_Halo']
        bulgedom_cen_DSage["zhaloJ_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['jZ_Halo']      
        bulgedom_cen_DSage["NgalcountsR8_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Ngal_counts_R8']
        bulgedom_cen_DSage["totparicles_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Total_Particles']
    
        bulgedom_cen_DSage["JSD_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgemassbin_cen_DSage['bulgedom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['j_Stellar_Disk']        
        
        
        bulgedom_cen_DSage["pos_z{0}_SM{1}_{2}".format(i,j,j+1)] = np.sqrt(bulgedom_cen_DSage['xpos_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    bulgedom_cen_DSage['ypos_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    bulgedom_cen_DSage['zpos_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2)
    
        bulgedom_cen_DSage["vel_z{0}_SM{1}_{2}".format(i,j,j+1)] = np.sqrt(bulgedom_cen_DSage['xvel_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    bulgedom_cen_DSage['yvel_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    bulgedom_cen_DSage['zvel_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2)
    
        bulgedom_cen_DSage["SDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = np.sqrt(bulgedom_cen_DSage['xSDspin_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    bulgedom_cen_DSage['ySDspin_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    bulgedom_cen_DSage['zSDspin_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2)
    
        bulgedom_cen_DSage["GDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = np.sqrt(bulgedom_cen_DSage['xGDspin_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    bulgedom_cen_DSage['yGDspin_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    bulgedom_cen_DSage['zGDspin_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2)
    
        bulgedom_cen_DSage["haloJ_z{0}_SM{1}_{2}".format(i,j,j+1)] = np.sqrt(bulgedom_cen_DSage['xhaloJ_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    bulgedom_cen_DSage['yhaloJ_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    bulgedom_cen_DSage['zhaloJ_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2) 
    


# In[112]:

#disk_gal_cen_DSage = [diskdom_cen_z0DSage, diskdom_cen_z1DSage, diskdom_cen_z2DSage, diskdom_cen_z3DSage]
i=0
j=0
for i in range(len(allgal_cen_DSage)):
    for j in range(4):
        j = j + 8
        diskdom_cen_DSage["Mvir_z{0}_SM{1}_{2}".format(i,j,j+1)] = 1e10*diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Mvir']
        diskdom_cen_DSage["TSM_z{0}_SM{1}_{2}".format(i,j,j+1)] = 1e10*diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Total_Stellar_Mass']
        diskdom_cen_DSage["CGMvir_z{0}_SM{1}_{2}".format(i,j,j+1)] = 1e10*diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Central_Galaxy_Mvir']
        diskdom_cen_DSage["bulgepseudo_z{0}_SM{1}_{2}".format(i,j,j+1)] = 1e10*diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Pseudobulge_Mass']
        diskdom_cen_DSage["DSM_z{0}_SM{1}_{2}".format(i,j,j+1)] = 1e10*diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Disk_Stellar_Mass']                    
        diskdom_cen_DSage["Vvir_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Vvir']
        diskdom_cen_DSage["Rvir_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Rvir']            
        diskdom_cen_DSage["xpos_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['X']
        diskdom_cen_DSage["ypos_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Y']
        diskdom_cen_DSage["zpos_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Z']
        diskdom_cen_DSage["xvel_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['X_Velocity']
        diskdom_cen_DSage["yvel_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Y_Velocity']
        diskdom_cen_DSage["zvel_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Z_Velocity']
        diskdom_cen_DSage["xSDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['X_Spin_of_Stellar_Disk']
        diskdom_cen_DSage["ySDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Y_Spin_of_Stellar_Disk']
        diskdom_cen_DSage["zSDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Z_Spin_of_Stellar_Disk']
        diskdom_cen_DSage["xGDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['X_Spin_of_Gas_Disk']
        diskdom_cen_DSage["yGDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Y_Spin_of_Gas_Disk']
        diskdom_cen_DSage["zGDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Z_Spin_of_Gas_Disk']
        diskdom_cen_DSage["xhaloJ_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['jX_Halo']
        diskdom_cen_DSage["yhaloJ_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['jY_Halo']
        diskdom_cen_DSage["zhaloJ_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['jZ_Halo']      
        diskdom_cen_DSage["NgalcountsR8_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Ngal_counts_R8']    
        diskdom_cen_DSage["totparicles_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['Total_Particles']    
        diskdom_cen_DSage["JSD_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskmassbin_cen_DSage['diskdom_z{0}_SM{1}_{2}'.format(i,j,j+1)]['j_Stellar_Disk']        
        
        
        
        diskdom_cen_DSage["pos_z{0}_SM{1}_{2}".format(i,j,j+1)] = np.sqrt(diskdom_cen_DSage['xpos_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    diskdom_cen_DSage['ypos_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    diskdom_cen_DSage['zpos_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2)
    
        diskdom_cen_DSage["vel_z{0}_SM{1}_{2}".format(i,j,j+1)] = np.sqrt(diskdom_cen_DSage['xvel_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    diskdom_cen_DSage['yvel_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    diskdom_cen_DSage['zvel_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2)
    
        diskdom_cen_DSage["SDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = np.sqrt(diskdom_cen_DSage['xSDspin_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    diskdom_cen_DSage['ySDspin_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    diskdom_cen_DSage['zSDspin_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2)
    
        diskdom_cen_DSage["GDspin_z{0}_SM{1}_{2}".format(i,j,j+1)] = np.sqrt(diskdom_cen_DSage['xGDspin_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    diskdom_cen_DSage['yGDspin_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    diskdom_cen_DSage['zGDspin_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2)
    
        diskdom_cen_DSage["haloJ_z{0}_SM{1}_{2}".format(i,j,j+1)] = np.sqrt(diskdom_cen_DSage['xhaloJ_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    diskdom_cen_DSage['yhaloJ_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2 + 
                                                    diskdom_cen_DSage['zhaloJ_z{0}_SM{1}_{2}'.format(i,j,j+1)]**2) 
    


# # # # # # # # # # # # # # # # Calculating Lambda Spin # # # # # # # # # # # # # # # # # # # # # # # # 

# # # # # # # # # # # # # # # # # # Disk Lambda # # # # # # # # # # # # # # # # # # # # # # # # # # 



i = 0
j = 0
for i in range(len(allgal_cen_DSage)):
    for j in range(4):
        j = j + 8        
        bulgedom_cen_DSage["slambda_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgedom_cen_DSage["JSD_z{0}_SM{1}_{2}".format(i,j,j+1)]/(np.sqrt(2)*bulgedom_cen_DSage["Vvir_z{0}_SM{1}_{2}".format(i,j,j+1)]*
                                                                bulgedom_cen_DSage["Rvir_z{0}_SM{1}_{2}".format(i,j,j+1)])

        diskdom_cen_DSage["slambda_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskdom_cen_DSage["JSD_z{0}_SM{1}_{2}".format(i,j,j+1)]/(np.sqrt(2)*diskdom_cen_DSage["Vvir_z{0}_SM{1}_{2}".format(i,j,j+1)]*
                                                                diskdom_cen_DSage["Rvir_z{0}_SM{1}_{2}".format(i,j,j+1)])


Mvir_bins = np.linspace(0, 50, 12)
n1 = 0
n2 = 0

for n1 in range(len(allgal_cen_DSage)):
    for n2 in range(4):
        n2 = n2 + 8
        bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z{0}_SM{1}_{2}'.format(n1,n2,n2+1)] = meanbin(bulgedom_cen_DSage['slambda_z{0}_SM{1}_{2}'.format(n1,n2,n2+1)], bulgedom_cen_DSage['NgalcountsR8_z{0}_SM{1}_{2}'.format(n1,n2,n2+1)])
        bulgedom_cen_DSage['slambda_Ngalcounts_stdbin_z{0}_SM{1}_{2}'.format(n1,n2,n2+1)] = stdbin(bulgedom_cen_DSage['slambda_z{0}_SM{1}_{2}'.format(n1,n2,n2+1)], bulgedom_cen_DSage['NgalcountsR8_z{0}_SM{1}_{2}'.format(n1,n2,n2+1)])
        
        diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z{0}_SM{1}_{2}'.format(n1,n2,n2+1)] = meanbin(diskdom_cen_DSage['slambda_z{0}_SM{1}_{2}'.format(n1,n2,n2+1)], diskdom_cen_DSage['NgalcountsR8_z{0}_SM{1}_{2}'.format(n1,n2,n2+1)])
        diskdom_cen_DSage['slambda_Ngalcounts_stdbin_z{0}_SM{1}_{2}'.format(n1,n2,n2+1)] = stdbin(diskdom_cen_DSage['slambda_z{0}_SM{1}_{2}'.format(n1,n2,n2+1)], diskdom_cen_DSage['NgalcountsR8_z{0}_SM{1}_{2}'.format(n1,n2,n2+1)])



plt.clf()
plt.close()
fig = plt.figure(facecolor='white', figsize=(10, 8))

plt.subplot(221)
plt.errorbar(bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z0_SM10_11'][0],
             bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z0_SM10_11'][1],
            yerr=bulgedom_cen_DSage['slambda_Ngalcounts_stdbin_z0_SM10_11'][1],
            marker='o', color='r', label='Bulge 10-11')

plt.errorbar(diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z0_SM10_11'][0],
             diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z0_SM10_11'][1],
            yerr=diskdom_cen_DSage['slambda_Ngalcounts_stdbin_z0_SM10_11'][1],
            marker='o', color='b', label='Disk 10-11')

plt.title('Stellar Mass bins at z=0')
#plt.xlim(0, 1)
#plt.ylim(-1, 21)
plt.xlabel('$\lambda_{Disk Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=3, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])

plt.subplot(222)
plt.errorbar(bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z0_SM11_12'][0],
             bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z0_SM11_12'][1],
            yerr=bulgedom_cen_DSage['slambda_Ngalcounts_stdbin_z0_SM11_12'][1],
            marker='o', color='r', label='Bulge 11-12')

plt.errorbar(diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z0_SM11_12'][0],
             diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z0_SM11_12'][1],
            yerr=diskdom_cen_DSage['slambda_Ngalcounts_stdbin_z0_SM11_12'][1],
            marker='o', color='b', label='Disk 11-12')


#plt.xlim(0, 1)
#plt.ylim(-1, 21)
plt.xlabel('$\lambda_{Disk Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=3, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])


plt.subplots_adjust(top=0.92, bottom=0.001, left=0.10, right=0.95, hspace=0.6,
                    wspace=0.35)
plt.style.use('seaborn-notebook')

plt.tight_layout()
#plt.savefig('morphdensity_Centrals_stellarSpin_lambdavsRvir_SMbin10-12_DSAGE_z0.png', dpi=100)

plt.show()


# In[117]:

plt.clf()
plt.close()
fig = plt.figure(facecolor='white', figsize=(10, 8))

plt.subplot(221)
plt.errorbar(bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z0_SM11_12'][0],
             bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z0_SM11_12'][1],
            yerr=bulgedom_cen_DSage['slambda_Ngalcounts_stdbin_z0_SM11_12'][1],
            marker='o', color='r', label='Bulge z=0')

plt.errorbar(diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z0_SM11_12'][0],
             diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z0_SM11_12'][1],
            yerr=diskdom_cen_DSage['slambda_Ngalcounts_stdbin_z0_SM11_12'][1],
            marker='o', color='b', label='Disk z=0')

plt.title('Stellar Mass bins at 11-12')
#plt.xlim(0, 1)
plt.ylim(0, 30)
plt.xlabel('$\lambda_{Disk Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=3, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])

plt.subplot(222)
plt.errorbar(bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z1_SM11_12'][0],
             bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z1_SM11_12'][1],
            yerr=bulgedom_cen_DSage['slambda_Ngalcounts_stdbin_z1_SM11_12'][1],
            marker='o', color='r', label='Bulge z=1')

plt.errorbar(diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z1_SM11_12'][0],
             diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z1_SM11_12'][1],
            yerr=diskdom_cen_DSage['slambda_Ngalcounts_stdbin_z1_SM11_12'][1],
            marker='o', color='b', label='Disk z=1')


#plt.xlim(0, 1)
plt.ylim(0, 30)
plt.xlabel('$\lambda_{Disk Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=3, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])


plt.subplot(223)
plt.errorbar(bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z2_SM11_12'][0],
             bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z2_SM11_12'][1],
            yerr=bulgedom_cen_DSage['slambda_Ngalcounts_stdbin_z2_SM11_12'][1],
            marker='o', color='r', label='Bulge z=2')

plt.errorbar(diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z2_SM11_12'][0],
             diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z2_SM11_12'][1],
            yerr=diskdom_cen_DSage['slambda_Ngalcounts_stdbin_z2_SM11_12'][1],
            marker='o', color='b', label='Disk z=2')


#plt.xlim(0, 1)
plt.ylim(0, 30)
plt.xlabel('$\lambda_{Disk Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=3, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])

plt.subplot(224)
plt.errorbar(bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z3_SM11_12'][0],
             bulgedom_cen_DSage['slambda_Ngalcounts_meanbin_z3_SM11_12'][1],
            yerr=bulgedom_cen_DSage['slambda_Ngalcounts_stdbin_z3_SM11_12'][1],
            marker='o', color='r', label='Bulge z=3')

plt.errorbar(diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z3_SM11_12'][0],
             diskdom_cen_DSage['slambda_Ngalcounts_meanbin_z3_SM11_12'][1],
            yerr=diskdom_cen_DSage['slambda_Ngalcounts_stdbin_z3_SM11_12'][1],
            marker='o', color='b', label='Disk z=3')


#plt.xlim(0, 1)
plt.ylim(0, 30)
plt.xlabel('$\lambda_{Disk Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=3, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])



plt.subplots_adjust(top=0.92, bottom=0.001, left=0.10, right=0.95, hspace=0.6,
                    wspace=0.35)
plt.style.use('seaborn-notebook')

plt.tight_layout()
#plt.savefig('morphdensity_Centrals_stellarSpin_lambdavsRvir_SMbin11-12_DSAGE_z0toz3.png', dpi=100)

plt.show()



# # # # # ## ## ## ## ## ## ## ## ## ## ## ## ## # Halo lambda # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #

i = 0
j = 0
Mvir_bins = np.linspace(0, 0.175, 10)

for i in range(len(allgal_cen_DSage)):
    for j in range(4):
        j = j + 8       
        bulgedom_cen_DSage["hlambda_z{0}_SM{1}_{2}".format(i,j,j+1)] = bulgedom_cen_DSage["haloJ_z{0}_SM{1}_{2}".format(i,j,j+1)]/(np.sqrt(2)*bulgedom_cen_DSage["Vvir_z{0}_SM{1}_{2}".format(i,j,j+1)]*
                                                                bulgedom_cen_DSage["Rvir_z{0}_SM{1}_{2}".format(i,j,j+1)])

        diskdom_cen_DSage["hlambda_z{0}_SM{1}_{2}".format(i,j,j+1)] = diskdom_cen_DSage["haloJ_z{0}_SM{1}_{2}".format(i,j,j+1)]/(np.sqrt(2)*diskdom_cen_DSage["Vvir_z{0}_SM{1}_{2}".format(i,j,j+1)]*
                                                                diskdom_cen_DSage["Rvir_z{0}_SM{1}_{2}".format(i,j,j+1)])

        bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z{0}_SM{1}_{2}'.format(i,j,j+1)] = meanbin(bulgedom_cen_DSage['hlambda_z{0}_SM{1}_{2}'.format(i,j,j+1)], bulgedom_cen_DSage['NgalcountsR8_z{0}_SM{1}_{2}'.format(i,j,j+1)])
        bulgedom_cen_DSage['hlambda_Ngalcounts_stdbin_z{0}_SM{1}_{2}'.format(i,j,j+1)] = stdbin(bulgedom_cen_DSage['hlambda_z{0}_SM{1}_{2}'.format(i,j,j+1)], bulgedom_cen_DSage['NgalcountsR8_z{0}_SM{1}_{2}'.format(i,j,j+1)])
        
        diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z{0}_SM{1}_{2}'.format(i,j,j+1)] = meanbin(diskdom_cen_DSage['hlambda_z{0}_SM{1}_{2}'.format(i,j,j+1)], diskdom_cen_DSage['NgalcountsR8_z{0}_SM{1}_{2}'.format(i,j,j+1)])
        diskdom_cen_DSage['hlambda_Ngalcounts_stdbin_z{0}_SM{1}_{2}'.format(i,j,j+1)] = stdbin(diskdom_cen_DSage['hlambda_z{0}_SM{1}_{2}'.format(i,j,j+1)], diskdom_cen_DSage['NgalcountsR8_z{0}_SM{1}_{2}'.format(i,j,j+1)])


plt.clf()
plt.close()
fig = plt.figure(facecolor='white', figsize=(10, 8))

plt.subplot(221)
plt.errorbar(bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z0_SM10_11'][0],
             bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z0_SM10_11'][1],
            yerr=bulgedom_cen_DSage['hlambda_Ngalcounts_stdbin_z0_SM10_11'][1],
            marker='o', color='r', label='Bulge 10-11')

plt.errorbar(diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z0_SM10_11'][0],
             diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z0_SM10_11'][1],
            yerr=diskdom_cen_DSage['hlambda_Ngalcounts_stdbin_z0_SM10_11'][1],
            marker='o', color='b', label='Disk 10-11')

plt.title('Stellar Mass bins at z=0')
plt.xticks([0, 0.05, 0.1, 0.15])
#plt.ylim(12, 22)
plt.xlabel('$\lambda_{Halo Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=2, numpoints=1)


plt.subplot(222)
plt.errorbar(bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z0_SM11_12'][0],
             bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z0_SM11_12'][1],
            yerr=bulgedom_cen_DSage['hlambda_Ngalcounts_stdbin_z0_SM11_12'][1],
            marker='o', color='r', label='Bulge 11-12')

plt.errorbar(diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z0_SM11_12'][0],
             diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z0_SM11_12'][1],
            yerr=diskdom_cen_DSage['hlambda_Ngalcounts_stdbin_z0_SM11_12'][1],
            marker='o', color='b', label='Disk 11-12')


plt.xticks([0, 0.05, 0.1, 0.15])
#plt.ylim(12, 22)
plt.xlabel('$\lambda_{Halo Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=2, numpoints=1)



plt.subplots_adjust(top=0.92, bottom=0.001, left=0.10, right=0.95, hspace=0.6,
                    wspace=0.35)
plt.style.use('seaborn-notebook')

plt.tight_layout()
#plt.savefig('morphdensity_Centrals_haloSpin_lambdavsRvir_SMbin10-12_DSAGE_z0.png', dpi=100)

plt.show()


# In[21]:


plt.clf()
plt.close()
fig = plt.figure(facecolor='white', figsize=(10, 8))

plt.subplot(221)
plt.errorbar(bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z0_SM10_11'][0],
             bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z0_SM10_11'][1],
            yerr=bulgedom_cen_DSage['hlambda_Ngalcounts_stdbin_z0_SM10_11'][1],
            marker='o', color='r', label='Bulge z=0')

plt.errorbar(diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z0_SM10_11'][0],
             diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z0_SM10_11'][1],
            yerr=diskdom_cen_DSage['hlambda_Ngalcounts_stdbin_z0_SM10_11'][1],
            marker='o', color='b', label='Disk z=0')

plt.title('Stellar Mass bins at 10-11')
plt.xticks([0, 0.05, 0.1, 0.15])
plt.ylim(0, 30)
plt.xlabel('$\lambda_{Halo Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=3, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])

plt.subplot(222)
plt.errorbar(bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z1_SM10_11'][0],
             bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z1_SM10_11'][1],
            yerr=bulgedom_cen_DSage['hlambda_Ngalcounts_stdbin_z1_SM10_11'][1],
            marker='o', color='r', label='Bulge z=1')

plt.errorbar(diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z1_SM10_11'][0],
             diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z1_SM10_11'][1],
            yerr=diskdom_cen_DSage['hlambda_Ngalcounts_stdbin_z1_SM10_11'][1],
            marker='o', color='b', label='Disk z=1')


plt.xticks([0, 0.05, 0.1, 0.15])
plt.ylim(0, 30)
plt.xlabel('$\lambda_{Halo Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=3, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])


plt.subplot(223)
plt.errorbar(bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z2_SM10_11'][0],
             bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z2_SM10_11'][1],
            yerr=bulgedom_cen_DSage['hlambda_Ngalcounts_stdbin_z2_SM10_11'][1],
            marker='o', color='r', label='Bulge z=2')

plt.errorbar(diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z2_SM10_11'][0],
             diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z2_SM10_11'][1],
            yerr=diskdom_cen_DSage['hlambda_Ngalcounts_stdbin_z2_SM10_11'][1],
            marker='o', color='b', label='Disk z=2')


plt.xticks([0, 0.05, 0.1, 0.15])
plt.ylim(0, 30)
plt.xlabel('$\lambda_{Halo Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=3, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])

plt.subplot(224)
plt.errorbar(bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z3_SM10_11'][0],
             bulgedom_cen_DSage['hlambda_Ngalcounts_meanbin_z3_SM10_11'][1],
            yerr=bulgedom_cen_DSage['hlambda_Ngalcounts_stdbin_z3_SM10_11'][1],
            marker='o', color='r', label='Bulge z=3')

plt.errorbar(diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z3_SM10_11'][0],
             diskdom_cen_DSage['hlambda_Ngalcounts_meanbin_z3_SM10_11'][1],
            yerr=diskdom_cen_DSage['hlambda_Ngalcounts_stdbin_z3_SM10_11'][1],
            marker='o', color='b', label='Disk z=3')


plt.xticks([0, 0.05, 0.1, 0.15])
plt.ylim(0, 30)
plt.xlabel('$\lambda_{Halo Bullock}$',  fontsize=10)
plt.ylabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=3, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])



plt.subplots_adjust(top=0.92, bottom=0.001, left=0.10, right=0.95, hspace=0.6,
                    wspace=0.35)
plt.style.use('seaborn-notebook')

plt.tight_layout()
#plt.savefig('morphdensity_Centrals_haloSpin_lambdavsRvir_SMbin11-12_DSAGE_z0toz3.png', dpi=100)

plt.show()











from astropy.io import fits
from scipy import stats
from scipy import spatial as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
#get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set_style('darkgrid')


########################## Loading fits data ##########################

load_z0DSage = fits.open('tao.1951.0.fits')
load_z1DSage = fits.open('tao.1959.0.fits')
load_z2DSage= fits.open('tao.1961.0.fits')
load_z3DSage = fits.open('tao.1957.0.fits')

data_z0DSage = load_z0DSage[1].data
data_z1DSage = load_z1DSage[1].data
data_z2DSage = load_z2DSage[1].data
data_z3DSage = load_z3DSage[1].data


########################### Dividing sample between centrals and satellites for DARK SAGE #######################################
########################### Dividing sample between centrals and satellites for DARK SAGE #######################################
########################### Dividing sample between centrals and satellites for DARK SAGE #######################################

idx_cen_z0DSage = np.where(data_z0DSage[:]['Galaxy_Classification'] == 0) [0]
idx_sat_z0DSage  = np.where(data_z0DSage[:]['Galaxy_Classification'] == 1) [0]
gal_cen_z0DSage = data_z0DSage[idx_cen_z0DSage]
gal_sat_z0DSage = data_z0DSage[idx_sat_z0DSage]

idx_cen_z1DSage = np.where(data_z1DSage[:]['Galaxy_Classification'] == 0) [0]
idx_sat_z1DSage  = np.where(data_z1DSage[:]['Galaxy_Classification'] == 1) [0]
gal_cen_z1DSage = data_z1DSage[idx_cen_z1DSage]
gal_sat_z1DSage = data_z1DSage[idx_sat_z1DSage]

idx_cen_z2DSage = np.where(data_z2DSage[:]['Galaxy_Classification'] == 0) [0]
idx_sat_z2DSage  = np.where(data_z2DSage[:]['Galaxy_Classification'] == 1) [0]
gal_cen_z2DSage = data_z2DSage[idx_cen_z2DSage]
gal_sat_z2DSage = data_z2DSage[idx_sat_z2DSage]

idx_cen_z3DSage = np.where(data_z3DSage[:]['Galaxy_Classification'] == 0) [0]
idx_sat_z3DSage  = np.where(data_z3DSage[:]['Galaxy_Classification'] == 1) [0]
gal_cen_z3DSage = data_z3DSage[idx_cen_z3DSage]
gal_sat_z3DSage = data_z3DSage[idx_sat_z3DSage]


########################## Taking all galaxies with TSM = 0 out of sample ############################################# 
########################## Taking all galaxies with TSM = 0 out of sample ############################################# 

TSM_idx_cen_z0DSage = np.where(gal_cen_z0DSage[:]['Total_Stellar_Mass']!=0)[0]
gal_cen_z0DSage = gal_cen_z0DSage[TSM_idx_cen_z0DSage]

TSM_idx_cen_z1DSage = np.where(gal_cen_z1DSage[:]['Total_Stellar_Mass']!=0)[0]
gal_cen_z1DSage = gal_cen_z1DSage[TSM_idx_cen_z1DSage]

TSM_idx_cen_z2DSage = np.where(gal_cen_z2DSage[:]['Total_Stellar_Mass']!=0)[0]
gal_cen_z2DSage = gal_cen_z2DSage[TSM_idx_cen_z2DSage]

TSM_idx_cen_z3DSage = np.where(gal_cen_z3DSage[:]['Total_Stellar_Mass']!=0)[0]
gal_cen_z3DSage = gal_cen_z3DSage[TSM_idx_cen_z3DSage]


########################## Counting how many galaxies are there in a spherical volume with radius R = 8 ##########################
########################## Counting how many galaxies are there in a spherical volume with radius R = 8 ##########################
########################## Counting how many galaxies are there in a spherical volume with radius R = 8 ##########################

gal_cen_z0 = {}
gal_cen_z1 = {}
gal_cen_z2 = {}
gal_cen_z3 = {}
allgal_cen_DSage = [gal_cen_z0DSage, gal_cen_z1DSage, gal_cen_z2DSage, gal_cen_z3DSage]
gal_dict = [gal_cen_z0, gal_cen_z1, gal_cen_z2, gal_cen_z3]

i=0
for i in range(len(allgal_cen_DSage)):
    gal_cen["keys_z{0}".format(i)] = allgal_cen_DSage[i].dtype.names
    key=0
    for key in gal_cen['keys_z{0}'.format(i)]:
        gal_dict[i][key] = list(allgal_cen_DSage[i][key])

    gal_cen["total_z{0}".format(i)] = pd.DataFrame(gal_dict[i])

    gal_cen["points_z{0}".format(i)] = np.array(zip(gal_cen['total_z{0}'.format(i)]['X'].ravel(),
                                                     gal_cen['total_z{0}'.format(i)]['Y'].ravel(),
                                                     gal_cen['total_z{0}'.format(i)]['Z'].ravel()))
    
    gal_cen["tree_z{0}".format(i)] = sp.cKDTree(gal_cen["points_z{0}".format(i)])

    gal_cen["ck_arr_8_z{0}".format(i)] = gal_cen["tree_z{0}".format(i)].query_ball_point(gal_cen["points_z{0}".format(i)], 8)

    ii=0
    gal_cen["ngal_arr_8_z{0}".format(i)] = np.array([len(gal_cen["ck_arr_8_z{0}".format(i)][ii])-1 for ii in range(len(gal_cen["ck_arr_8_z{0}".format(i)]))]).astype(float)
    
    #allgal_cen_DSage[i]['Ngal_counts_R8'] = gal_cen['ngal_arr_8_z{0}'.format(i)]
    gal_cen["keys_z{0}".format(i)] = gal_cen["ngal_arr_8_z{0}".format(i)].dtype.names
    gal_cen["g_dict_z{0}".format(i)] = list(gal_cen["ngal_arr_8_z{0}".format(i)])
    gal_cen["ngal_tot_z{0}".format(i)] = pd.DataFrame(gal_cen["g_dict_z{0}".format(i)], columns = ['Ngal_counts_R8'])
    gal_cen['tot_z{0}'.format(i)] = pd.concat([gal_cen["total_z{0}".format(i)], gal_cen["ngal_tot_z{0}".format(i)]], axis=1)


############################################ Bulge and Disk dom for redshift 0 to 3 #################################################################
############################################ Bulge and Disk dom for redshift 0 to 3 #################################################################
############################################ Bulge and Disk dom for redshift 0 to 3 #################################################################

bulgedom_cen_DSage = {}
diskdom_cen_DSage = {}
i=0
for i in range(len(allgal_cen_DSage)):
    gal_cen["morph_z{0}".format(i)] = gal_cen['tot_z{0}'.format(i)]['Disk_Stellar_Mass']/gal_cen['tot_z{0}'.format(i)]['Total_Stellar_Mass']
    bulgedom_cen_DSage["idx_z{0}".format(i)] = np.where(gal_cen['morph_z{0}'.format(i)] < 0.5)[0]
    bulgedom_cen_DSage["total_z{0}".format(i)] = gal_cen['tot_z{0}'.format(i)].loc[bulgedom_cen_DSage['idx_z{0}'.format(i)],:]

    diskdom_cen_DSage["idx_z{0}".format(i)] = np.where(gal_cen['morph_z{0}'.format(i)] > 0.5)[0]
    diskdom_cen_DSage["total_z{0}".format(i)] = gal_cen['tot_z{0}'.format(i)].loc[diskdom_cen_DSage['idx_z{0}'.format(i)],:]


failval = np.nan
# failval = -999
mean_func = np.nanmean
std_func = np.nanstd

############################################ This function bins my data #################################################################


############################################ meanbin takes the mean for each bin #################################################################
def meanbin(a, b):
    idx_Mvir = np.digitize(a, Mvir_bins)    ## Mvir_bins is the amounts of binning and their length 
    count = len(np.where(b > 0.5)[0])
    mean_a = np.array([ mean_func(a[idx_Mvir==ii]) if len(a[idx_Mvir==ii]) > 0 else failval for ii in range(1, len(Mvir_bins))])
    mean_b = np.array([ len(b[idx_Mvir==ii]) if len(a[idx_Mvir==ii]) > 0 else failval for ii in range(1, len(Mvir_bins))])
    ratio = count/mean_b
    return mean_a, ratio



############################################ stdbin takes the standard deviation of the mean #################################################################
def stdbin(a, b):
    idx_Mvir = np.digitize(a, Mvir_bins)
    std_a = np.array([ std_func(a[idx_Mvir==ii]) if len(a[idx_Mvir==ii]) > 0 else failval for ii in range(1, len(Mvir_bins))])
    std_b = np.array([ std_func(b[idx_Mvir==ii])/np.sqrt(len(b[idx_Mvir==ii])) if len(a[idx_Mvir==ii]) > 0 else failval for ii in range(1, len(Mvir_bins))])
    #std_meanb = np.array(std_b/np.sqrt(len(b[idx_Mvir==k])) for k in range(1, len(Mvir_bins)))
    return std_a, std_b


morph = gal_cen['total_z0']['Disk_Stellar_Mass']/gal_cen['total_z0']['Total_Stellar_Mass']
Mvir_bins = np.linspace(0, 80, 10)
diskdom_cen_DSage['Ngalcounts_ratios_meanbin_z0'] = meanbin(gal_cen['tot_z0']['Ngal_counts_R8'], morph)
diskdom_cen_DSage['Ngalcounts_ratios_stdbin_z0'] = stdbin(gal_cen['tot_z0']['Ngal_counts_R8'], morph)





plt.clf()
plt.close()
fig = plt.figure(facecolor='white', figsize=(10, 8))

plt.errorbar(diskdom_cen_DSage['Ngalcounts_ratios_meanbin_z0'][0],
             diskdom_cen_DSage['Ngalcounts_ratios_meanbin_z0'][1],
            yerr=diskdom_cen_DSage['Ngalcounts_ratios_stdbin_z0'][1],
            marker='o', color='g', label='z=0')

# plt.errorbar(bulgedom_cen_DSage['Ngalcounts_ratios_meanbin_z0'][0],
#              bulgedom_cen_DSage['Ngalcounts_ratios_meanbin_z0'][1],
#             yerr=bulgedom_cen_DSage['Ngalcounts_ratios_stdbin_z0'][1],
#             marker='o', color='g', label='z=0')

plt.title('Centrals at z=0')
#plt.xlim(0, 1)
#plt.ylim(-1, 21)
plt.xlabel('Ngal Counts for R=8Mpc/h',  fontsize=10)
plt.ylabel('Ratio of Disk-dom galaxies to total galaxies inside each bin',  fontsize=10)
#plt.title('Central Galaxies Mvir vs. Stellar Mass ')
plt.legend(loc=2, numpoints=1)
#plt.axis([1e11, 1e15, 1e9, 1e12])


plt.style.use('seaborn-notebook')

#plt.tight_layout()
#plt.savefig('morphdensity_Centrals_stellarSpin_lambdavsRvir_SMbin10-12_DSAGE_z0.png', dpi=100)

plt.show()



