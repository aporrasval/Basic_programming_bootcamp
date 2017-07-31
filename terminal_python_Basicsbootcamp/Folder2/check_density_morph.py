
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



