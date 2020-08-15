import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline

# I) Load SK data:

# measured 
CO_multiring_nue,EV_multiring_nue= np.transpose(np.array(pd.read_csv('SK_data/multiring_multigev_nue.csv' )))
CO_multiring_nuebar,EV_multiring_nuebar= np.transpose(np.array(pd.read_csv('SK_data/multiring_multigev_nuebar.csv' )))
CO_singlering_nue,EV_singlering_nue= np.transpose(np.array(pd.read_csv('SK_data/singlering_multigev_nue.csv' )))
CO_singlering_nuebar,EV_singlering_nuebar= np.transpose(np.array(pd.read_csv('SK_data/singlering_multigev_nuebar.csv' )))
# expected 
CO_multiring_nue_pre,EV_multiring_nue_pre= np.transpose(np.array(pd.read_csv('SK_data/multiring_multigev_nue_prediction.csv' )))
CO_multiring_nuebar_pre,EV_multiring_nuebar_pre= np.transpose(np.array(pd.read_csv('SK_data/multiring_multigev_nuebar_prediction.csv' )))
CO_singlering_nue_pre,EV_singlering_nue_pre= np.transpose(np.array(pd.read_csv('SK_data/singlering_multigev_nue_prediction.csv' )))
CO_singlering_nuebar_pre,EV_singlering_nuebar_pre= np.transpose(np.array(pd.read_csv('SK_data/singlering_multigev_nuebar_prediction.csv' )))

measured=[]
for k, elements in enumerate(EV_singlering_nuebar):
    suma_m=EV_singlering_nuebar[k]+EV_singlering_nue[k]+EV_multiring_nuebar[k]+EV_multiring_nue[k]
    measured.append(round(suma_m,2))

predicted=[]
for k, elements in enumerate(EV_singlering_nuebar_pre):
    suma_p=EV_singlering_nuebar_pre[k]+EV_singlering_nue_pre[k]+EV_multiring_nuebar_pre[k]+EV_multiring_nue_pre[k]
    predicted.append(round(suma_p,2))

Background=predicted
Data=measured

print('Total number of measured events (for an arrival direction with increasing cosine angle and bin size equal to 0.2) in SK data sample :')
print(Data)
print(' ')
print('Total number of background events for this search :')
print(Background)
print(' ')
print('Data has been extracted from arXiv:1710.09126v2')
print(' ')

# II) Decay effective area:

# function created to interpolate in 2d
def bivariate_interpolation_surf(df, surface= 'log_top_sfc', cos_th='costheta', E='logE'): 
    df_copy = df.copy().sort_values(by=[cos_th, E], ascending=True)
    x = np.array(df_copy[E].unique(), dtype='float64')
    y = np.array(df_copy[cos_th].unique(), dtype='float64')
    z = np.array(df_copy[surface].values, dtype='float64')

    Z = z.reshape(len(y), len(x))

    interp_spline = interpolate.RectBivariateSpline(y, x, Z, kx=1, ky=1)
    return interp_spline

# Load decay effective area data (generated with 'Aeff.py' script )
A_sup_km2, A_lat_km2, cos_array, ctaulab_array, area_total_cm2 = np.transpose(np.loadtxt("SK_data/expanded_surface_data_SK.txt"))
df_surface_SK = pd.DataFrame.from_dict({'costheta':cos_array, 'logE':ctaulab_array, 'log_top_sfc': area_total_cm2})
interpolador_area= bivariate_interpolation_surf(df_surface_SK)

# Interpolated function
def Seff_SK(cosine,ct):
    return float(interpolador_area(cosine, ct, grid=False))

# Define ctau in laboratory frame
def ctau_lab_func(ct_rest,m,E):
    return ct_rest*(E/m)

print('The relevant functions of this script are : ctau_lab_func(ctau_rest,m,E) and Seff_SK(cosine,ctau) ')
print('Use ctau in [km] ')
print('Seff_SK returns a value in [cm2] ')