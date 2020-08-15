import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
print('Functions.py called')
print('')
# Constants used
CM_to_GeVM1=5.063e+13

mKp = 0.4937 #GeV
mPip= 0.13957 #GeV
me = 0.511e-3 #GeV 
mL = me #GeV for electron # for muon mL = 105.65e-3 #GeV

ctaurest_Kp= (1.24e-08)*(1.52e+24)
ctaurest_Pip= (2.6e-08)*(1.52e+24) #GeV-1

eficiencia=0.75
deltaT=(5326)*24*60*60
fn= eficiencia*2*np.pi*deltaT

my_dict_integral={"epsabs": 0.00, "epsrel" : 1e-2, "limit" : 300} 

# Integration limits (Ep):
def lim_sup_Ep(mM,mN,ml,EN):
    return ( 1/(2 *mN**2) )*(EN *(-ml**2+mM**2+mN**2)+np.sqrt( (EN-mN)*(EN+mN)*(-ml-mM+mN)*(ml-mM+mN)*(-ml+mM+mN)*(ml+mM+mN)) )

def lim_inf_Ep(mM,mN,ml,EN):
    return ( 1/(2 *mN**2) )*(EN *(-ml**2+mM**2+mN**2)-np.sqrt( (EN-mN)*(EN+mN)*(-ml-mM+mN)*(ml-mM+mN)*(-ml+mM+mN)*(ml+mM+mN)) )


# Interpolation Functions
def ThreeD_interpolation(df, X3= 'X3', X2='X2', X1='X1' , FX= 'FX'): 
    df_copy = df.copy().sort_values(by=[X1, X2, X3], ascending=True)
    x = np.array(df_copy[X1].unique(), dtype='float64')
    y = np.array(df_copy[X2].unique(), dtype='float64')
    z = np.array(df_copy[X3].unique(), dtype='float64')
    a = np.array(df_copy[FX].values, dtype='float64')
    A = a.reshape(len(x), len(y), len(z))
    RegularGrid = RegularGridInterpolator((x,y,z), A , bounds_error=False, fill_value=0)
    return RegularGrid

def FourD_interpolation(df, X3= 'X3', X2='X2', X1='X1', X4='X4' , FX= 'FX'): 
    df_copy = df.copy().sort_values(by=[X1, X2, X3, X4], ascending=True)
    x = np.array(df_copy[X1].unique(), dtype='float64')
    y = np.array(df_copy[X2].unique(), dtype='float64')
    z = np.array(df_copy[X3].unique(), dtype='float64')
    w = np.array(df_copy[X4].unique(), dtype='float64')
    a = np.array(df_copy[FX].values, dtype='float64')
    A = a.reshape(len(x), len(y), len(z),len(w))
    RegularGrid = RegularGridInterpolator((x,y,z,w), A , bounds_error=False, fill_value=0)
    return RegularGrid

# Functions used for HNL production
def beta(E,m):
    if E < m:
        raise ValueError('E should be > m !')
    else:
        return np.sqrt(E**2-m**2)/E
    
def gamma(E,m):
    return E/m

def dec_length_Kp(E): # GeV-1
        return gamma(E,mKp)*beta(E,mKp)*ctaurest_Kp
    
def dec_length_Pip(E): # GeV-1
        return gamma(E,mPip)*beta(E,mPip)*ctaurest_Pip
    
def lambdaFunc(mp,md1,md2): # no dim
    return  np.sqrt(1 + (md1**4)/(mp**4) + (md2**4)/(mp**4) - (2*md1**2)/(mp**2) - (2*md2**2)/(mp**2) - (2*md2**2*md1**2)/(mp**4) ) 

def sqrMom_P(E,mp): # GeV
    if E<mp:
        raise ValueError('E should be > m ! ')
    else: 
         return np.sqrt( (E**2 - mp**2) )

def twobody_dist(Ep,mp,md1,md2):
    return 1/(sqrMom_P(Ep,mp)*lambdaFunc(mp,md1,md2))