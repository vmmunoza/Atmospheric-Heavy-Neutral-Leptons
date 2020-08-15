import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'cm'
plt.rcParams['mathtext.it'] = 'cm:italic'
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'DejaVu Sans','serif':['Computer Modern Roman']}) 

print('Mesons.py called')
print('')
# Load Meson flux data extracted from MCEq
h_d,l_d,COSINE,E_d,flux_Pip,flux_Pim,flux_Kp,flux_Km,rho_d,X_d = np.transpose(np.loadtxt("SK_data/SK_FLUXES.txt" ))

# Define function that returns the meson fluxes as function of slant depth and energy 
def FLUJO_X_E(string):

    phi0_Pip=funciones_Pip[string]
    phi0_Pim=funciones_Pim[string]
    phi0_Kp=funciones_Kp[string]
    phi0_Km=funciones_Km[string]
    
    
    X0=slantes[string]
    E0= energias[string]
    rho0=densidades[string]
    l0=distancias[string]
    h0=alturas[string]
    
    my_df_Pip = pd.DataFrame.from_dict({'X':X0, 'E':E0, 'F': phi0_Pip})
    my_df_Pim = pd.DataFrame.from_dict({'X':X0, 'E':E0, 'F': phi0_Pim})
    my_df_Kp = pd.DataFrame.from_dict({'X':X0, 'E':E0, 'F': phi0_Kp})
    my_df_Km = pd.DataFrame.from_dict({'X':X0, 'E':E0, 'F': phi0_Km})

    
    def bivariate_interpolation_D(df, flujo= 'F', slant='X', E='E'): 
        df_copy = df.copy().sort_values(by=[slant, E], ascending=True)
        x = np.array(df_copy[E].unique(), dtype='float64')
        y = np.array(df_copy[slant].unique(), dtype='float64')
        z = np.array(df_copy[flujo].values, dtype='float64')
        Z = z.reshape(len(y), len(x))

        interp_spline_D = interpolate.RectBivariateSpline(y, x, Z, kx=1, ky=1)
        return interp_spline_D


    interpolador_Pip= bivariate_interpolation_D(my_df_Pip)
    interpolador_Pim= bivariate_interpolation_D(my_df_Pim)
    interpolador_Kp= bivariate_interpolation_D(my_df_Kp)
    interpolador_Km= bivariate_interpolation_D(my_df_Km)
    
    def dFlux_Pip(x,e):
        return interpolador_Pip(x, e, grid=False)
    
    def dFlux_Pim(x,e):
        return interpolador_Pim(x, e, grid=False)

    def dFlux_Kp(x,e):
        return interpolador_Kp(x, e, grid=False)
    
    def dFlux_Km(x,e):
        return interpolador_Km(x, e, grid=False)
    
    return(dFlux_Pip,dFlux_Pim,dFlux_Kp,dFlux_Km)

# Pre-processing routine
def l_func(h,cosine):
    Rt=6371
    return -Rt*cosine + np.sqrt(Rt**2*cosine**2 + h**2 +2*Rt*h)

cos_unique = np.unique(COSINE)
E_vec=np.unique(E_d)

funciones_Pip=[]
funciones_Pim=[]
funciones_Kp=[]
funciones_Km=[]
distancias=[]
slantes=[]
energias=[]
alturas=[]
densidades=[]
for k, co in enumerate(cos_unique):

    array_Pip=[]
    array_Pim=[]
    array_Kp=[]
    array_Km=[]
    
    array1=[]
    array2=[]
    array3=[]
    array4=[]
    array5=[]
    array6=[]
    for i, cos in enumerate(COSINE):

        if COSINE[i] == cos_unique[k]:
            
            array_Pip.append(flux_Pip[i] )
            array_Pim.append(flux_Pim[i] )
            array_Kp.append(flux_Kp[i] )
            array_Km.append(flux_Km[i] )
            
            array2.append(l_func(h_d[i], COSINE[i]))
            array3.append(X_d[i])
            array4.append(E_d[i])
            array5.append(h_d[i])
            array6.append(rho_d[i])
            
    funciones_Pip.append(array_Pip)
    funciones_Pim.append(array_Pim)
    funciones_Kp.append(array_Kp)
    funciones_Km.append(array_Km)
        
    distancias.append(array2)
    slantes.append(array3)
    energias.append(array4)
    alturas.append(array5)
    densidades.append(array6)


func_Pip =[]
func_Pim =[]
func_Kp =[]
func_Km =[]
l_list=[]
slant_list=[]
rho_list=[]
h_list=[]

for ic, cos in enumerate(cos_unique):
    func_Pip.append(FLUJO_X_E(ic)[0])
    func_Pim.append(FLUJO_X_E(ic)[1])
    func_Kp.append(FLUJO_X_E(ic)[2])
    func_Km.append(FLUJO_X_E(ic)[3])
    int_h=interpolate.interp1d(slantes[ic], alturas[ic])
    h_list.append(int_h)
    int_rho_D=interpolate.interp1d(slantes[ic], densidades[ic])
    rho_list.append(int_rho_D)
    int_l_D=interpolate.interp1d(slantes[ic], distancias[ic])
    l_list.append( int_l_D)

print('cosine values: ' , cos_unique)
print(' ')
print('Fluxes of mesons are stored with an index idc for each cosine value, and are given as functions of the slant depth X and the energy E. The fluxes of Pi+,Pi-,K+,K- are: ')
print(' ')
print('func_Pip[idc](X,E),func_Pim[idc](X,E),func_Kp[idc](X,E),func_Km[idc](X,E)')
print(' ')
print('Energy grid stored in E_vec')
print(' ')
print('Original data extracted from MCEq software: https://mceq.readthedocs.io/en/latest/')
print(' ')
print('Plotting Meson flux...')

# PLOT FLUX AT FIXED HEIGHT AND COSINE
fig,ax=plt.subplots(figsize=(9,6))

plt.plot(E_vec,E_vec**3*func_Pip[-1](121,E_vec),color='purple',label=r"$\pi^{+}$")
plt.plot(E_vec,E_vec**3*func_Pim[-1](121,E_vec),color='hotpink',label=r"$\pi^{-}$")
plt.plot(E_vec,E_vec**3*func_Kp[-1](121,E_vec),color='navy',label=r"$K^{+}$")
plt.plot(E_vec,E_vec**3*func_Km[-1](121,E_vec),color='steelblue',label=r"$K^{-}$")

plt.xlabel("E [GeV]", fontdict={'size': 20})
plt.ylabel(r"$ \textrm{E}^{3}\,\,\frac{d\Phi}{dE\,d\Omega}\,\,\left[\textrm{GeV}^{2}\,\,\textrm{sr}^{-1}\,\,\textrm{s}^{-1}\,\,\textrm{cm}^{-2} \,\right]$" , fontdict={'size': 20})

leg = plt.legend(loc = 'best' ,fontsize=17)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(2.0)
plt.title(" h = 15.4 [km],  " + r"$ \,\cos{\theta} =$"+str(cos_unique[-1])  ,fontsize=17)

plt.xlim([1e1, 1e11])
plt.ylim([1e-8, 1e1])
plt.yscale("log")
plt.xscale("log")

plt.minorticks_on()
plt.tick_params(which='both',direction='in',axis='both',right=True, top=True, width=2.0) 
plt.tick_params(which='major',length=7)
plt.xticks(fontweight='light',fontsize=17)
plt.yticks(fontweight='light',fontsize=17)

plt.tight_layout()
#plt.savefig('flux_mesons_MCEq.pdf', format='pdf')
plt.show();