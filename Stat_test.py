import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline

print('Stat_test.py called')
print('')
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
print('Data has been extracted from https://arxiv.org/abs/1710.09126')
print(' ')


# II) Load Events data and perform chi2 test
K_ev, Pi_ev, ct_pts, cos_pts= np.transpose(np.array(pd.read_csv('results/Events_SK.csv' )))
df_ev=pd.read_csv('results/Events_SK.csv' )

MN=0.1
CT_REST=np.unique(ct_pts)
cos_value=np.unique(cos_pts)

Evts_Kp=[]
Evts_Pip=[]

for ict, CTAU in enumerate(CT_REST):

    events_Kp_aux=[]
    events_Pip_aux=[]
    for i, cos in enumerate(cos_value):
        cos_val=cos
        ct_val=CTAU

        df_tmp = df_ev[df_ev.cosine == cos_val][df_ev[df_ev.cosine == cos_val].ctau == ct_val]
        
        evtK_val=np.array(df_tmp.EvtKaon)
        evtPi_val=np.array(df_tmp.EvtPion)
        #print(evtPi_val)
        events_Kp_aux.append(evtK_val[0])
        events_Pip_aux.append(evtPi_val[0])
    
    Evts_Kp.append(events_Kp_aux)
    Evts_Pip.append(events_Pip_aux)
    
def Chi_sq_bin(s,b,d):
    return s + b -d + d*np.log(d/(s+b))

def bivariate_interpolation_surf(df, surface= 'log_top_sfc', cos_th='costheta', E='logE'): 
    df_copy = df.copy().sort_values(by=[cos_th, E], ascending=True)
    x = np.array(df_copy[E].unique(), dtype='float64')
    y = np.array(df_copy[cos_th].unique(), dtype='float64')
    z = np.array(df_copy[surface].values, dtype='float64')
    Z = z.reshape(len(y), len(x))
    interp_spline = interpolate.RectBivariateSpline(y, x, Z, kx=1, ky=1)
    return interp_spline

BR=np.logspace(-11,-3,40)
evts_array_Kp=[]
evts_array_Pip=[]

for i, br in enumerate(BR):

    evts_br_Kp=[]
    evts_br_Pip=[]
    for j,  CTAU_N_REST in enumerate(CT_REST):

        evts_br_aux_Kp=[]
        evts_br_aux_Pip=[]
        
        for e, elements in enumerate(Evts_Kp[0]):

            evts_br_aux_Kp.append(BR[i]*Evts_Kp[j][e])
            evts_br_aux_Pip.append(BR[i]*Evts_Pip[j][e])
            
        evts_br_Kp.append(evts_br_aux_Kp)
        evts_br_Pip.append(evts_br_aux_Pip)
        
    evts_array_Kp.append(evts_br_Kp)
    evts_array_Pip.append(evts_br_Pip)

#Chi2 test    
data=[]
for i, br in enumerate(BR):
    for j, CTAU_N_REST in enumerate(CT_REST):
        Signal_CC_e_Kp= evts_array_Kp[i][j]
        suma_CC_e_Kp=0
        Signal_CC_e_Pip= evts_array_Pip[i][j]
        suma_CC_e_Pip=0
        
        
        for K, element in enumerate(Data):
            suma_CC_e_Kp = suma_CC_e_Kp + Chi_sq_bin(Signal_CC_e_Kp[K],Background[K],Data[K])
            suma_CC_e_Pip = suma_CC_e_Pip + Chi_sq_bin(Signal_CC_e_Pip[K],Background[K],Data[K])
        
        data.append([MN, br,CTAU_N_REST,2*suma_CC_e_Kp,2*suma_CC_e_Pip])

np.savetxt("results/SK_K_Pi_uncorrelated_chi2_mN_{:.2f}.txt".format(MN),data)    
print(' ')
print('chi2 test data saved !')
print(' ')


M_N,BR,CTAUN,CHI2_K,CHI2_Pi = np.transpose(np.loadtxt("results/SK_K_Pi_uncorrelated_chi2_mN_{:.2f}.txt".format(MN) ))
print('min and max chi2 value found: ')
print(min(CHI2_Pi),max(CHI2_Pi))

chi_square_CL90 = 14.69 # This is for 9 parameters 
my_df_K = pd.DataFrame.from_dict({'costheta':BR, 'logE':CTAUN, 'log_top_sfc': CHI2_K})
my_df_Pi = pd.DataFrame.from_dict({'costheta':BR, 'logE':CTAUN, 'log_top_sfc': CHI2_Pi})

interpolador_K= bivariate_interpolation_surf(my_df_K)
interpolador_Pi= bivariate_interpolation_surf(my_df_Pi)

def chi2_func_K(br,ct):
    return interpolador_K(br, ct, grid=False)

def chi2_func_Pi(br,ct):
    return interpolador_Pi(br, ct, grid=False)

FunVec_K=np.vectorize(chi2_func_K)
FunVec_Pi=np.vectorize(chi2_func_Pi)

br_value=np.unique(BR)
ctauN_value=np.unique(CTAUN)
X, Y = np.meshgrid(ctauN_value, br_value)
Z_K= FunVec_K(Y,X)
Z_Pi= FunVec_Pi(Y,X)

# FINAL PLOT:
print(' ')
print('Plotting Excluded Region...')
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'cm'
plt.rcParams['mathtext.it'] = 'cm:italic'
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'DejaVu Sans','serif':['Computer Modern Roman']}) 

fig,ax=plt.subplots(figsize=(9,6))
CS_K =plt.contour(X, Y , Z_K, levels=[chi_square_CL90], colors='steelblue')
CS_Pi =plt.contour(X, Y , Z_Pi, levels=[chi_square_CL90], colors='hotpink')

datK= CS_K.allsegs[0][0]   # sacar puntos
datPi=CS_Pi.allsegs[0][0]

ct_K,ct_Pi=datK[:,0],datPi[:,0]
br_K,br_Pi=datK[:,1],datPi[:,1]

plt.plot(ct_K,br_K,color='steelblue')
plt.plot(ct_Pi,br_Pi,color='hotpink')

plt.fill_between(ct_K, 1e-3,br_K,color='steelblue', alpha=0.8,label='K')
plt.fill_between(ct_Pi,1e-3,br_Pi,color='hotpink', alpha=0.3,label='$\pi$')

leg = plt.legend(loc = 'lower left' ,fontsize=17)
leg.get_frame().set_edgecolor('white')

plt.ylim([1e-11,1e-3])
plt.xlim([1e-5,1e5])

plt.yscale("log")
plt.xscale("log")

plt.ylabel('Branching Fraction', fontdict={'size': 20})
plt.xlabel(r'$ c\tau \,\,$'+'  [km]', fontdict={'size': 20})

plt.minorticks_on()
plt.tick_params(which='both',direction='in',axis='both',right=True, top=True, width=1) #width=2 make them sees as bold
plt.tick_params(which='major',length=7)
plt.xticks(fontweight='light',fontsize=17)
plt.yticks(fontweight='light',fontsize=17)

plt.tight_layout()
plt.show();

print('')
print('Part of this code was used in https://arxiv.org/abs/1911.09129')