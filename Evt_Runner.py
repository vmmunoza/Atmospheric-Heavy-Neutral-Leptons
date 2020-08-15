from Functions import *
from Mesons import *
from scipy import integrate

print('Evt_Runner.py called')
print('')
print('IMPORTANT NOTE: This version of the code is very slow and can be easily optimized, nevertheless it is used since it is good for test/debugging purpose. ')

# Decay effective area:
A_sup_km2, A_lat_km2, cos_array, ctaulab_array, area_total_cm2 = np.transpose(np.loadtxt("SK_data/expanded_surface_data_SK.txt"))

def ctau_lab_func(ct_rest,m,E):
    return ct_rest*(E/m)

def bivariate_interpolation_surf(df, surface= 'log_top_sfc', cos_th='costheta', E='logE'): 
    df_copy = df.copy().sort_values(by=[cos_th, E], ascending=True)
    x = np.array(df_copy[E].unique(), dtype='float64')
    y = np.array(df_copy[cos_th].unique(), dtype='float64')
    z = np.array(df_copy[surface].values, dtype='float64')
    Z = z.reshape(len(y), len(x))
    interp_spline = interpolate.RectBivariateSpline(y, x, Z, kx=1, ky=1)
    return interp_spline

df_surface_SK = pd.DataFrame.from_dict({'costheta':cos_array, 'logE':ctaulab_array, 'log_top_sfc': area_total_cm2})
interpolador_area= bivariate_interpolation_surf(df_surface_SK)

def Seff_SK(coseno,ct):
    return interpolador_area(coseno, ct, grid=False)

# Function to integrate:
def Pip_integrando_inX(x,Ep,Ed,ctau_r,mN,idxc):
    return CM_to_GeVM1*np.exp(-(l_list[idxc](x)*mN)/(ctau_r*Ed)) * (func_Pip[idxc](x,Ep)+func_Pim[idxc](x,Ep) ) *twobody_dist(Ep,mPip,mN,mL) / (rho_list[idxc](x)*dec_length_Pip(Ep)) 

def Kp_integrando_inX(x,Ep,Ed,ctau_r,mN,idxc):
    return CM_to_GeVM1*np.exp(-(l_list[idxc](x)*mN)/(ctau_r*Ed)) * (func_Kp[idxc](x,Ep)+func_Km[idxc](x,Ep) ) *twobody_dist(Ep,mKp,mN,mL) / (rho_list[idxc](x)*dec_length_Kp(Ep)) 

# lists used
Evec_parent=np.unique(E_d)[0:17]
Evec_daughter=np.unique(E_d)[0:16]
cos_vec=np.linspace(-0.9,0.9,10)
cos_value=[-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9]
cos_min=[]
cos_max=[]

for cos in cos_value:
    cos_min.append( round(cos-0.199,2) )
    cos_max.append(cos)

################################################# SET MASS AND CTAU VALUES ###################################################
print('')
print('production of HNL will be computed for a mass of: ')
MN=0.1  
print(MN,' GeV')

CT_N_REST=np.logspace(-4,5,2)    
print('')
print('ctau values spanned: ')
print(CT_N_REST, ' km')
print('')
##############################################################################################################################

# 1st integration
K_points=[]
Pi_points=[]
Ep_points=[]
Ed_points=[]
ctau_points=[]
cos_points=[]
data_1=[]
print('Integration in X started:')
for c,CTAU_N_REST in enumerate(CT_N_REST):
    print('iteration and ctau : ', c, CTAU_N_REST)
    for ic, cos in enumerate(cos_vec):
        for E_parent in Evec_parent:
            for E_dau in Evec_daughter:
                
                IK=integrate.quad(Kp_integrando_inX,min(slantes[ic]),max(slantes[ic]),args=(E_parent,E_dau,CTAU_N_REST,MN,ic),epsabs= 0.00, epsrel= 1e-2, limit=300)
                IP=integrate.quad(Pip_integrando_inX,min(slantes[ic]),max(slantes[ic]),args=(E_parent,E_dau,CTAU_N_REST,MN,ic),epsabs= 0.00, epsrel= 1e-2,limit=300)
                K_points.append(IK[0])
                Pi_points.append(IP[0])
                Ep_points.append(E_parent)
                Ed_points.append(E_dau)
                ctau_points.append(CTAU_N_REST)
                cos_points.append(cos)
                data_1.append([MN,IK[0],IP[0],E_parent,E_dau,CTAU_N_REST,cos])
                
np.savetxt("1st_integration_mN_{:.2f}.txt".format(MN), data_1)
mN_points, K_points, Pi_points, Ep_points, Ed_points, ctau_points, cos_points= np.transpose(np.loadtxt("1st_integration_mN_{:.2f}.txt".format(MN) ))

x1= Ep_points
x2= Ed_points
x3= ctau_points
x4= cos_points
fx_Kp= K_points
fx_Pip= Pi_points

df_Kp= pd.DataFrame.from_dict({'X2':x2, 'X1':x1, 'X3': x3, 'X4': x4, 'FX':fx_Kp })
df_Pip= pd.DataFrame.from_dict({'X2':x2, 'X1':x1, 'X3': x3, 'X4': x4, 'FX':fx_Pip })

interpolador_Kp= FourD_interpolation(df_Kp)
interpolador_Pip= FourD_interpolation(df_Pip)

def Kp_integrando_Ep(Ep,Ed,ctauR,cos):
    return  float( interpolador_Kp([Ep,Ed,ctauR,cos])[0] )
def Pip_integrando_Ep(Ep,Ed,ctauR,cos):
    return  float( interpolador_Pip([Ep,Ed,ctauR,cos])[0] )
def Kp_integrando_LogEp(LogEp,Ed,ctauR,cos):
    return np.exp(LogEp)*Kp_integrando_Ep( np.exp(LogEp),Ed,ctauR,cos)
def Pip_integrando_LogEp(LogEp,Ed,ctauR,cos):
    return np.exp(LogEp)*Pip_integrando_Ep( np.exp(LogEp),Ed,ctauR,cos)

# 2nd integration
K2_points=[]
Pi2_points=[]
E_points=[]
ctau_points=[]
cos_points=[]
data_2=[]

print('Integration in Ep started:')
for c,CTAU_N_REST in enumerate(CT_N_REST):
    print('iteration and ctau : ', c, CTAU_N_REST)
    for ic, cos in enumerate(cos_vec):
        for energy in Evec_daughter:
                I2K=integrate.quad(Kp_integrando_LogEp,np.log(lim_inf_Ep(mKp,MN,me,energy)),np.log(lim_sup_Ep(mKp,MN,me,energy)),args=(energy,CTAU_N_REST,cos),epsabs= 0.00, epsrel= 1e-2, limit=300)
                I2P=integrate.quad(Pip_integrando_LogEp,np.log(lim_inf_Ep(mPip,MN,me,energy)),np.log(lim_sup_Ep(mPip,MN,me,energy)),args=(energy,CTAU_N_REST,cos),epsabs= 0.00, epsrel= 1e-2,limit=300)
                K2_points.append(I2K[0])
                Pi2_points.append(I2P[0])
                
                E_points.append(energy)
                ctau_points.append(CTAU_N_REST)
                cos_points.append(cos)
                data_2.append([MN,I2K[0],I2P[0],energy,CTAU_N_REST,cos])
                
np.savetxt("2nd_integration_mN_{:.2f}.txt".format(MN), data_2)                
mN_points, K_points, Pi_points, Ed_points, ctau_points, cos_points= np.transpose(np.loadtxt("2nd_integration_mN_{:.2f}.txt".format(MN) ))

x1= Ed_points
x2= cos_points
x3= ctau_points
fx_Kp= K_points
fx_Pip= Pi_points

df_Kp= pd.DataFrame.from_dict({'X2':x2, 'X1':x1, 'X3': x3, 'FX':fx_Kp })
df_Pip= pd.DataFrame.from_dict({'X2':x2, 'X1':x1, 'X3': x3,'FX':fx_Pip })

interpolador3D_Kp= ThreeD_interpolation(df_Kp)
interpolador3D_Pip= ThreeD_interpolation(df_Pip)

def Kp_integrando_Ed(Ed,cos,ctauR):
    return  float( interpolador3D_Kp([Ed,cos,ctauR])[0] )
def Pip_integrando_Ed(Ed,cos,ctauR):
    return  float( interpolador3D_Pip([Ed,cos,ctauR])[0] )

# 3rd integration
print('Calculation of number of events started:')
Evts_Kp=[]
Evts_Pip=[]
data_3=[]

for CTAU_N_REST in CT_N_REST:
    events_Kp=[]
    events_Pip=[]
    for i, cos in enumerate(cos_value):

        IKp=integrate.nquad( lambda e,cos: Kp_integrando_Ed(e,cos,CTAU_N_REST)*fn*Seff_SK(cos,ctau_lab_func(CTAU_N_REST,MN,e)) , [ [min(Evec_daughter),max(Evec_daughter)] ,[cos_min[i],cos_max[i] ] ], opts=my_dict_integral )
        IPip=integrate.nquad( lambda e,cos: Pip_integrando_Ed(e,cos,CTAU_N_REST)*fn*Seff_SK(cos,ctau_lab_func(CTAU_N_REST,MN,e)) , [ [min(Evec_daughter),max(Evec_daughter)] ,[cos_min[i],cos_max[i] ] ], opts=my_dict_integral )
    
        events_Kp.append(IKp[0])
        events_Pip.append(IPip[0])
        #print(CTAU_N_REST,IKp[0],cos)
        data_3.append([MN,IKp[0],IPip[0],CTAU_N_REST,cos])
        
    Evts_Kp.append(events_Kp)
    Evts_Pip.append(events_Pip)
    

np.savetxt("3rd_integration_mN_{:.2f}.txt".format(MN), data_3) 
# save number of events in a csv:
mN_po, K_ev, Pi_ev, ct_po, cos_po= np.transpose(np.loadtxt("3rd_integration_mN_{:.2f}.txt".format(MN) ))
dff=pd.DataFrame.from_dict({'EvtKaon':K_ev, 'EvtPion':Pi_ev,'ctau':ct_po,'cosine':cos_po})
dff.to_csv (r'Events_SK.csv', index = None, header=True)
print('')
print('File: Events_SK.csv saved succesfully! ')
print('')
print(dff.head())