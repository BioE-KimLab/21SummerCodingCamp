import numpy as np
import itertools
#from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import multiprocessing
import sys

def normalize(X,X_pred): # sklearn.preprocessing.StandardScaler. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# or https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
    X_mean=[]
    X_std=[]
    new_X=[]
    for x in X:
        mu=np.mean(x)
        sigma=np.std(x)
        X_mean.append(mu)
        X_std.append(sigma)
        new_X.append((x-mu)/sigma)

    new_X_pred=[]
    for i,x in enumerate(X_pred):
        new_X_pred.append((x-X_mean[i])/X_std[i])

    return np.array(new_X),np.array(new_X_pred)

#########################
solvent=         ['acetone', 'dmso','water', '1-butanol','2-butanol','acetontirile','i-proh', 'thf', 'dmac', 'methf','MiBK', 'toluene', 'dioxane']
G1     =np.array([    -8.69, -18.23,  9.33,        5.95,         3.63,       -2.47,     3.64,-14.96,-16.41,  -15.53 , -7.88,    -15.05,   -15.11])   #fruc-h3o-sol complexation
G2     =np.array([    -3.73, -11.13, -8.74,       -7.85,        -6.15,       -9.67,    -7.95, -2.57,  1.30,   -0.81 , -3.60,      0.80,    -6.56])   #complex - carbenium + 2h2o
G4     =np.array([     0.42,  -15.4,   0.0,       -0.24,         1.25,       -1.97,    -0.93, -1.80, -1.11,   -0.34 ,  3.30,      7.52,     1.60])   #carbenium-water -> carbenium-org
nbo    =np.array([   -0.618, -1.035,-0.995,      -0.796,         -0.8,       -0.45,   -0.801,-0.623,-0.708,  -0.633 ,-0.611,    -0.228,   -0.612])   #of solvent w/ smd model
dipole =np.array([   3.9752, 6.075,   2.65,       2.376,       2.3686,      5.2532,   2.4512,2.1984,5.4553,   2.1027,3.7134,    0.4938,   0.0001])   #of solvent w/ smd model
COlen  =np.array([   2.5787, 1.4791,2.5507,     2.53885,      2.53582,     2.76543,  2.49458,2.3831,2.4412,   2.5633,2.5665,    3.2942,   2.3321])   #carbenium carbon - solvent O distance
HLGap  =np.array([ 2.221382192,27.25271994,49.33476495,16.94274553,14.65861243,49.97482423,15.97638153, 6.682971849, 60.49187664,  18.17893845, 0.094126364,19.528083,8.797677495  ]) #(h3o-fruc)..(sol)
HLGap2 =np.array([ 32.74344571,41.31522548,64.91585778,26.89505717,24.99997848,60.15306067,28.55795734,33.06347556,17.10590897,38.01452551,23.19902622,25.02507886,5.00752581   ])    #(h3o-sol)..(fruc)
OHFreq =np.array([ 3456.54, 3569.53, 3582.58, 3575.66, 3657.01, 3407.66, 3681.85, 3537.2, 3694.91, 3600.69, 2909.17, 2170.89, 3495.28   ])
Alpha =np.array([ 0.0, 0.0, 0.82, 0.37, 0.33, 0.07, 0.33, 0.0,   0.0,  0.0,  0.0, 0.0, 0.0])
Beta  =np.array([ 0.5, 0.78,0.35, 0.48, 0.56, 0.32, 0.56, 0.48, 0.78, 0.53, 0.49,0.14, 0.64 ])
B1=np.array([2.28,2.62,1.27,2.57,2.01,1.29,1.90,1.89,2.62,2.34,2.25,1.75,1.55])
B5=np.array([7.51,7.17,7.79,7.72,8.08,7.61,8.15,7.53,7.28,7.40,7.96,7.47,7.31])
nbo_carbocation=np.array([0.716, 0.580, 0.729, 0.727, 0.729, 0.713, 0.728, 0.721, 0.721, 0.722, 0.720, 0.700, 0.718  ])
epsinf=np.array([ 1.8496,2.187441,1.776889,1.957201,1.95384484,1.79667216,1.896129,1.979649,2.06640625,1.976836,1.951609,2.24041024,2.01725209])
ETnorm=np.array([0.35,0.44,1,0.6,0.51,0.46,0.57,0.21,0.4,0.18,0.27,0.1,0.16])
AN=np.array([12.5,19.3,54.8,36.8,28.128,18.9,33.5,8,13.6,6.865,10.1,4.239,10.3])
DN=np.array([17,29.8,18,19.5,31.06,14.1,21.1,20,27.8,12,18.836,0.1,14.3])
pi=np.array([0.71,1,1.09,0.503,0.4,0.75,0.48,0.58,0.88,0.53,0.65,0.54,0.553])
Zvalue=np.array([65.7,70.2,94.6,77.7,75.4,71.3,76.3,58.8,66.9,55.3,62,61.408,64.5])
KTa=np.array([0.08,0,1.17,0.84,0.69,0.19,0.76,0,0,0,0.02,0,0])
KTb=np.array([0.43,0.76,0.47,0.84,0.8,0.4,0.84,0.55,0.76,0.58,0.48,0.11,0.37])
HMF_stab=np.array([ -3.23,   -2.59, -1.89,  -1.41,        -1.18,         -2.83,   -1.49,  0.23,  -4.81,  -0.16,  -1.89,  3.61,     1.68  ])
HMF_lumo=np.array([ 7.12,    6.18,  0.09,   -0.29,         0.16,           5.16,      0.46,   5.07,  7.73,   4.74,  7.54,   4.25,     3.39   ]) 
Gsolv4_sac=        np.array([-78.4,-80.3,-77.1,-86.4,-83.8,-71.9,-83,-82.3,-83,-83.9,-85.7,-89.4,-94.5])
Gsolv6_sac=        np.array([-78.9,-85.5,-77.7,-85.6,-85.9,-74.7,-83.6,-83.9,-85,-86.1,-86.3,-89,-88.7])
Gsolv8_sac=        np.array([-84.6,-86.8,-77.7,-83,-83.2,-81.8,-83.9,-84.8,-87.2,-84.4,-83.3,-78.6,-85.2])
fruc_cat_sac=      np.array([-82.2,-83.9,-75.5,-80.7,-80.8,-79.5,-81.4,-82.6,-84.6,-82.2,-81.2,-77.4,-82.9])


###########################
solvent_new=['etoh','gvl','cpme','mtbe']
G1_new =np.array([ 5.63, -9.20,  -21.12, -28.58])   #fruc-h3o-sol complexation
G2_new =np.array([ -9.54, -6.88, 9.27, 10.42 ])   #complex - carbenium + 2h2o
G4_new =np.array([ -1.00, -1.80, 7.12, 6.22 ])   #carbenium-water -> carbenium-org
nbo_new=np.array([   -0.803, -0.635, -0.629, -0.637 ])   #of solvent w/ smd model
dipole_new=np.array([ 2.4173, 6.4098, 1.6223, 1.6503 ])   #of solvent w/ smd model
COlen_new=np.array([  2.52507, 1.50846, 1.58861, 1.56612 ])   #carbenium carbon - solvent O distance
HLGap_new=np.array([   29.79, 13.64, 3.71, 7.47  ]) #(h3o-fruc)..(sol)
HLGap2_new=np.array([  30.17, 35.24, 21.87, 24.30 ])    #(h3o-sol)..(fruc)
OHFreq_new=np.array([ 3631.07,3429.84, 3397.49, 3343.67 ])
Alpha_new=np.array([ 0.37, 0.0,  0.0, 0.0 ])
Beta_new=np.array([  0.48,0.55,  0.53, 0.49  ])
B1_new=np.array([2.09,1.32,3.40,2.61])
B5_new=np.array([7.93,7.49,7.32,7.13])
nbo_carbocation_new=np.array([0.726, 0.580, 0.637 ,0.637 ])
epsinf_new=np.array([1.81629529,2.050624,2.01327721,1.874161])
ETnorm_new=np.array([0.65,0.310030864,0.164506173,0.29])
AN_new=np.array([37.9,11.15245,6.3903,10.602])
DN_new=np.array([19.2,23.42,20.746,17.69])
pi_new=np.array([0.54,0.83,0.42,0.25])
Zvalue_new=np.array([79.6,64.366,60.184,58.45])
KTa_new=np.array([0.86,0.0,0.0,0.0])
KTb_new=np.array([0.75,0.6,0.53,0.45])
HMF_stab_new=np.array([ -3.03,-0.75,  -0.41, 0.52 ])
HMF_lumo_new=np.array([  1.10, 4.40, 4.75, 7.06  ]) 
Gsolv4_sac_new=     np.array([-81.5,-85.2,-87.5,-83.2])
Gsolv6_sac_new=     np.array([-81.4,-86.5,-90.1,-87.3]) 
Gsolv8_sac_new=     np.array([-83.5,-84.2,-83.0,-83.6])
fruc_cat_sac_new=   np.array([-80.9,-82.2,-81.0,-81.4])
#######################################

exp_yield=np.array([     36,   67.4,   6.7,        10.2,         12.6,         9.1,      7.5,     5,   0.4,   16.5  , 14.8 ,      10.3,     57.6])
exp_yield_new=[38, 61, 31, 19]

X=np.array([G1,G2,G4,B1,B5,OHFreq,nbo_carbocation,COlen,HLGap,HLGap2,\
            HMF_stab,HMF_lumo,Gsolv8_sac,Gsolv6_sac,fruc_cat_sac,Gsolv4_sac,\
            dipole,nbo,Alpha,Beta,epsinf,ETnorm,\
            AN,DN,Zvalue,KTa,KTb,pi])

X_pred=np.array([G1_new,G2_new,G4_new,B1_new,B5_new,OHFreq_new,nbo_carbocation_new,COlen_new,HLGap_new,HLGap2_new,\
            HMF_stab_new,HMF_lumo_new,Gsolv8_sac_new,Gsolv6_sac_new,fruc_cat_sac_new,Gsolv4_sac_new,\
            dipole_new,nbo_new,Alpha_new,Beta_new,epsinf_new,ETnorm_new,\
            AN_new,DN_new,Zvalue_new,KTa_new,KTb_new,pi_new])

Properties= 'G1,G2,G4,B1,B5,OHFreq,nbo_carbocation,COlen,HLGap,HLGap2,HMF_stab,HMF_lumo,Gsolv8_sac,Gsolv6_sac,fruc_cat_sac,Gsolv4_sac,dipole,nbo,Alpha,Beta,epsinf,ETnorm,AN,DN,Zvalue,KTa,KTb,pi'.split(',')

#Standardization
X,X_pred=normalize(X,X_pred)

#Covariance (bivariate)
X=np.transpose(X)
X_pred=np.transpose(X_pred)

mean_vec = np.mean(X,axis=0)
cov_mat = np.corrcoef(X.T) 
#print(len(cov_mat))
#print(cov_mat)

highly_correlated=[]
for i in range(len(cov_mat)):
    for j in range(i+1,len(cov_mat)):
        if cov_mat[i][j]> 0.7:
            #print(Properties[i],Properties[j],'%.2f' % cov_mat[i][j])
            print(i,j,'%.2f' % cov_mat[i][j])
            highly_correlated.append((i,j))

combs = []
#for i in range(1,7):
for i in range(6,7):
    combs+= itertools.combinations(range(len(Properties)),i)

comb_chosen=[]
Validation_Errors=[]

for comb in combs:
    skip=False
    for pair in highly_correlated:
        if len(set(pair) & set(comb))==2: 
            skip=True
            break
    if skip:
        continue

    X=np.array([G1,G2,G4,B1,B5,OHFreq,nbo_carbocation,COlen,HLGap,HLGap2,\
                HMF_stab,HMF_lumo,Gsolv8_sac,Gsolv6_sac,fruc_cat_sac,Gsolv4_sac,\
                dipole,nbo,Alpha,Beta,epsinf,ETnorm,\
                AN,DN,Zvalue,KTa,KTb,pi])

    X_pred=np.array([G1_new,G2_new,G4_new,B1_new,B5_new,OHFreq_new,nbo_carbocation_new,COlen_new,HLGap_new,HLGap2_new,\
                HMF_stab_new,HMF_lumo_new,Gsolv8_sac_new,Gsolv6_sac_new,fruc_cat_sac_new,Gsolv4_sac_new,\
                dipole_new,nbo_new,Alpha_new,Beta_new,epsinf_new,ETnorm_new,\
                AN_new,DN_new,Zvalue_new,KTa_new,KTb_new,pi_new])

    X,X_pred=normalize(X,X_pred)
    mask_array=list(comb)
    mask = np.zeros(len(X),dtype=bool)
    mask[mask_array] = True
    X = X[mask,...]
    X_pred = X_pred[mask,...]

    X=np.transpose(X)
    X_pred=np.transpose(X_pred)


    #Linear Regression
    reg = LinearRegression().fit(X,exp_yield)

    yield_known_solvents=np.clip(reg.predict(X),0, 100)
    yield_new_solvents=np.clip(reg.predict(X_pred),0,100)

    MAE=np.mean(np.absolute(yield_known_solvents - exp_yield))
    max_deviation=max(np.absolute(yield_known_solvents - exp_yield))
    R2=reg.score(X,exp_yield)

    case1=0.0 not in yield_known_solvents and 100.0 not in yield_known_solvents
    case2=0.0 not in yield_new_solvents and 100.0 not in yield_new_solvents

    if 0.95<R2<0.99 and case1 and case2: # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
        # Leave-one-out cross-validation, estimate mean error
        Validation_set_indices=itertools.combinations(range(len(X)),1)

        Err=0.0
        X_for_Q2=[]
        Y_for_Q2=[]        
        for validation_set in Validation_set_indices:
            training_set=sorted(list(set(range(len(X))) - set(validation_set)))

            X_train=np.array([ X[x] for x in training_set ])
            X_valid=np.array([ X[x] for x in list(validation_set) ])
            Y_train=np.array([ exp_yield[x] for x in training_set ])
            Y_valid=np.array([ exp_yield[x] for x in list(validation_set) ])

            reg = LinearRegression().fit(X_train,Y_train)
            R2=reg.score(X_train,Y_train)
            yield_validation_set=reg.predict(X_valid)

            X_for_Q2.append(yield_validation_set[0])
            Y_for_Q2.append(Y_valid[0])

            Err+=np.mean(np.absolute(yield_validation_set - Y_valid))
        Err/=13.0
        q2=r2_score(Y_for_Q2,X_for_Q2)

        comb_chosen.append(comb)
        Validation_Errors.append(Err)

        #print(comb)

        reg = LinearRegression().fit(X,exp_yield)
        yield_known_solvents=reg.predict(X)
        yield_new_solvents=reg.predict(X_pred)
        MAE=np.mean(np.absolute(yield_known_solvents - exp_yield))
        RMSE=np.sqrt(np.mean(  np.square(yield_known_solvents - exp_yield)  ))
        R2=reg.score(X,exp_yield)

        MAE_new=np.mean(np.absolute(yield_new_solvents - exp_yield_new))

        #For analyzing fitted coeffs, contributions of each variable
        print(reg.coef_)
        print(reg.intercept_)

        '''
        print(reg.coef_*X)
        print(reg.coef_*X_pred)
        '''

        print("R2: "+str(R2))
        print("Q2: "+str(q2))
        print("MAE: "+ str(MAE) )
        print("MAE_new: "+ str(MAE_new) )
        print("RMSE: "+ str(RMSE) )
        print("======================")
        for i, expyield in enumerate(exp_yield):
            print(solvent[i]+' '+str(yield_known_solvents[i])+' '+str(expyield))
        print("======================")
        for i, pred_yield in enumerate(yield_new_solvents):
            print(solvent_new[i]+ ' '+str(pred_yield))
        print("======================")

