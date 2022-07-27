import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, wiener

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time
from tqdm import tqdm

data = pd.read_csv('data rhabdophane/Nem 1/pol__Polarizer000_X-Axis0000_Y-Axis0000.csv',sep=',') # Import data from csv

wav=[float(a) for a in list(data)[1:-1]]
spectra=data.values[:][0][1:-1]
angles=[]

print('retrieving data')
for z in tqdm(range(0,37)) :
    for j in ['Nem 1','Nem 2','Nem 3','Nem 4','Tact 1','Tact 2','Tact 3']:
        if j=='Nem 1': i=(z+7)%36
        if j=='Nem 2': i=(z+5)%36
        if j=='Nem 3': i=(z+5)%36
        if j=='Nem 4': i=(z+3)%36
        if j=='Tact 1': i=(z+3)%36
        if j=='Tact 2': i=(z+0)%36
        if j=='Tact 3': i=(z+8)%36

        a=''
        if i<10: a ='0'
        try :
            data = pd.read_csv('data rhabdophane/'+j+'/pol__Polarizer0'+a+str(i)+'_X-Axis0000_Y-Axis0000.csv',sep=',') # Import data from csv
        except:
            data = pd.read_csv('data rhabdophane/'+j+'/pol__Polarizer0'+a+str(i)+'_Y-Axis0000_Z-Axis0000.csv',sep=',') # Import data from csv
   
        wav=[float(a) for a in list(data)[1:-1]]
        val=list(data.values[:][0])[1:-1]
        mini=min(val)
        val=[(x-mini) for x in val]
        #val=savgol_filter(val, 25, polyorder = 5)
        #val=wiener(val,10)
        spectra = np.vstack([spectra, val])
        angles.append(z*10)


angles=np.array(angles)
spectra=spectra[1:]

###### PRINCIPAL COMPONENT ANALYSIS
    
skpca = PCA(n_components=2)
Xt2 = skpca.fit_transform(spectra)

###### PLOT PRINCIPAL COMPONENTS

comp1=skpca.components_[0] #PC1
comp2=skpca.components_[1] #PC2
plt.plot(wav,comp1,label='PC1 '+str(round(100*skpca.explained_variance_ratio_[0]))+'% of variance',lw=2)
plt.plot(wav,comp2,label='PC2 '+str(round(100*skpca.explained_variance_ratio_[1]))+'% of variance',lw=2)
plt.gca().get_yaxis().set_visible(False)
plt.xlabel('wavelength (nm)',fontsize=20)
plt.xticks(fontsize=20) ; plt.legend(fontsize=20)
plt.show()
  
######PLOT COEFFICIENTS

coeff1=[] ; coeff2=[] ; angle=[]
for i in range(37):
    angle.append(i/18*np.pi)
    a=0 ; b=0
    for l in range(7):   # to average on the 7 measurements
        a+=Xt2[i*7+l][0] ; b+=Xt2[i*7+l][1]   
    coeff1.append(a) ; coeff2.append(b)

fig=plt.figure()
ax1 = fig.add_subplot(projection='polar')
ax1.plot(angle,coeff1, label='average PC1 coefficient')
ax1.plot(angle,coeff2, label='average PC2 coefficient')
plt.gca().get_yaxis().set_visible(False)
plt.xticks(fontsize=15) ; plt.legend(fontsize=15)
plt.show()
