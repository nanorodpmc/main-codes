import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, wiener

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

###### RETRIEVE DATA

data = pd.read_csv('data/10x NA 0.3/image_Pos0_Polarizer0_X-Axis0000_Y-Axis0000.csv',sep=',') # Import data from csv

wav=[float(a) for a in list(data)[1:-1]] # the wavelength vector to plot the principal components
spectra=data.values[:][0][1:-1] # initiate tab of intensity to use np.vstack
angles=[]

for i in range(0,37) :
    print('retrieving '+str(i*10)+'° data')
    for j in ['10x NA 0.3','20x NA 0.75','40x NA 0.6','60x oil NA 1.4','100x NA 0.9']:
        data = pd.read_csv('data/'+j+'/image_Pos0_Polarizer'+str(i)+'_X-Axis0000_Y-Axis0000.csv',sep=',') # Import data from csv
        val=list(data.values[:][0])[1:-1] ; mini=min(val)
        val=[(x-mini) for x in val]
        #val=wiener(val,10) 
        spectra = np.vstack([spectra, val])
        angles.append(i*10)

angles=np.array(angles) # change list in array
spectra=spectra[1:]     # remove first value used for initiation

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
    for l in range(5):   # to average on the 5 measurements
        a+=Xt2[i*5+l][0] ; b+=Xt2[i*5+l][1]   
    coeff1.append(a) ; coeff2.append(b)

fig=plt.figure()
ax1 = fig.add_subplot(projection='polar')
ax1.plot(angle,coeff1, label='average PC1 coefficient')
ax1.plot(angle,coeff2, label='average PC2 coefficient')
plt.gca().get_yaxis().set_visible(False)
plt.xticks(fontsize=15) ; plt.legend(fontsize=15)
plt.show()
