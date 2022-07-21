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

data = pd.read_csv('C:/Users/morel/OneDrive/Bureau/data/20220222/nem monazite EG/10x NA 0.3/images/Spectrum/image_Pos0_Polarizer0_X-Axis0000_Y-Axis0000.csv',sep=',') # Import data from csv

wav=[float(a) for a in list(data)[1:-1]] # the wavelength vector to plot the principal components
spectra=data.values[:][0][1:-1] # initiate tab of intensity to use np.vstack
angles=[]

for i in range(0,37) :
    print('retrieving '+str(i*10)+'° data')
    for j in ['10x NA 0.3','20x NA 0.75','40x NA 0.6','60x oil NA 1.4','100x NA 0.9']:
        data = pd.read_csv('C:/Users/morel/OneDrive/Bureau/data/20220222/nem monazite EG/'+j+'/images/Spectrum/image_Pos0_Polarizer'+str(i)+'_X-Axis0000_Y-Axis0000.csv',sep=',') # Import data from csv
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

####################################################################
'''
PART 2 : ANGLE PREDICTION PART
'''

###### SEPARATING TRAIN SET AND TEST SET

x=spectra ; y=angles


###### PRINCIPAL COMPONENT ANALYSIS ON TRAIN SET

accuracy=[] ; n=[] ; meanabsdev=[]
for number in range(10):
    tsize=0.2
    acc=0 ; mad=0
    for rando in range(1):
        X_train, X_test, y_train, y_test = train_test_split(x, y, 
        test_size=0.2, random_state=0)
    
        pca = PCA(n_components=2)
        X_train = pca.fit_transform(X_train)
        
        ###### MATCH ANGLES AND COEFFICIENTS IN THE TRAIN SET
        
        classifier = RandomForestClassifier(random_state=0)
        classifier.fit(X_train, y_train)
        
        ###### PRINCIPAL COMPONENT ANALYSIS OF TEST SET AND ANGLE PREDICTION
        
        X_test = pca.transform(X_test) # tranform and not fit_transform
        y_pred = classifier.predict(X_test)
        
        ###### COMPARING PREDICTED ANGLES AND REAL ANGLES
        y_test=[abs((a-20)%180) for a in y_test]
        y_pred=[abs((a-20)%180) for a in y_pred]
        
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.set_aspect('equal')
        plt.plot([0,180],[0,180],color='black',label='ideal result')
        plt.plot([0,180],[180,0],color='black')
        plt.scatter(y_test,y_pred,label='predictions')
        plt.xlabel('known angle')
        plt.ylabel('predicted angle')
        plt.title('monazite angle prediction with 80% train 20% test')
        plt.legend()
        plt.show()
        
        acc+=accuracy_score(y_test, y_pred)
        mad+=np.mean([abs(x-y) for (x,y) in zip(y_test,y_pred)])
    
    n.append((1-tsize))
    accuracy.append(acc/100)
    meanabsdev.append(mad/100)

plt.plot(n,accuracy,lw=3, color='k')
plt.title('accuracy',fontsize=15)
plt.xlabel('train set size',fontsize=15)
plt.ylabel('ratio of perfect predictions',fontsize=15)
plt.grid()
plt.show()

plt.plot(n,meanabsdev,lw=3, color='k')
plt.title('mean absolute deviation',fontsize=15)
plt.xlabel('train set size',fontsize=15)
plt.ylabel('mean absolute deviation (°)',fontsize=15)
plt.grid()
plt.show()

