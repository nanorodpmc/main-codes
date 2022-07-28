import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import promptlib
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.widgets import Slider,Button

def select_data(val):
    global data
    global files
    global max_angle
    global max_z
    print('select folder')
    prompter = promptlib.Files()
    path = prompter.dir()

    folder=path+'/images/Spectrum/'

    files=[_ for _ in os.listdir(folder) if _.endswith('.csv')]

    print('retrieving data : ',path)
    data=pd.DataFrame()
    ind=0 ; df=pd.DataFrame()
    for file in tqdm(files):
        datum=pd.read_csv(folder+file, header = None).T
        if file == files[0] : data['wavelength']=datum[0][1:-1]
        try :
            begin='Polarizer' ; end='_'
        except:
            try:
                begin='pol' ; end='.'
            except:
                begin='extraIndex' ; end='_'
        angle=int((file.split(begin))[1].split(end)[0])
        z=int(file[:2])
        
        df[str(z)+'_'+str(angle)]=datum[1][1:-1]-min(datum[1][1:-1])
        if z!=ind :
            data=pd.concat([data, df], axis=1)
            ind+=1 ; df=pd.DataFrame()
    data=pd.concat([data, df], axis=1)
    max_angle=angle
    max_z=z                       

    data=data.reindex(index=data.index[::-1]) 
    data.reset_index(inplace=True, drop=True)

    #data = data[['wavelength']+[str(i) for i in range(int(max_angle)+1)]] #to sort the columns by angle value

    data['mean intensity']=data.drop('wavelength', axis=1).mean(axis=1) #remove wavelength to calculate mean intensity


select_data(0)

fig = plt.figure()

ax1 = fig.add_subplot(211,projection='polar')
line, = ax1.plot([],[],'--o')
linefit, = ax1.plot([],[])

ax2 = fig.add_subplot(212)
spectrum, =ax2.plot(data['wavelength'],data['mean intensity'])
linevalue, =ax2.plot([data['wavelength'][1],data['wavelength'][1]],[0,max(data.drop('wavelength', axis=1).max())])
ax2.set_ylim(ymax=max(data['mean intensity'])*1.05)


value=0
z=0

def polardiag(val):
    global value
    global z
    if value%2==0 :
        def lobe_fitting(x, a, b,c,alpha,phi):
            return a*np.cos(x+alpha)**2 + b*np.sin(x+alpha)**2 + c*np.sin(2*x+phi)**2

        x=np.array([i/int(max_angle)*2*np.pi for i in range(int(max_angle)+1)])       #angles for polar diagram
        y=np.array([data[str(int(z_slider.val))+'_'+str(i)][point_slider.val] for i in range(int(max_angle)+1)]) #intensities for polar diagrams

        y=y/max(y) #normalize
        
        errfunc = lambda p, x, y: (lobe_fitting(x, *p) - y)**2
        guess = [0.5,0.5,0,0,0]
        optim,success = optimize.leastsq(errfunc, guess[:], args=(x, y))

        angles_fit=np.arange(0,2*np.pi+0.1,0.1)
        intensities_fit=[lobe_fitting(i, optim[0], optim[1], optim[2], optim[3], optim[4]) for i in angles_fit]
        
        linevalue.set_xdata([data['wavelength'][point_slider.val],data['wavelength'][point_slider.val]]) #moving vertical bar
        line.set_xdata(x) ; line.set_ydata(y)
        linefit.set_xdata(angles_fit) ; linefit.set_ydata(intensities_fit)
        
        ax1.set_title(str(round(data['wavelength'][point_slider.val],1))+' nm',fontsize=14, y=0.0, pad=-30)
        ax2.set_title('a cos²(x) + b sin²(x) + c sin²(2x)'+'\n'
                    +'a='+str(round(optim[0],2))+'\n'
                    +'b='+str(round(optim[1],2))+'\n'
                    +'c='+str(round(optim[2],2)),fontsize=14, y=1.1,loc='left')
    if value%2==1 :
        def lobe_fitting(x, a, b,alpha):
            return a*np.cos(x+alpha)**2 + b*np.sin(x+alpha)**2 

        x=np.array([i/int(max_angle)*2*np.pi for i in range(int(max_angle)+1)])       #angles for polar diagram
        y=np.array([data[str(int(z_slider.val))+'_'+str(i)][point_slider.val] for i in range(int(max_angle)+1)]) #intensities for polar diagrams

        y=y/max(y) #normalize
        
        errfunc = lambda p, x, y: (lobe_fitting(x, *p) - y)**2
        guess = [0.5,0.5,0]
        optim,success = optimize.leastsq(errfunc, guess[:], args=(x, y))

        angles_fit=np.arange(0,2*np.pi+0.1,0.1)
        intensities_fit=[lobe_fitting(i, optim[0], optim[1], optim[2]) for i in angles_fit]
        
        linevalue.set_xdata([data['wavelength'][point_slider.val],data['wavelength'][point_slider.val]]) #moving vertical bar
        line.set_xdata(x) ; line.set_ydata(y)
        linefit.set_xdata(angles_fit) ; linefit.set_ydata(intensities_fit)
        
        ax1.set_title(str(round(data['wavelength'][point_slider.val],1))+' nm',fontsize=14, y=0.0, pad=-30)
        ax2.set_title('a cos²(x) + b sin²(x)'+'\n'
                    +'a='+str(round(optim[0],2))+'\n'
                    +'b='+str(round(optim[1],2)),fontsize=14, y=1.1,loc='left')

    fig.canvas.draw_idle()

def change_fit(val):
    global value
    value-=1
    polardiag(val)

def changespectrum(val):
    global z
    spectrum.set_ydata(data[str(z_slider.val)+'_'+str(angle_slider.val)])
    ax2.set_ylim(ymin=0,ymax=1.05*max(data[str(int(z_slider.val))+'_'+str(angle_slider.val)]))
    fig.canvas.draw_idle()

def resetspectrum(val):
    global linevalue
    spectrum.set_ydata(data['mean intensity'])
    linevalue.set_xdata([data['wavelength'][1],data['wavelength'][1]])
    linevalue.set_xdata([data['wavelength'][point_slider.val],data['wavelength'][point_slider.val]])
    linevalue.set_ydata([0,max(data.drop('wavelength', axis=1).max())])
    ax2.set_ylim(ymin=0,ymax=max(data['mean intensity'])*1.05)
    fig.canvas.draw_idle()

def update_data(val):
    select_data(val)
    resetspectrum(val)
    polardiag(val)


axpoint = plt.axes([0.091, 0.05, 0.819, 0.04])
point_slider = Slider(ax=axpoint,label='',
                      valmin=0, valmax=len(data['wavelength'])-1,
                      valinit=1, valstep=1)
point_slider.on_changed(polardiag)


axangle = plt.axes([0.8, 0.65, 0.04, 0.30])
angle_slider = Slider(ax=axangle,label='',
                      valmin=0, valmax=int(max_angle),
                      valinit=0, valstep=1, orientation='vertical')
angle_slider.on_changed(changespectrum)

axz = plt.axes([0.9, 0.65, 0.04, 0.30])
z_slider = Slider(ax=axz,label='',
                      valmin=0, valmax=max_z,
                      valinit=0, valstep=1, orientation='vertical')
z_slider.on_changed(polardiag)
z_slider.on_changed(changespectrum)

meanbutton= Button(plt.axes([0.85, 0.56, 0.04, 0.05]),label='mean', hovercolor='0.975')
meanbutton.on_clicked(resetspectrum)

changefitbutton= Button(plt.axes([0.05, 0.75, 0.15, 0.05]),label='change fit', hovercolor='0.975')
changefitbutton.on_clicked(change_fit)

changedatabutton= Button(plt.axes([0.05, 0.85, 0.15, 0.05]),label='select folder', hovercolor='0.975')
changedatabutton.on_clicked(update_data)

fig.subplots_adjust(bottom=0.14,top=1,left=0.05,right=0.95)
plt.show()