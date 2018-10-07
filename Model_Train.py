from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uproot as up
import numpy as np
import sys
import tensorflow as tf;
import os.path
import keras.backend as K
import time
from keras.models import load_model
from keras.models import Sequential
from keras import optimizers
from keras.models import Model
from keras.layers import Dense,Dropout,Activation,TimeDistributed
from keras.layers import LSTM,Input,BatchNormalization,concatenate
from keras.utils import plot_model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.regularizers import l2
from sklearn.utils import shuffle
from sklearn import metrics
np.seterr(all='ignore')

out_png0 = 'FHC_LSTM_Loss0.png'
out_png1 = 'FHC_LSTM_Loss1.png'
out_png2 = 'FHC_LSTM_Loss2.png'
out_png3 = 'FHC_LSTM_Loss3.png'

def DataHandle(idx,swap,horn,limit,nepoch,frac):
    mainlist = ["SliceEnergy","SliceTrueNuPdg","SliceTrueNuCCNC","SliceTrueNuMode",
                "Shw0","Shw1","Shw2","Shw3","Shw4","Shw5","Shw6"]
    #29 features for every reconstructed prong. 1 feature is redundant and will be removed.
    numfeatures = 29
    nslc = 100
    E= []
    X0 = []
    X1 = []
    X2 = []
    X3 = []
    X4 = []
    X5 = []
    X6 = []
    Pdg = []
    CCNC = []
    #We are dealing with many files. Reading everything at once will cause the memory
    #to exceed that of the quota. Therefore, iterate over subsamples of the files.
    for arrays in up.iterate(path='root_files/'+horn+'/fardet*'+swap+'*.root',\
                             treepath="eventtrain/fSlice",branches=mainlist,\
                             entrysteps=nslc):
        SlcE = arrays[b'SliceEnergy'] #Calorimetric energy of all hits in the slice
        numslc = SlcE.shape[0]
        if (limit < (idx+1)*frac) & (limit>=(idx)*frac): #Don't grab too much data
            E += SlcE.tolist()
            Pdg  += arrays[b'SliceTrueNuPdg'].tolist()
            CCNC += arrays[b'SliceTrueNuCCNC'].tolist()
            #Create Nxfeatures array for each of 7 reconstructed prongs.
            #If fewer than 7 prongs, return 0 array for remaining prongs
            x0 = np.asarray(arrays[b'Shw0']).reshape(numslc,numfeatures)
            x1 = np.asarray(arrays[b'Shw1']).reshape(numslc,numfeatures)
            x2 = np.asarray(arrays[b'Shw2']).reshape(numslc,numfeatures)
            x3 = np.asarray(arrays[b'Shw3']).reshape(numslc,numfeatures)
            x4 = np.asarray(arrays[b'Shw4']).reshape(numslc,numfeatures)
            x5 = np.asarray(arrays[b'Shw5']).reshape(numslc,numfeatures)
            x6 = np.asarray(arrays[b'Shw6']).reshape(numslc,numfeatures)
            #Manually normalize the prong length by 100 to make feature space more compact
            x0[:,1] = x0[:,1]/100
            x1[:,1] = x1[:,1]/100
            x2[:,1] = x2[:,1]/100
            x3[:,1] = x3[:,1]/100
            x4[:,1] = x4[:,1]/100
            x5[:,1] = x5[:,1]/100
            x6[:,1] = x6[:,1]/100
            X0 += x0.tolist()
            X1 += x1.tolist()
            X2 += x2.tolist()
            X3 += x3.tolist()
            X4 += x4.tolist()
            X5 += x5.tolist()
            X6 += x6.tolist()
        else: break
        limit += SlcE.shape[0]
    #E is slice level, not shower level
    E = np.asarray(E)
    #Make 3D to cooperate with LSTM dimensional requirement
    X0 = np.asarray(X0).reshape(E.shape[0],1,numfeatures)
    X1 = np.asarray(X1).reshape(E.shape[0],1,numfeatures)
    X2 = np.asarray(X2).reshape(E.shape[0],1,numfeatures)
    X3 = np.asarray(X3).reshape(E.shape[0],1,numfeatures)
    X4 = np.asarray(X4).reshape(E.shape[0],1,numfeatures)
    X5 = np.asarray(X5).reshape(E.shape[0],1,numfeatures)
    X6 = np.asarray(X6).reshape(E.shape[0],1,numfeatures)
    #Create an input array with dimensions Nx7xfeatures
    X = np.concatenate([X0,X1,X2,X3,X4,X5,X6],axis=1)
    X = np.delete(X,18,axis=2)
    Pdg = np.asarray(Pdg)
    CCNC = np.asarray(CCNC)
    Y = np.int_(np.logical_and(Pdg==-12,CCNC==0))
    return X,E,Y,limit

numfeatures = 29
numfs = 4000000
#Create LSTM model with prong level inputs
Shw  = Input(shape=(7,numfeatures-1),dtype=np.float32,name='Shw')
SlcE = Input(shape=(1,),dtype=np.float32,name='SlcE')
#16 neurons in each LSTM unit
lstm = LSTM(16,dropout=0.,recurrent_dropout=0.,name='lstm',return_sequences=False)(Shw)
#Concatenate the output of the LSTM with the Slice Energy
conc = concatenate([lstm,SlcE],name='conc')
#Feed this into sigmoid function
final = Dense(1,activation='sigmoid',name='final',kernel_regularizer=l2(0.))(conc)
#Create 4 models distinguished by the fraction of the energy of the leading prong
#over the total slice energy. This can speed up training because the highest fraction events 
#should be the easiest to classify. Once convergence of particular model is achieved,
#more time and resources can go to training the more difficult models.
model0 = Model(inputs=[Shw,SlcE],outputs=final) #pngE/slcE < 0.25 
model1 = Model(inputs=[Shw,SlcE],outputs=final) #0.25 <= pngE/slcE < 0.5
model2 = Model(inputs=[Shw,SlcE],outputs=final) #0.5 <= pngE/slcE < 0.75
model3 = Model(inputs=[Shw,SlcE],outputs=final) #0.75 <= pngE/slcE

if os.path.exists("save/FHC_LSTM_Model0.h5"):
    model0 = load_model("save/FHC_LSTM_Model0.h5")
if os.path.exists("save/FHC_LSTM_Model1.h5"):
    model1 = load_model("save/FHC_LSTM_Model1.h5")
if os.path.exists("save/FHC_LSTM_Model2.h5"):
    model2 = load_model("save/FHC_LSTM_Model2.h5")
if os.path.exists("save/FHC_LSTM_Model3.h5"):
    model3 = load_model("save/FHC_LSTM_Model3.h5")

nepoch = 100
frac = numfs/nepoch
bestloss0 = 999.
bestloss1 = 999.
bestloss2 = 999.
bestloss3 = 999.
train_loss0 = []
val_loss0 = []
train_loss1 = []
val_loss1 = []
train_loss2 = []
val_loss2 = []
train_loss2 = []
val_loss2 = []
train_loss3 = []
val_loss3 = []
train_loss3 = []
val_loss3 = []
FNum,NNum,TNum = 0,0,0
lr = 0.005
for i in range(0,nepoch):
    print("Inside epoch: "+str(i))
    print("Learning Rate: "+str(lr))
    np.random.seed(int(time.time()))
    #Use gradient clipped RMSprop. Adam better for when training over larger samples at once.
    rms = optimizers.RMSprop(lr=lr,clipvalue=0.5)
    model0.compile(loss='binary_crossentropy',optimizer=rms)
    model1.compile(loss='binary_crossentropy',optimizer=rms)
    model2.compile(loss='binary_crossentropy',optimizer=rms)
    model3.compile(loss='binary_crossentropy',optimizer=rms)
    FX,FE,FY,FNum = DataHandle(i,swap='fluxswap',horn='FHC',limit=FNum,nepoch=nepoch,frac=frac)
    NX,NE,NY,NNum = DataHandle(i,swap='nonswap',horn='FHC',limit=NNum,nepoch=nepoch,frac=frac)
    TX,TE,TY,TNum = DataHandle(i,swap='tau',horn='FHC',limit=TNum,nepoch=nepoch,frac=frac)

    X = np.concatenate([FX,NX,TX],axis=0)
    E = np.concatenate([FE,NE,TE],axis=0)
    Y = np.concatenate([FY,NY,TY],axis=0)
    X,E,Y = shuffle(X,E,Y)
    X0 = X[(X[:,0,0]/E)<0.25]
    Y0 = Y[(X[:,0,0]/E)<0.25]
    E0 = E[(X[:,0,0]/E)<0.25]
    X1 = X[(X[:,0,0]/E>=0.25)&(X[:,0,0]/E<0.5)]
    Y1 = Y[(X[:,0,0]/E>=0.25)&(X[:,0,0]/E<0.5)]
    E1 = E[(X[:,0,0]/E>=0.25)&(X[:,0,0]/E<0.5)]
    X2 = X[(X[:,0,0]/E>=0.5)&(X[:,0,0]/E<0.75)]
    Y2 = Y[(X[:,0,0]/E>=0.5)&(X[:,0,0]/E<0.75)]
    E2 = E[(X[:,0,0]/E>=0.5)&(X[:,0,0]/E<0.75)]
    X3 = X[(X[:,0,0]/E)>=0.75]
    Y3 = Y[(X[:,0,0]/E)>=0.75]
    E3 = E[(X[:,0,0]/E)>=0.75]

    fitter = model0.fit([X0,E0],Y0,validation_split=0.3,class_weight='auto',\
                         epochs=100,callbacks=[EarlyStopping(monitor='loss',min_delta=0.0005)])
    model0.save('FHC_LSTM_Model0.h5')
    trainloss = fitter.history['loss']
    valloss   = fitter.history['val_loss']
    trainloss = sum(trainloss)/len(trainloss)
    valloss = sum(valloss)/len(valloss)
    train_loss0.append(trainloss)
    val_loss0.append(valloss)
    print('Current Training Loss0: '+str(trainloss))
    print('Current Validation Loss0: '+str(valloss))
    plt.figure(1)
    plt.title('Model0 Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(np.asarray(train_loss0),label='Train')
    plt.plot(np.asarray(val_loss0),label='Cross-Validation')
    plt.savefig(out_png0)
    fitter = model1.fit([X1,E1],Y1,validation_split=0.3,class_weight='auto',\
                         epochs=100,callbacks=[EarlyStopping(monitor='loss',min_delta=0.0005)])
    model1.save('FHC_LSTM_Model1.h5')
    trainloss = fitter.history['loss']
    valloss   = fitter.history['val_loss']
    trainloss = sum(trainloss)/len(trainloss)
    valloss = sum(valloss)/len(valloss)
    train_loss1.append(trainloss)
    val_loss1.append(valloss)
    print('Current Training Loss1: '+str(trainloss))
    print('Current Validation Loss1: '+str(valloss))
    plt.figure(2)
    plt.title('Model1 Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(np.asarray(train_loss1),label='Train')
    plt.plot(np.asarray(val_loss1),label='Cross-Validation')
    plt.savefig(out_png1)
    fitter = model2.fit([X2,E2],Y2,validation_split=0.3,class_weight='auto',\
                         epochs=100,callbacks=[EarlyStopping(monitor='loss',min_delta=0.0005)])
    model2.save('FHC_LSTM_Model2.h5')
    trainloss = fitter.history['loss']
    valloss   = fitter.history['val_loss']
    trainloss = sum(trainloss)/len(trainloss)
    valloss = sum(valloss)/len(valloss)
    train_loss2.append(trainloss)
    val_loss2.append(valloss)
    print('Current Training Loss2: '+str(trainloss))
    print('Current Validation Loss2: '+str(valloss))
    plt.figure(3)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(np.asarray(train_loss2),label='Train')
    plt.plot(np.asarray(val_loss2),label='Cross-Validation')
    plt.savefig(out_png2)
    fitter = model3.fit([X3,E3],Y3,validation_split=0.3,class_weight='auto',\
                         epochs=100,callbacks=[EarlyStopping(monitor='loss',min_delta=0.0005)])
    model3.save('FHC_LSTM_Model3.h5')
    trainloss = fitter.history['loss']
    valloss   = fitter.history['val_loss']
    trainloss = sum(trainloss)/len(trainloss)
    valloss = sum(valloss)/len(valloss)
    train_loss3.append(trainloss)
    val_loss3.append(valloss)
    print('Current Training Loss3: '+str(trainloss))
    print('Current Validation Loss3: '+str(valloss))
    plt.figure(4)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(np.asarray(train_loss3),label='Train')
    plt.plot(np.asarray(val_loss3),label='Cross-Validation')
    plt.savefig(out_png3)  
   
    #Oh no! It looks like I accidentally deleted the rest of the code.
    #The remaining piece of code manually decreased the learning rate for a particular model,
    #and provided control over when to stop training a particular model, based on the change of
    #average validation loss.
    if i>0: lr = 0.005/i
