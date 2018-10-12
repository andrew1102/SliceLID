# SliceLID (Under Development)
LSTM Log-Likelihood Based Classifier

# Motivation
One of the main objectives of the NOvA Experiment is to measure the rate of muon to electron flavor oscillation in neutrinos. The signal for this analysis is a charged current interaction of an electron neutrino with the nuclei in the NOvA far detector. This signal is quite small compared to the numerous backgrounds of the analysis. Therefore, a high performing electron neutrino classifier is needed to perform the analysis. The official selector uses a deep convolutional neural network with the name of Convolutional Visual Network (CVN). CVN uses raw detector hits as its inputs to peform an end-to-end classification of a hypothesis neutrino interaction. CVN is a high performing selector, but its long training and inference times adds challenges to the analysis. In addition to this, physical laws are not directly learned (i.e. interaction which violates four momentum conservation could be considered signal). 

For all of these reasons, it was decided that a high performing electron neutrino which directly learns physics would be useful as a physical cross-check for CVN.

#Keras-to-Tensorflow
https://github.com/amir-abdi/keras_to_tensorflow.git
