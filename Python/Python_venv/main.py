###########################################################################
######################## fierClass example script #########################
###########################################################################
### Authors: Jessica Leoni (jessica.leoni@polimi.it) ######################
############ Francesco Zinnari (francesco.zinnari@polimi.it) ##############
############ Simone Gelmini (gelminisimon@gmail.com) ## Date: 2019/05/03 ##
########################### (simone.gelmini@polimi.it) ####################
###########################################################################
### Note: this script is an introductory example for using the functions ##
### designed according to the fierClass algorithmin. ######################
### In the present example, the algorithm tries to classify data coming ###
### from an Inertial Measurement Unit (IMU), comparing the stream of data #
### with respect to a set of examples provided. The classification is #####
### performed considering two scenarios: ##################################
####### 1) when (example) instances are grouped toghether through #########
####### subclasses; #######################################################
####### 2) when each instance represents an individual operating mode #####
####### that needs to be classified separately with respect to the ########
####### others. ###########################################################
###########################################################################
### Further details on how to use the training and classifying functions ##
### are available through the help. #######################################
###########################################################################
### If you are going to use fierClass in your research project, please ####
### cite its reference article - S. Gelmini, S. Formentin, et al. #########
### "fierClass: A multi-signal, cepstrum-based, time series classifier," ##
### Engineering Applications of Artificial Intelligence, Volume 87, 2020 ##
### https://doi.org/10.1016/j.engappai.2019.103262. #######################
###########################################################################
### Copyright and license: ? Jessica Leoni, Francesco Zinnari #############
### Simone Gelmini, Politecnico di Milano #################################
###########################################################################
### Licensed under the [MIT License](LICENSE). ############################
###########################################################################
########## In case of help, feel free to contact the author ###############
###########################################################################

###########################################################################
#                               Libraries                                 #
###########################################################################
import numpy as np
from custom_lib.visualization import *
from custom_lib.fierClass import *
from custom_lib.functions import*
import warnings
warnings.filterwarnings("ignore")


###########################################################################
#                           Simulation Params                             #
###########################################################################
timespan=100 #[s]
fs=100 #[Hz]
samples=timespan*fs #[-] recorded samples
np.random.seed(1)
e=np.random.randn(samples, 1)


###########################################################################
#                       Train Data Simulation                             #
###########################################################################
# TrainData structure generation
# Generate data for subclass 1 - instance 1
ax11=lowpass(1,1,1,0.8, fs, e)
ay11=lowpass(1,1,1,0.4, fs, e)
az11=lowpass(1,1,1,0.6, fs, e)
gx11=lowpass(1,1,1,-0.2, fs, e)
gy11=lowpass(1,1,-1,0.3, fs, e)
gz11=lowpass(1,1,1,-0.1, fs, e)
Ts11=1/fs #[s]
label11=0
sub_label11=0

# Generate data for subclass 2 - instance 1
ax21=lowpass(1,0.3,1,0.1, fs, e)
ay21=lowpass(1,-0.8,1,0.1, fs, e)
az21=lowpass(20,1,1,0.1, fs, e)
gx21=lowpass(-44,1,1,-0.1, fs, e)
gy21=lowpass(7,1,-1,0.1, fs, e)
gz21=lowpass(1,0.02,1,-0.7, fs, e)
Ts21=1/fs #[s]
label21=1
sub_label21=1

# Generate data for subclass 2 - instance 2
ax22=lowpass(1,0.3,1,0.1, fs, 2*e)
ay22=lowpass(1,-0.8,1,0.1, fs, 2 *e)
az22=lowpass(20,1,1,0.1, fs, 2 *e)
gx22=lowpass(-44,1,1,-0.1, fs, 2*e)
gy22=lowpass(7,1,-1,0.1, fs, 2 *e)
gz22=lowpass(1,0.02,1,-0.7, fs, 2*e)
Ts22=1/fs #[s]
label22=2
sub_label22=1

# trainData size (classes, signals, samples)
# y_train size (classes,)
# y_train_sub size (classes,)
trainData = np.zeros([3, 6, len(ax11)])
y_train = np.zeros(3)
y_train_sub = np.zeros(3)

trainData[0,0,:]=ax11
trainData[0,1,:]=ay11
trainData[0,2,:]=az11
trainData[0,3,:]=gx11
trainData[0,4,:]=gy11
trainData[0,5,:]=gz11
y_train[0]=label11
y_train_sub[0]=sub_label11

trainData[1,0,:]=ax21
trainData[1,1,:]=ay21
trainData[1,2,:]=az21
trainData[1,3,:]=gx21
trainData[1,4,:]=gy21
trainData[1,5,:]=gz21
y_train[1]=label21
y_train_sub[1]=sub_label21

trainData[2,0,:]=ax22
trainData[2,1,:]=ay22
trainData[2,2,:]=az22
trainData[2,3,:]=gx22
trainData[2,4,:]=gy22
trainData[2,5,:]=gz22
y_train[2]=label22
y_train_sub[2]=sub_label22


###########################################################################
#                           Test Data Simulation                          #
###########################################################################
# Test data structures shoul be of size (1, signals, samples).
# Therefore we concatenate all the instances signal by signal.
# Please notice that it is just an example. In real applications 
#test data should be different from training ones.
ax_con=np.concatenate((ax11,ax11,ax21,ax22,ax21,
	ax11,ax21,ax11))
ay_con=np.concatenate((ay11,ay11,ay21,ay22,ay21,
	ay11,ay21,ay11))
az_con=np.concatenate((az11,az11,az21,az22,az21,
	az11,az21,az11))

gx_con=np.concatenate((gx11,gx11,gx21,gx22,gx21,
	gx11,gx21,gx11))
gy_con=np.concatenate((gy11,gy11,gy21,gy22,gy21,
	gy11,gy21,gy11))
gz_con=np.concatenate((gz11,gz11,gz21,gz22,gz21,
	gz11,gz21,gz11))

# Test labels generation
labels_con=np.concatenate((np.zeros([1,len(ax11)]), 
	np.zeros([1,len(ax11)]), np.ones([1,len(ax21)]), 
	np.ones([1,len(ax22)])*2, np.ones([1,len(ax21)]), 
    np.zeros([1,len(ax11)]), np.ones([1,len(ax21)]), 
    np.zeros([1,len(ax11)])), axis=1)
labels_con_sub=np.concatenate((np.zeros([1,len(ax11)]), 
	np.zeros([1,len(ax11)]), np.ones([1,len(ax21)]), 
	np.ones([1,len(ax22)]), np.ones([1,len(ax21)]), 
    np.zeros([1,len(ax11)]), np.ones([1,len(ax21)]), 
    np.zeros([1,len(ax11)])), axis=1)

# testData size (1, signals, samples)
# y_train size (samples,)
# y_train_sub size (samples,)
testData = np.zeros([1, 6, len(ax_con)])
y_test = np.squeeze(labels_con)
y_test_sub = np.squeeze(labels_con_sub)

testData[0,0,:]=ax_con;
testData[0,1,:]=ay_con;
testData[0,2,:]=az_con;
testData[0,3,:]=gx_con;
testData[0,4,:]=gy_con;
testData[0,5,:]=gz_con;


###########################################################################
#                                   Train                                 #
###########################################################################
# Set training parameters
windowLength=10 #[s]
n_signal=testData.shape[1]-1
windowing=1
overlap=0

# fierClass structure initialization
fc=fierClass(fs,windowLength,windowing,n_signal,overlap)

# fierClass training
fc.fit(trainData,y_train,y_train_sub=y_train_sub,
	 							subclasses=1)

# Automatic order calibration
order=fc.get_order() #[-]

# Cepstra matrix plot
label = ['Class 0', 'Class 1, Subclass 1', 
	'Class 1, Sublcass 2']
train_cepstra_plot(order, fc.get_train_cepstra(), label)


###########################################################################
#                                   Test                                  #
###########################################################################
# Single vs multiclass option
featurefusion = 'conv'

# Prediction
y_pred, y_pred_sub = fc.fierClassClassify(testData,
	order,None,featurefusion)


###########################################################################
#                           Results Evaluation                            #
###########################################################################
# Predictions vs Actual Classes
results_plot(y_test,y_pred,'Classes Labels Evaluation',
        featurefusion)
if(fc.subclasses):
    # Prediction vs Actual Subclasses
    results_plot(y_test_sub, y_pred_sub,
        'SubClasses Labels Evaluation',featurefusion)