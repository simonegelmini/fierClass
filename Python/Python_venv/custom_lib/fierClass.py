#   Authors: Jessica Leoni (jessica.leoni@polimi.it)
#            Francesco Zinnari (francesco.zinnari@polimi.it)
#            Simone Gelmini (simone.gelmini@polimi.it, gelminisimon@gmail.com)
#   Date: 2019/05/03.
#
#   If you are going to use fierClass in your research project, please cite its reference article
#   S. Gelmini, S. Formentin, et al. "fierClass: A multi-signal, cepstrum-based, time series classifier,"
#   Engineering Applications of Artificial Intelligence, Volume 87, 2020, https://doi.org/10.1016/j.engappai.2019.103262.
#
#   Copyright and license: ? Simone Gelmini, Politecnico di Milano
#   Licensed under the [MIT License](LICENSE).
#
#  In case of need, feel free to ask the author.

#########################################################################################################################################
#                                                               Libraries                                                               #
#########################################################################################################################################
import pandas as pd
import math
from math import ceil,floor
import numpy as np
from numpy.fft import fft,ifft
from numpy import zeros,log
from sklearn.utils import check_X_y, check_array
from tqdm import tqdm


#########################################################################################################################################
#                                                               fierClass                                                               #
#########################################################################################################################################
class fierClass:
    def __init__(self,fs,window_length_seconds, windowing=1,selectedSignals=None,overlap = 0):
        """
        fierClass structure initialization
        
        """
        fierClass.init_consistency(fs,window_length_seconds,windowing,selectedSignals,overlap)
        self.fs = fs
        self.window_length_seconds = window_length_seconds
        self.windowing = 1
        self.selectedSignals=selectedSignals
        self.overlap = overlap
        self.window_length_samples = ceil(window_length_seconds*fs)
        self.fitted = False
        self.subclasses = False
        self.y_train_sub = None
        
    def fit(self,X_train,y_train,y_train_sub=None,subclasses=False):
        """
        Compute the reference cepstrum coefficient for each instance in trainData.
        
        Parameters
        ----------
        trainData - It has to be a numpy array of 3-dimension with shape (classes, selectedSignals, samples).
            classes: number of different classes we aim to train the classifier on
            selectedSignals: number of features for each instance. For a univariate dataset, selectedSignals=1.
            samples: number of samples in each signal;
        y_train - Classes target variable. Should be a numpy array of shape (classes,).
        sublcasses - boolean equal to 1 if subclasses labels are available, 0 otherwise (default 0);
        y_train_sub - Subclasses target variable. Should be a numpy array of shape (classes,).
        
        """
        
        self.subclasses = subclasses
        
        ### Consistency checks
        fierClass.train_consistency(X_train,y_train,y_train_sub,
                                    self.fs,self.window_length_samples,self.windowing,
                                    self.selectedSignals,self.overlap,self.subclasses)
        self.number_signals = X_train.shape[1]
        
        ### Classifier training
        self.C_train = fierClass.compute_cepstrum(X_train,self.window_length_samples,self.windowing,
                                                  self.overlap)
        self.y_train = np.unique(y_train)
        self.fitted = True
        if(subclasses):
            self.y_train_sub = y_train_sub
        
    def fierClassClassify(self,X_test,order=None,selectedSignals=None,featurefusion=None):
        """
        Attributes each signal window to the reference whose spectrum is more closer according to their Martin distance.
        
        Input Parameters
        ----------
        testData - It has to be a numpy array of 3-dimension with shape (1, selectedSignals, samples).
            selectedSignals: number of features in the acquired signal. It must be equal to trainData one.
            samples: number of samples in each signal; it must be greater than windowLength*fs;
        order -  maximum order of the cepstrum used to compute the cepstral distance;
        selectedSignals - array of features' indexes to consider (default all);
        featurefusion - 'conv' or None defines whether to use a single or multiclass fierClass prediction (default None).
        
        Output Parameters
        ----------
        y_pred - array of class label predicted
        y_pred_sub - optional. Array of subclass label predicted. It is returned just if self.sublcasses is True.
        
        """
        
        ### Consistency checks
        if(self.fitted==False):
            raise Exception("Classifier has not been fitted yet, use method fit(X_train,y_train) first")
        
        if(not(order)):
            order = self.window_length_samples
            
        fierClass.test_consistency(X_test,order,self.window_length_samples,self.number_signals,self.selectedSignals)
        self.order = order
        
        ### Selected signals selection
        if(selectedSignals):
            C_train = self.C_train[:,self.selectedSignals,:]
            signal_index_min = min(self.selectedSignals)
            signal_index_max = max(self.selectedSignals)
        else:
            C_train = self.C_train[:,:,:]
            signal_index_min = 0
            signal_index_max = X_test.shape[1] - 1
            
        ### Prediction
        init = self.window_length_samples
        prediction = []
        classes = self.C_train.shape[0]
        
        # Output vectors initialization
        if(featurefusion!=str('conv')):
            y_pred = np.ones([X_test.shape[1], X_test.shape[2]])
            y_pred_sub = np.ones([X_test.shape[1], X_test.shape[2]])
        else:
            y_pred = np.ones(X_test.shape[2])
            y_pred_sub = np.ones(X_test.shape[2])
            
        # For each window the cepstrum is computed. Then are calculated the Martin distance
        # between it and all the reference cepstra computed during the training phase.
        # The label predicted is the same of the closer reference cepstra.
        for k in tqdm(range(init, X_test.shape[2]), desc='Prediction'):
            # Cepstrum
            self.C_test = fierClass.compute_cepstrum(X_test[:,:,k-init:k],self.window_length_samples,self.windowing,
                                             self.overlap,test=1)
            if(featurefusion!=str('conv')):
                for i in range(X_test.shape[1]):
                        for n in range(classes):
                            # Martin Distance
                            distance = fierClass.cepstraldistance(np.squeeze(self.C_test[0,i,:]),
                                                                np.squeeze(self.C_train[n,i,:]),self.order)
                            if(n==0):
                                min_distance = distance
                                prediction = n
                            elif(distance<min_distance):
                                min_distance = distance
                                prediction = n
                        y_pred[i,k] = prediction
                        if(self.subclasses):
                            y_pred_sub[i,k] = self.y_train_sub[int(prediction)]
            else:
                C_test = self.C_test.sum(axis=1,keepdims=True)
                C_train = self.C_train.sum(axis=1,keepdims=True)
                for n in range(classes):
                    # Martin Distance
                    distance = fierClass.cepstraldistance(np.squeeze(self.C_test[0,0,:]),
                                                                np.squeeze(self.C_train[n,0,:]),self.order)
                    if(n==0):
                        min_distance = distance
                        prediction = n
                    elif(distance<min_distance):
                        min_distance = distance
                        prediction = n
                y_pred[k] = prediction
                if(self.subclasses):
                    y_pred_sub[k] = self.y_train_sub[int(prediction)]
              
        if(self.subclasses):   
            return y_pred, y_pred_sub
        
        return y_pred
    
    def get_train_cepstra(self):
        """
        Attributes each signal window to the reference whose spectrum is more closer according to their Martin distance.
        
        Output Parameters
        ----------
        train_cesptra - Numpy array of dimension (classes, selectedSignals, cepstrum_samples);
            cepstrum_samples = (windowLength*fs)/2 as it is computed from the signal spectrum.
        
        """
        return(self.C_train)
    
    def get_order(self):
        """
        Automatic cepstrum order computation. It is computed as the maximum cepstra index in which the 
        references coefficients are less closer than a threshold.
        
        Output Parameters
        ----------
        order - Numpy array of dimension (classes, selectedSignals, cepstrum_samples);
            cepstrum_samples = (windowLength*fs)/2 as it is computed from the signal spectrum.
        
        """
        C_train = self.C_train[:,:,:]
        distance = np.zeros([C_train.shape[1], C_train.shape[2]])
        
        # For each reference cepstra, compute the distance.
        for i in range(C_train.shape[0]):
            for j in range(i+1,C_train.shape[0]):
                distance = distance + np.abs(C_train[i,:,:] - C_train[j,:,:])
        threshold = 1e-1
        
        # Set the distances below the threshold to zero
        distance = np.where(distance<threshold, 0, distance)
        
        # Get the maximum index of non-zero distance
        order = np.max(np.where(distance!=0))
        return order
        
    @staticmethod
    def init_consistency(fs,window_length_seconds,windowing,selectedSignals,overlap):
        """
        Basic parameters consistency checks.
        
        """
        if(fs<=0):
            raise Exception('Sampling frequency must be a positive number.')
            
        if(window_length_seconds<=0):
            raise Exception('Window length must be a positive number.')
        
        if(overlap < 0 or overlap > 1):
            raise Exception('Overlap must be within [0,1].')
    
    @staticmethod
    def train_consistency(X_train,y_train,y_train_sub,fs,window_length_samples,
                          windowing,selectedSignals,overlap,subclasses):
        """
        Train consistency checks.
        
        """
        
        # Input data structures check
        check_X_y(X_train,y_train,ensure_2d=False,allow_nd=True)
        if(len(X_train.shape) != 3):
            raise Exception('X_train is ill defined. Input must be a 3d array.')
            
        if(X_train.shape[0] != len(y_train)):
            raise Exception('Training samples number is inconsistent considering provided labels.')
            
        if(subclasses):
            if((len(np.unique(y_train_sub))<2)):
                raise Exception('Subclasses labels not defined.')
        
        # Train parameters consistency checks
        if(window_length_samples > X_train.shape[2]):
            raise Exception('Window is larger than the time series.')
            
        if(windowing != 1):
            if(len(windowing) != window_length_samples):
                raise Exception('Window weights should have the same length as the window')
        
        if(selectedSignals):
            if(not(np.all(np.isin(selectedSignals,range(0,X_train.shape[1]))))):
                raise Exception('One or more of the selected signals are not available.')
        
    @staticmethod
    def test_consistency(X_test,order,window_length_samples,number_signals,selectedSignals):
        """
        Test parameters consistency checks.
        
        """
        
        # Input data structures check
        check_array(X_test,ensure_2d=False,allow_nd=True)
        
        if(len(X_test.shape) != 3):
            raise Exception('X_test is ill defined. Input must be a 3d array.')
        
        if(X_test.shape[1]!=number_signals):
            raise Exception('Incompatible number of signals: should be {} but is {}'.format(number_signals,X_test.shape[1]))
        
        # Test parameters consistency checks
        if(order < 1 or order > window_length_samples):
            raise Exception('Coefficient of cepstrum must be within [1,{}]'.format(window_length_samples))

        if(selectedSignals):
            if(not(np.all(np.isin(selectedSignals,range(0,X_test.shape[1]))))):
                raise Exception('One or more of the selected signals are not available.')
                
    
    @staticmethod  
    def compute_cepstrum(X,window_length_samples,windowing,overlap,test=None):
        """
        This function compute the cepstrum for all the instances contained in the input matrix given.
        
        Input Parameters
        ----------
        X - It has to be a numpy array of 3-dimension with shape (instances, selectedSignals, samples).
            instaces: number of instances in the dataset. If X = trainData, instances = classes.
            selectedSignals: number of features in the acquired signal. It must be equal to trainData one.
            samples: number of samples in each signal; it must be greater than windowLength*fs;
        window_length_samples -  windowLength*fs;
        windowing - array of window weights' coefficients (default 1, rectangular window);
        overlap - ratio of overlapped samples between consecutive windows. 0 -> no overlap, 1 -> full overlap (default 0);
        test - True if the the function is called by the prediction function. Optional (default None).
        
        Output Parameters
        ----------
        C - cepstra numpy matrix of the shape (instances, selectedSignals, cepstrum_samples);
            cepstrum_samples = (windowLength*fs)/2 as it is computed from the signal spectrum.
        
        """
        
        # Initialize C matrix as a zeor matrix with shape (instances, selectedSignals, cepstrum_samples)
        C = np.zeros((X.shape[0],X.shape[1],ceil(window_length_samples/2)))
        
        # Compute the cepstrum for each instance contained in X
        for i in range(0,X.shape[0]):
            C[i] = fierClass.multi_signal_cepstrum(X[i],window_length_samples,
                                                   windowing,overlap,test)
        return C
        
        
    @staticmethod    
    def multi_signal_cepstrum(xi,window_length_samples,windowing,overlap,test):
        """
        This function compute the cepstrum for each instance contained in a multisignal instance.
        
        Input Parameters
        ----------
        xi - It has to be a numpy array of 3-dimension with shape (1, selectedSignals, samples).
            selectedSignals: number of features in the acquired signal. It must be equal to trainData one.
            samples: number of samples in each signal; it must be greater than windowLength*fs;
        window_length_samples -  windowLength*fs;
        windowing - array of window weights' coefficients (default 1, rectangular window);
        overlap - ratio of overlapped samples between consecutive windows. 0 -> no overlap, 1 -> full overlap (default 0);
        test - True if the the function is called by the prediction function. Optional (default None).
        
        Output Parameters
        ----------
        ci - instance cepstra numpy array of the shape (selectedSignals, cepstrum_samples);
            cepstrum_samples = (windowLength*fs)/2 as it is computed from the signal spectrum.
        
        """
        
        # Initialize C matrix as a zeor matrix with shape (selectedSignals, cepstrum_samples).
        ci = np.zeros((xi.shape[0],ceil(window_length_samples/2)))

        j=0
        # Compute the cepstrum for each signal contained in xi
        for j in range(0,xi.shape[0]):
            ci[j] = fierClass.single_signal_cepstrum(xi[j],window_length_samples,
                                                     windowing,overlap,test)
            j=j+1
        return np.array(ci)
    
    
    @staticmethod
    def single_signal_cepstrum(xij,window_length_samples,windowing,overlap,test):
        """
        This function compute the cepstrum for each signal contained in a multisignal instance.
        
        Input Parameters
        ----------
        xij - It has to be a numpy array of 2-dimension with shape (1, samples).
            samples: number of samples in each signal; it must be greater than windowLength*fs;
        window_length_samples -  windowLength*fs;
        windowing - array of window weights' coefficients (default 1, rectangular window);
        overlap - ratio of overlapped samples between consecutive windows. 0 -> no overlap, 1 -> full overlap (default 0);
        test - True if the the function is called by the prediction function. Optional (default None).
        
        Output Parameters
        ----------
        cij - signal cepstrum numpy array of the shape (cepstrum_samples);
            cepstrum_samples = (windowLength*fs)/2 as it is computed from the signal spectrum.
        
        """
        
        n=window_length_samples 
        if(test):
            x = np.squeeze(np.array(xij))
            
            ### windowing
            xw = x*windowing
            
            ### Spectrum
            spectrum = 1/n * ((abs(fft(xw)))**2)
            
            if (max(spectrum[2:])>1e-6):
                cij = ifft(log(spectrum)).real
            else:
                cij = np.zeros(n)
               
        else:
            N=len(xij) 
            spectrum=np.zeros(n)
            iterations=1

            ### from overlap to stride
            step = max(1,ceil(n*(1-overlap)))

            ### Sliding window
            for i in range(n,N+1,step):
                x = np.array(xij[i-n:i])
                
                ### windowing
                xw = x*windowing

                ### Spectrum
                mag = 1/n * ((abs(fft(xw)))**2)
                spectrum = spectrum+mag
                iterations = iterations+1

            spectrum=spectrum/iterations

            if (max(spectrum[2:])>1e-6):
                cij = ifft(log(spectrum)).real
            else:
                cij = np.zeros(n)

        return cij[:ceil(n/2)]
    
    @staticmethod
    def cepstraldistance(cepstrumCoefficients1,cepstrumCoefficients2,order):
        """
        This function computes the distance between two cepstra according to the Martin Distance.
        
        Input Parameters
        ----------
        cepstrumCoefficients1 - Cepstrum array of size (cepstrum_samples, );
        cepstrumCoefficients2 - Cepstrum array of size (cepstrum_samples, );
        order -  maximum order of the cepstrum used to compute the cepstral distance;
        
        Output Parameters
        ----------
        distance - Martin Distance between the two cepstra
        
        """

        distance=0
        cepstrumCoefficients1 = np.squeeze(cepstrumCoefficients1)[:order]
        cepstrumCoefficients2 = np.squeeze(cepstrumCoefficients2)[:order]
        for i in range(0,order):
            distance = distance + (i+1)*(abs(cepstrumCoefficients1[i]-cepstrumCoefficients2[i]))**2
        distance=math.sqrt(distance)
        return distance