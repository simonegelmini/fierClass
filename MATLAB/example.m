%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% fierClass example script %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Author: Simone Gelmini (gelminisimon@gmail.com) %%% Date: 2019/05/03 %%
%%%%%%%%%%%%%%%%%%%%%%%%%% (simone.gelmini@polimi.it) %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Note: this script is an introductory example for using the functions %%
%%% designed according to the fierClass algorithmin which the algorithm. %%
%%% In the present example, the algorithm tries to classify data coming %%%
%%% from an Inertial Measurement Unit (IMU), comparing the stream of data %
%%% with respect to a set of examples provided. The classification is %%%%%
%%% performed considering two scenarios: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% 1) when (example) instances are grouped toghether through %%%%%%%%%
%%%%%%% subclasses; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% 2) when each instance represents an individual operating mode %%%%%
%%%%%%% that needs to be classified separately with respect to the %%%%%%%%
%%%%%%% others. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Further details on how to use the training and classifying functions %%
%%% are available through the help. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% If you are going to use fierClass in your research project, please %%%%
%%% cite its reference article - S. Gelmini, S. Formentin, et al. %%%%%%%%%
%%% "fierClass: A multi-signal, cepstrum-based, time series classifier," %%
%%% Engineering Applications of Artificial Intelligence, Volume 87, 2020 %%
%%% https://doi.org/10.1016/j.engappai.2019.103262. %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Copyright and license: ? Simone Gelmini, Politecnico di Milano %%%%%%%%
%%% Licensed under the [MIT License](LICENSE). %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% In case of help, feel free to contact the author %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%% Add library
addpath('../')

%% Simulation parameters
timespan=100; %[s]
fs=100; %[Hz]
samples=timespan*fs; %[-] recorded samples
rng(1)
e=randn([samples 1]);

%% Generate data for subclass 1 - instance 1
ax11=filter(1,[1 0.8],e);
ay11=filter(1,[1 0.4],e);
az11=filter(1,[1 0.6],e);

gx11=filter(1,[1 -0.2],e);
gy11=filter(1,[-1 0.3],e);
gz11=filter(1,[1 -0.1],e);
Ts11=1/fs; %[s]

label11='mode 1 experiment 1';

%% Generate data for subclass 2 - instance 1
ax21=filter([1 0.3],[1 0.1],e);
ay21=filter([1 -0.8],[1 0.1],e);
az21=filter([20 1],[1 0.1],e);

gx21=filter([-44 1],[1 -0.1],e);
gy21=filter([7 1],[-1 0.1],e);
gz21=filter([1 0.02],[1 -0.7],e);
Ts21=1/fs; %[s]

label21='mode 2 experiment 1';

%% Generate data for subclass 2 - instance 2
ax22=filter([1 0.3],[1 0.1],2*e);
ay22=filter([1 -0.8],[1 0.1],2*e);
az22=filter([20 1],[1 0.1],2*e);

gx22=filter([-44 1],[1 -0.1],2*e);
gy22=filter([7 1],[-1 0.1],2*e);
gz22=filter([1 0.02],[1 -0.7],2*e);
Ts22=1/fs; %[s]

label22='mode 2 experiment 2';

%% Prepare data for training
trainData(1).signal(1).value=ax11;
trainData(1).signal(1).sampling=1/(Ts11);
trainData(1).signal(1).label='ax';
trainData(1).signal(2).value=ay11;
trainData(1).signal(2).sampling=1/(Ts11);
trainData(1).signal(2).label='ay';
trainData(1).signal(3).value=az11;
trainData(1).signal(3).sampling=1/(Ts11);
trainData(1).signal(3).label='az';
trainData(1).signal(4).value=gx11;
trainData(1).signal(4).sampling=1/(Ts11);
trainData(1).signal(4).label='gx';
trainData(1).signal(5).value=gy11;
trainData(1).signal(5).sampling=1/(Ts11);
trainData(1).signal(5).label='gy';
trainData(1).signal(6).value=gz11;
trainData(1).signal(6).sampling=1/(Ts11);
trainData(1).signal(6).label='gz';
trainData(1).label=label11;
trainData(1).subclass=0;

trainData(2).signal(1).value=ax21;
trainData(2).signal(1).sampling=1/(Ts21);
trainData(2).signal(1).label='ax';
trainData(2).signal(2).value=ay21;
trainData(2).signal(2).sampling=1/(Ts21);
trainData(2).signal(2).label='ay';
trainData(2).signal(3).value=az21;
trainData(2).signal(3).sampling=1/(Ts21);
trainData(2).signal(3).label='az';
trainData(2).signal(4).value=gx21;
trainData(2).signal(4).sampling=1/(Ts21);
trainData(2).signal(4).label='gx';
trainData(2).signal(5).value=gy21;
trainData(2).signal(5).sampling=1/(Ts21);
trainData(2).signal(5).label='gy';
trainData(2).signal(6).value=gz21;
trainData(2).signal(6).sampling=1/(Ts21);
trainData(2).signal(6).label='gz';
trainData(2).label=label21;
trainData(2).subclass=1;

trainData(3).signal(1).value=ax22;
trainData(3).signal(1).sampling=1/(Ts22);
trainData(3).signal(1).label='ax';
trainData(3).signal(2).value=ay22;
trainData(3).signal(2).sampling=1/(Ts22);
trainData(3).signal(2).label='ay';
trainData(3).signal(3).value=az22;
trainData(3).signal(3).sampling=1/(Ts22);
trainData(3).signal(3).label='az';
trainData(3).signal(4).value=gx22;
trainData(3).signal(4).sampling=1/(Ts22);
trainData(3).signal(4).label='gx';
trainData(3).signal(5).value=gy22;
trainData(3).signal(5).sampling=1/(Ts22);
trainData(3).signal(5).label='gy';
trainData(3).signal(6).value=gz22;
trainData(3).signal(6).sampling=1/(Ts22);
trainData(3).signal(6).label='gz';
trainData(3).label=label22;
trainData(3).subclass=1;

train_matrix = zeros([3,6,10000]);
train_matrix(1,:,:) = [ax11 ay11 az11 gx11 gy11 gz11]';
train_matrix(2,:,:) = [ax21 ay21 az21 gx21 gy21 gz21]';
train_matrix(3,:,:) = [ax22 ay22 az22 gx22 gy22 gz22]';

writematrix(train_matrix,'trainData.csv') 

%% Generate test data
ax_con=[ax11;ax11;ax21;ax22;ax21;ax11;ax21;ax11];
ay_con=[ay11;ay11;ay21;ay22;ay21;ay11;ay21;ay11];
az_con=[az11;az11;az21;az22;az21;az11;az21;az11];

gx_con=[gx11;gx11;gx21;gx22;gx21;gx11;gx21;gx11];
gy_con=[gy11;gy11;gy21;gy22;gy21;gy11;gy21;gy11];
gz_con=[gz11;gz11;gz21;gz22;gz21;gz11;gz21;gz11];

labels_con=[zeros([1,length(ax11)]) zeros([1,length(ax11)]) ones([1,length(ax21)])...
    ones([1,length(ax22)]) ones([1,length(ax21)]) zeros([1,length(ax11)])...
    ones([1,length(ax21)]) zeros([1,length(ax11)])];

labels_instance_con=[ones([1,length(ax11)]) ones([1,length(ax11)]) 2*ones([1,length(ax21)])...
    3*ones([1,length(ax22)]) 2*ones([1,length(ax21)]) ones([1,length(ax11)])...
    2*ones([1,length(ax21)]) ones([1,length(ax11)])];

%% Prepare data for classification
testData.signal(1).value=ax_con;
testData.signal(1).sampling=fs;
testData.signal(1).label='ax';
testData.signal(2).value=ay_con;
testData.signal(2).sampling=fs;
testData.signal(2).label='ay';
testData.signal(3).value=az_con;
testData.signal(3).sampling=fs;
testData.signal(3).label='az';
testData.signal(4).value=gx_con;
testData.signal(4).sampling=fs;
testData.signal(4).label='gx';
testData.signal(5).value=gy_con;
testData.signal(5).sampling=fs;
testData.signal(5).label='gy';
testData.signal(6).value=gz_con;
testData.signal(6).sampling=fs;
testData.signal(6).label='gz';
testData.label=labels_con;

test_matrix = [ax_con ay_con az_con gx_con gy_con gz_con]';
writematrix(test_matrix,'testData.csv') 

%% Classifier's parameters
windowLength=10; %[s]
n_signal=max(size(testData.signal));
selected=1:n_signal;

%% Train
[fierClass,order]=fierClassTrain(trainData,windowLength,'showplot','yes','selected',selected);

%% Prediction
prediction_struct=...
    fierClassClassify(fierClass,testData,'order',order,...
    'selected',selected,'featurefusion',{'no','conv','euristic'});

%% Plot
figure
hold on
stairs(labels_con,'linewidth',1.2)
stairs(prediction_struct.conv.prediction_subclass,'--','linewidth',1.2)
legend('Label','Prediction')
title('Subclass-based classification')
xlabel('Samples [-]')
ylim([-.1 1.1])
xlim([1 length(labels_con)])
set(gca,'fontsize',12,'ytick',[0 1],'yticklabel',{'Subclass 0', 'Subclass 1'})
box on

figure
hold on
stairs(labels_instance_con,'linewidth',1.2)
stairs(prediction_struct.conv.prediction,'--','linewidth',1.2)
legend('Label','Prediction')
title('Instance-based classification')
xlabel('Samples [-]')
ylim([0.9 3.1])
xlim([1 length(labels_con)])
set(gca,'fontsize',12,'ytick',[1 2 3],'yticklabel',{label11 label21 label22})
box on