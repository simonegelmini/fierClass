function [prediction_struct] = fierClassClassify(varargin)
% trainFyclass classify a new data instance based on the cepstral distance
% and a trained fierClass
%
%  Inputs:
%   fierClass - struct of the trained fierClass
%   test_data - struct with all the signals to be tested - must have as
%               many signals as the ones used to train the algorithm
%
%  Parameters (optional):
%   order - {default max cepstrum order} defines the maximum order of the
%       cepstrum used to compute the cepstral distance
%   selected - {default max number of signals} defines which signals should
%       be considered in the prediction
%   featurefusion - {'no','conv','euristic' - default 'no'} defines whether
%       to use a single or multiclass fierClass prediction. Listing all the
%       alternatives in curled brakets in case of multiple types of fusion
%       want to be tested
%   predictsubclass - {'no','yes', - default 'no'} defines whether to
%       predict the output based on the instances or the defined sublcasses
%   w - column vector window coefficients
%
%  Outputs:
%   prediction_struct - predicted struct output
%
%   Author: Simone Gelmini (simone.gelmini@polimi.it, gelminisimon@gmail.com)
%   Date: 2019/05/03.
%   Version: 1.2
%   Note on version 1.1: input management added.
%   Note on version 1.2: windowing added.
%
%   If you are going to use fierClass in your research project, please 
%   cite its reference article - S. Gelmini, S. Formentin, et al.
%   "fierClass: A multi-signal, cepstrum-based, time series classifier,"
%   Engineering Applications of Artificial Intelligence, Volume 87, 2020 
%   https://doi.org/10.1016/j.engappai.2019.103262.
%
%   Copyright and license: ? Simone Gelmini, Politecnico di Milano
%   Licensed under the [MIT License](LICENSE).
%
%  In case of need, feel free to ask the author.

%% Default values
order=1;
selected=1;
feature_fusion='no';
predict_subclass='yes';
no_fusion=1;
conv_fusion=0;
euristic_fusion=0;
w=1; % rectangular window

%% Input management
switch nargin
    case 1
        error('Error #0: incorrect number of arguments.')
    case 2
        fierClass=varargin{1};
        test_data=varargin{2};
    otherwise
        fierClass=varargin{1};
        test_data=varargin{2};
        
        % User selected values
        for i=3:2:nargin
            switch lower(varargin{i})
                case 'order'
                    order=varargin{i+1};
                case 'selected'
                    selected=varargin{i+1};
                case 'featurefusion'
                    feature_fusion=lower(varargin{i+1});
                case 'predictsubclass'
                    predict_subclass=lower(varargin{i+1});
                case 'window'
                    w=varargin{i+1};
                otherwise
                    error(['Error #1: option command not found: ' varargin{i}]);
            end
        end
end

%% Consistency check
% Number of trained instances
N_instances=size(fierClass,2);

% Number of signals for each instance trained
N_signals_instance=size(fierClass(N_instances).signal,2);

% Number of signals for testing
N_signals_test=size(test_data.signal,2);

% Number of signals to be tested
N_signals_selected=length(selected);

% Max signal index to be tested
signal_index_max=max(selected);

% Min signal index to be tested
signal_index_min=min(selected);

% Window length
window_length=fierClass(N_instances).window_length;

% Dataset lengthN_signals_selected
dataset_length=length(test_data.signal(N_signals_test).value);

% Feature fusion
for ff=1:length(feature_fusion)
    switch char(feature_fusion(ff))
        case 'no'
            no_fusion=1;
        case 'conv'
            conv_fusion=1;
        case 'euristic'
            euristic_fusion=1;
        otherwise
            error(['Error #2: incorrect featurefusion option.'])
    end
end

% Consistency checks
if N_signals_instance>signal_index_max
    error('Error #3: incorrect selected feature.');
end

if N_signals_instance~=N_signals_selected
    error('Error #4: unbalanced number of tested signals.');
end

if N_signals_instance~=N_signals_test
    error('Error #5: train and test instances must contain the same '+...
    'number of singals.');
end

if N_signals_selected==1 && (conv_fusion || euristic_fusion)
    warning('Warning #1: feature fusion is not enable with a single signal.')
end

%% Vectors and variables initialization
init=ceil(window_length*test_data.signal(N_signals_test).sampling);
test_data_cepstrum_coefficients=zeros(dataset_length,N_instances,order);
belonging_class=zeros(N_instances,1);

% For each instance, it is defined the class to which it belongs
for n=1:N_instances
    if isempty(fierClass(n).subclass) || strcmpi(predict_subclass,'no')
        belonging_class(n)=n;
    else
        belonging_class(n)=fierClass(n).subclass;
    end
end

if no_fusion
    prediction=nan(dataset_length,N_signals_instance);
    prediction_subclass=nan(dataset_length,N_signals_instance);
    distance=zeros(dataset_length,N_signals_selected,N_instances);
end

if conv_fusion
    prediction_conv=nan(dataset_length,1);
    prediction_conv_subclass=nan(dataset_length,1);
    distance_conv=zeros(dataset_length,N_instances);
end

if euristic_fusion
    prediction_euristic=nan(dataset_length,1);
    prediction_euristic_subclass=nan(dataset_length,1);
    distance_euristic=zeros(dataset_length,N_signals_selected,N_instances);
end

%% Prediction
% Cepstrum coefficients are computed for the entire timeseries
for k=init:dataset_length
    % Cepstrum coefficients are computed only for the selected signals
    for i=1:N_signals_selected
        clear min_distance min_distance_index cepstrum_coefficients
        cepstrum_coefficients=cepstrumClassify(test_data.signal(selected(i)).value(k-init+1:k),w);
        
        if max(isnan(cepstrum_coefficients))
            disp('qui')
        end
        
        % Cepstrum coefficients are compared to all the learned instances
        for n=1:N_instances
            % If timeseries are not fused together, there is prediction one
            % prediction for each timeseries based on the minimum distance
            % for all the instances.
            if no_fusion
                distance(k,i,n)=cepstraldistance(cepstrum_coefficients,fierClass(n).signal(i).cepstrum,order);
                
                if n==1
                    min_distance(i)=distance(k,i,n);
                    min_distance_index(i)=n;
                else
                    if distance(k,i,n)<min_distance
                        min_distance(i)=distance(k,i,n);
                        min_distance_index(i)=n;
                    end
                end
            end
            
            % If timeseries are fused through convolution, there is a single
            % prediction for all the signals belonging to the same instance
            if conv_fusion
                if selected(i)==signal_index_min
                    test_data_cepstrum_coefficients(k,n,:)=cepstrum_coefficients(1:order);
                else
                    test_data_cepstrum_coefficients(k,n,:)=squeeze(test_data_cepstrum_coefficients(k,n,:))+cepstrum_coefficients(1:order);
                end
                
                if selected(i)==signal_index_max
                    distance_conv(k,n)=cepstraldistance(test_data_cepstrum_coefficients(k,n,:),...
                        fierClass(n).convolution,order);
                end
            end
            
            % If timeseries are fused by means of the euristic distance, there is a single
            % prediction for all the signals belonging to the same instance
            if euristic_fusion
                distance_euristic(k,i,n)=cepstraldistance(cepstrum_coefficients,fierClass(n).signal(i).cepstrum,order);
            end
        end
        
        % No fusion prediction
        if no_fusion
            prediction(k,i)=min_distance_index(i);
            prediction_subclass(k,i)=belonging_class(min_distance_index(i));
        end
    end
    
    % Convolution-based prediction
    if conv_fusion
        min_distance_index_conv=find(distance_conv(k,:)==min(distance_conv(k,:)));
        prediction_conv(k)=min_distance_index_conv;
        prediction_conv_subclass(k)=belonging_class(min_distance_index_conv);
    end
    
    % Euristic-based prediction
    if euristic_fusion
        epsilon_max=NaN;
        
        for s=1:N_signals_selected
            epsilon=euristicCepstraldistance(distance_euristic(k,s,:),belonging_class,N_instances);
            if (selected(s)==signal_index_min && N_signals_selected>1) || ...
                    epsilon>epsilon_max || N_signals_selected==1
                epsilon_max=epsilon;
                instance_min=find(distance_euristic(k,s,:)==min(distance_euristic(k,s,:)));
                prediction_euristic(k)=instance_min;
                prediction_euristic_subclass(k)=belonging_class(instance_min);
            end
        end
    end
    
end

%% Output data is structured according to the type of the chosen signal fusion
if no_fusion
    prediction_struct.no_fusion.prediction=prediction;
    prediction_struct.no_fusion.prediction_subclass=prediction_subclass;
    prediction_struct.no_fusion.distance=distance;
end

if conv_fusion
    prediction_struct.conv.prediction=prediction_conv;
    prediction_struct.conv.prediction_subclass=prediction_conv_subclass;
    prediction_struct.conv.distance=distance_conv;
end

if euristic_fusion
    prediction_struct.euristic.prediction=prediction_euristic;
    prediction_struct.euristic.prediction_subclass=prediction_euristic_subclass;
end

end

function C = cepstrumClassify(x,w)
% cepstrumClassify computes the cepstrum coefficients of a signal x
%
%  Inputs:
%   x - signal in the time domain
%   w - column vector window coefficients
%
%  Output:
%   C - real cepstrum
%
%   Author: Simone Gelmini (simone.gelmini@polimi.it, gelminisimon@gmail.com)
%   Based on: Katrien De Cock's Ph.D. Thesis "Principal Angles in system theory, information theory and signal processing"
%   Date: 2019/05/03.
%
%  In case of need, feel free to ask the author.

N=length(x);

% The cepstrum is computed only if the signal is not constant
if var(x)==0
    % If the signal is constant, there is no dynamics and the cepstrum are
    % forced to zero
    C=zeros(1,ceil(N/2));
else
    % The signal is not constant, meaning that there is a dynamics.
    % Cepstrum coefficients can be computed using the definition
    xw=windowing(x,w);
    spectrum=1/N*abs(fft(xw)).^2;
    C=real(ifft(log(abs(spectrum))));
    C=C(1:ceil(N/2));
end

% Cepstrum coefficients vector is always a column vector
C_dim=size(C);
if C_dim(2)>C_dim(1)
    C=C';
end

end

function distance = cepstraldistance(cepstrumCoefficients1,cepstrumCoefficients2,order)
% cepstraldistance evaluates the cepstral distance according to Richard J.
% Martin's "A Metric for ARMA processes", 2000.
%
%   Author: Simone Gelmini (simone.gelmini@polimi.it, gelminisimon@gmail.com)
%   Date: 2018/01/27.
%
%  In case of need, feel free to ask the author.

%% Output initialization
distance=0;

%% distance evaluation
for i=1:order
    distance=distance+i*abs(cepstrumCoefficients1(i)-cepstrumCoefficients2(i))^2;
end

distance=sqrt(distance);

end

function epsilon = euristicCepstraldistance(distance,belonging_class,N_instances)
% euristicCepstraldistance computes the internal distance of the same class.
%
%   Author: Simone Gelmini (simone.gelmini@polimi.it, gelminisimon@gmail.com)
%   Date: 2018/05/28.
%
%  In case of need, feel free to ask the author.

[signal_list,order_list]=sort(squeeze(distance));
belonging_class_ordered=squeeze(belonging_class(order_list));
index1_class=1;
min_distance_class=belonging_class_ordered(index1_class);

notFound=1;
cl=2;
index2_class=1;

while notFound && cl<=N_instances
    if belonging_class_ordered(cl)~=belonging_class_ordered(index1_class)
        index2_class=cl;
        notFound=0;
    end
    cl=cl+1;
end

epsilon=signal_list(index2_class)-signal_list(index1_class);

end

function xw = windowing(x,w)
% windowing evaluates the windowed signals according to the chosen window
%
%  Inputs:
%   x - signal in the time domain
%   w - column vector window coefficients
%
%  Outputs:
%   xw - signal output of the windowing process
%
%   Author: Simone Gelmini (simone.gelmini@polimi.it, gelminisimon@gmail.com)
%   Date: 2019/05/03.
%
%  In case of need, feel free to ask the author.

try
    xw=x.*w;
catch
    error('Error #7: windowing matrix dimensions must agree.');
end

end