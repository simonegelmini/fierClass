function [fierClass,order] = fierClassTrain(varargin)
% fierClassTrain trains a classifier based on the cepstral distance
%
%  Inputs:
%   train_data - struct with all the signals for all the istances used
%               for training
%   window_length - length of the moving window used for regularize the
%               spectrum, expressed in [s]
%
%  Parameters (optional):
%   w - column vector window coefficients
%   overlap - {[0-1) - default 0} ratio of overlapped samples between one iteration and
%              the following when computing the spectrum: 0 means no
%              overlap, 1 means 100% overlap. Overlap it's bottom saturated to 1 sample
%              so to slide
%   showplot - {'Yes','No' - default 'No'} if 'yes', the plot of all the
%              cepstrum coefficients obtained during training is shown
%
%  Outputs:
%   fierClass - struct of the trained fierClass
%   order - maximum index in which the distance between the cepstra 
%           is relevant
%
%   Author: Simone Gelmini (simone.gelmini@polimi.it, gelminisimon@gmail.com)
%   Date: 2019/07/10.
%   Version: 1.3
%   Note on version 1.1: input management added.
%   Note on version 1.2: windowing added.
%   Note on version 1.3: overlap percentage added.
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

%% Vectors and variables initialization
fierClass=[];
show_plot=0;
w=1; % rectangular window
overlap=0; % overlapping percentage (0 means no overlap)

%% Input management
switch nargin
    case 1
        error('Error #0: incorrect number of arguments.')
    case 2
        train_data=varargin{1};
        window_length=varargin{2};
        selected=1:length(train_data(1).signal);
    otherwise
        train_data=varargin{1};
        window_length=varargin{2};
        selected=1:length(train_data(1).signal);
        
        % User selected values
        for i=3:2:nargin
            switch lower(varargin{i})
                case 'selected'
                    selected=varargin{i+1};
                case 'window'
                    w=varargin{i+1};
                case 'showplot'
                    show_plot=strcmpi(varargin{i+1},'yes');
                case 'overlap'
                    overlap=varargin{i+1};
                otherwise
                    error(['Error #1: option command not found: ' varargin{i}]);
            end
        end
end

%% Consistency checks
% Number of training classes
N_classes=size(train_data,2);

% Number of signals to be tested
N_signals_selected=length(selected);

% Max signal index to be tested
signal_index_max=max(selected);

% Min signal index to be tested
signal_index_min=min(selected);

for i=1:N_classes
    % Structure consistency check
    if ~isfield(train_data(i),'signal')
        error(['Error #2: class ', num2str(i), ' ill defined.']);
    else
        if ~isfield(train_data(i).signal,'value') || ~isfield(train_data(i).signal,'sampling')
            error(['Error #3: mispelled fields in class ',num2str(i)]);
        end
    end
    
    % Number of signals per class
    N_signals_Class(i)=size(train_data(i).signal,2);
    
    % Class signals consistency
    if i>1 && N_signals_Class(i)~=N_signals_Class(i-1)
        error('Error #4: unbalanced number of signals between classes.');
    end
    
    % Class signals not belonging to the selection set
    if i>0 && N_signals_Class(i)<signal_index_max
        error('Error #5: one or more of the selected signals are not available.');
    end
    
    % Sampling frequency
    fs = train_data(i).signal(1).sampling;
    if fs <= 0
        error('Error #6: sampling frequency must be a positive number.');
    end
    
    % Window length
    samples = window_length*fs;
    if fs <= 0 
        error('Error #7: window length must be a positive number.');
    elseif samples>length(train_data(i).signal(1).value)
        error('Error #8: window is larger than the time series.');
    end
    
end

% Subclass
subclass_not_defined=length([train_data(:).subclass])~=length(train_data);

if subclass_not_defined
    error('Error #9: subclass label not defined for one or more classes.');
end

% Overlap
if overlap<0 || overlap>1
    error('Error #10: overlap must be within [0,1].');
end

%% Train
% Cepstrum coefficients are computed for all the classes
for i=1:N_classes
    % Cepstrum coefficients are computed for all the signals
    for n=1:N_signals_selected
        clear Cepstrum_coefficient Quefrency
        
        if ~isempty(train_data(i).signal(n).label)
            fierClass(i).label=train_data(i).label;
        else
            warning(['Warning #3: label not defined for instance ' num2str(i) '.']);
        end
        
        if ~isempty(train_data(i).signal(n).label)
            fierClass(i).signal(n).label=train_data(i).signal(n).label;
        else
            warning(['Warning #2: label not defined for signal ' num2str(n) '.']);
        end
        
        cepstrum_coefficients=cepstrumTrain(train_data(i).signal(n).value,...
            train_data(i).signal(n).sampling,window_length,show_plot,w,overlap);
        fierClass(i).signal(n).cepstrum=cepstrum_coefficients;
        
        fierClass(i).window_length=window_length;
        
        % Cepstrum coefficients of the signals convolution are computed
        if selected(n)==signal_index_min
            fierClass(i).convolution=cepstrum_coefficients;
        else
            fierClass(i).convolution=fierClass(i).convolution+cepstrum_coefficients;
        end
        
        % Subclass definition
        if ~isempty(train_data(i).subclass)
            fierClass(i).subclass=train_data(i).subclass;
        else
            warning(['Warning #1: subclass not defined for signal ' num2str(n) '.']);
        end
    end
end

%% Order
order = get_order(fierClass);

end

function C = cepstrumTrain(x,fs,window_length,show_plot,w,overlap)
% cepstrum evaluates the cepstrum coefficients of a signal x with sampling
% frequency fs
%
%  Inputs:
%   x - signal in the time domain
%   fs - sampling frequency, Hz
%   window_length - length of the moving window used for regularize the
%                   spectrum, expressed in [s]
%   w - column vector window coefficients
%   overlap - {[0-1) - default 0} ratio of overlapped samples between one iteration and
%              the following when computing the spectrum: 0 means no
%              overlap, 1 means 100% overlap. Overlap it's bottom saturated to 1 sample
%              so to slide
%   showplot - {'Yes','No' - default 'No'} if 'yes', the plot of all the
%              cepstrum coefficients obtained during training is shown
%
%  Outputs:
%   C - real cepstrum
%
%   Author: Simone Gelmini (simone.gelmini@polimi.it, gelminisimon@gmail.com)
%   Date: 2019/05/03.
%
%  In case of need, feel free to ask the author.

N=length(x); % [-] - number of samples in the timeseries
n=ceil(window_length*fs); % [-] - number of samples in the window
step=max(1,ceil(window_length*fs*(1-overlap))); % [-] - number of samples between one interaction
                                                % and the following, saturated to at least one 
                                                % sample.

% Cepstral analysis
spectrum=zeros(n,1);

% Spectrum regularization
iterations=1;

for i=n:step:N
    % Signal windowing
    xw=windowing(x(i-n+1:i),w);
    % Spectrum
    mag=1/n*abs(fft(xw)).^2;
    spectrum=spectrum+mag;
    iterations=iterations+1;
end

% Regularized spectrum
spectrum=spectrum/iterations;

% The cepstrum is computed on the averaged spectrum
if max(spectrum(2:length(spectrum)))>1e-6
    % The spectrum is not flat to zero - there is a dynamics and
    % coefficients can be computed using the definition
    C=real(ifft(log(spectrum)));
    C=C(1:ceil(n/2));
    C(isnan(C))=0;
else
    % The spectrum is flat to zero - constant signal meaning that the
    % coefficients are forced to be zero
    C=zeros(1,ceil(n/2));
end

%Plot coefficients
if show_plot
    figure(1)
    hold all
    plot(C,'linewidth',1.5)
    xlim([0 40])
    xlabel('Lag [samples]')
    ylabel('Cepstrum coefficients')
    set(gca,'fontsize',15)
end

% Cepstrum coefficients vector is always a column vector
C_dim=size(C);
if C_dim(2)>C_dim(1)
    C=C';
end

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


function order = get_order(fierClass)
% Automatic cepstrum order computation. It is computed as the maximum  
% cepstra index in which the references coefficients are less closer 
% than a threshold.
%
%  Inputs:
%   fierClass - struct of the trained fierClass
%
%  Outputs:
%   order - maximum index in which the distance between the cepstra 
%           is relevant
%
%   Author: Jessica Leoni (jessica.leoni@polimi.it)
%   Date: 2020/08/25.
%
%  In case of need, feel free to ask the author.
    C_train = zeros(size(fierClass,2),...
        size(fierClass(1).signal,2), ...
        size(fierClass(1).signal(1).cepstrum,1));
    
    for class=1:size(fierClass,2)
        for signal=1:size(fierClass(class).signal,2)
            C_train(class,signal,:) = squeeze(fierClass(class).signal(signal).cepstrum);
        end
    end
    threshold = 1e-1;
    
    % For each reference cepstra, compute the distance.
    distance = zeros(size(C_train,2), size(C_train,3));
    for i=1:size(C_train,1)
        for j=i+1:size(C_train,1)
            distance = distance + squeeze(abs(C_train(i,:,:) - C_train(j,:,:)));
        end
    end
    
    % Set the distances below the threshold to zero
    distance(distance<threshold) = 0;
    
    % # Get the maximum index of non-zero distance
    order =max(find(max(distance)));
end