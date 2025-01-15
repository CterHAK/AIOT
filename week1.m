%CHAPTER 1
x = [1 2 2 4 4 5 5];
figure;
plot(x);
%% 
5 + 2
%%
(-5) + (-2)
%%
-5 + -2
%%
5-2
%%
(-5)-(-2)
%%
-5 - -2
%%
5*2
%%
(-5)*(-2)
%%
-5 * -2
%%
5/2
%%
-5/-2
%%
5^2
%%
(-5)^(-2)
%%
log(5)
%%
log(-5)
%%
exp(5)
%%
exp(-5)
%%
sin(pi/2)
%%
sin(-pi/2)
%%
x = 1;
y = 2;
z = x + y
%%
5+[1 3]
%%
[5 2 ]+[1 3]
%%
5-[1 3]
%%
[5 2 ]- [1 3]
%%
5* [1 3]
%%
% Ở trường hợp này, phép nhân được hiểu là nhân ma trận nhưng kích thước
% của 2 ma trận không đảm bảo điều kiện nên bị lỗi: Inner matrix dimensions must agree
[5 2] * [1 3]
%%
% Để khắc phục ta dùng phép nhân từng phần tử .* để nhân các phần tử tương
% ứng của 2 vector
[5 2] .* [1 3]
%%
dot([5 2],[1 3])
%%
[1 3]/5
%%
[5  2]./[1  3]
%%
[5  2]'
%%
x = [1 2 3];
y = [2 4 9];
z = x + y
%%
x = ones(1 ,4)
%%
x = zeros(4 ,1)
%%
x = linspace(1,4,5)
%%
x = 1:2:6
%%
x = linspace(1, 6, 2)
%%
x = linspace(1, 4, 5)
x(3)
%%
x = linspace(1, 16, 8)
x(2:4)
%%
x = linspace(1,4,5)
x(6)=10
%%
5+[3 4;2 3]
%%
[1  3;1  2]+[3  4;2  3]
%%
% lỗi do không cùng kích thước
[1  5]+[3  4;2  3]
%%
 5-[3  4;2  3]
 %%
 [1  3;1  2]-[3 4;2  3]
 %%
 5*[3  4;2  3]
 %%
 [1  3;1  2]*[3  4;2  3]
 %%
 [1  3;1  2].*[3  4;2  3]
 %%
 [1  3;1  2]/5
 %%
 [1  3;1  2]./[3  4;2  3]
 %%
 [1  3;1  2]'
 %%
  x=[1  3;1   2];
  y=[3  4;2  3];
  x+y
  %%
  x = ones(4, 4)
  %%
  x = zeros(3, 3)
  %%
  A =[1 2 3; 4 5 6; 7 8 9]
  A(2,3)
  A(2,:)
  %%
  5>2
  5>[6 3 2 9 7]
  [5 6 1 7 9]>[6 3 2 9 7]
  %%
  -Inf+Inf
  0*Inf
  Inf/Inf
  0/0
  isnan(NaN)
  %%
  first =  'Derek';
  last     =  'Abbott';
  name     = [first,' ',last]
  %%
  first(3)
  %%
  x=int2str(15)
  whos x
  x=num2str(15.34)
  whos x
  %%
  x='Nigel H. Lovell'
  lower(x)
  %%
  x='Gary Clifford'
  upper(x)
  %%
  x='UBC'
  y='ubc'
  strcmp(x,y)
  %%
  strrep('Life', 'L', 'W')
  findstr('Life', 'fe')
  %%
  Participant.FirstName='Kirk';
  Participant.LastName='Shelley';
  Participant.Age=45;
  Participant.BloodPressure=[110, 140, 113];
  Participant.Hypertension=logical([0,1,0]);
  Participant
  %%
  Participant.BloodPressure(2)
  %%
  Participant(2).FirstName='John';
  Participant(2).LastName='Allen';
  Participant(2).Age=50;
  Participant(2).BloodPressure=[139,  130,  141];
  Participant(2).Hypertension=logical([1,0,1]);
  Participant(2)
  %%
  Participant(2).BloodPressure(3)
  %%
  SubjectName={'Rafael Ortega','Andrew Reisner', 'Gerald Dziekan'}
  BloodPressure=[100 110 109],[140 145 143],[100 101 100]
  Data={SubjectName,BloodPressure}
  %%
  Data{1,1} = {'Rafael  Ortega','Andrew  Reisner', 'Gerald Dziekan'}
  Data {1,2}={[100   110   109],[140   145  143],[100   101   100]}
  Data{1,1}{1,1}
  %%
  a =[1,  2,  3,  4;23  24  25  26]
  b ='My first test'
  c =[0,1]
  save  test1
  %%
  save  test2  a
  %%
  load('test1.mat')
  whos
  load('test2.mat')
  whos
  %%
  xlswrite('test2.xls',a)
  xlsread('test2.xls')
  %%
  x=input('How  old  are  you? ')
  %%
  x=input('What  is  your  name?  ','s')
  %%
  3<2
  [1 2 5 3 0 3] ~= 3
  0 == [000]
  %%

  %CHAPTER 2
  x=0:pi/100:2*pi;
  wave_1=cos(x*2);
  figure , plot(wave_1)
  xlabel("Angle")
  ylabel("Amplitude")
  title ("An example of generating a waveform using a sinusoid");
  %%
  x=0:pi/100:2*pi;
  wave_2=cos(x*3)+cos(x*7-2);
  figure , plot(wave_2)
  xlabel("Angle")
  ylabel("Amplitude")
  title ("How to simulate PPG waveforms using sinusoids in Matlab.");
  %%
  a = 1;
  mu = 50;
  sigma = 10;
  x = 1:1:100;
  y = a * exp(-(((x-mu)/sigma).^2)/2) ;
  figure; plot(x ,y);
  xlabel("Sampling points");
  ylabel("Amplitude");
  title("An example of generating a waveform using one Gaussian function");
  %%
  a = [0.8,0.4];
  mu = [25,50];
       sigma = [10,20];
  x = 1:1:100;
  y1 = a(1) * exp(-(((x-mu(1))/sigma(1)).^2)/2);
  y2 = a(2) * exp(-(((x-mu(2))/sigma(2)).^2)/2);
  y = y1 + y2;

  figure; plot(x,y,'b');
  hold on; plot(x,y1,'k--');plot(x,y2,'r--');
  xlabel("Sampling points");
  ylabel("Amplitude");
  legend("Synthetic PPG","1^{st} Gaussian", "2^{nd} Gaussian");
  title("An example of generating a waveform using two Gaussian functions");
  %%
  Duration = 1;
  Fs = 125; %Sampling Frequency
  a = [0.82,0.4];
  mu = [-pi/2,0];
  sigma = [0.6,1.2];

  Samples = Fs/Duration;
  V_angle = 2*pi/Samples;
  angle = -pi+V_angle :V_angle:pi;
  y1 = a(1) * exp(-(((angle-mu(1))/sigma(1)).^2)/2);
  y2 = a(2) * exp(-(((angle-mu(2))/sigma(2)).^2)/2);
  y = y1 + y2;

  figure; plot(angle,y,'b');
  hold on; plot(angle,y1,'k-');plot(angle,y2,'r-');
  xlabel("Angle");
  ylabel("Amplitude");
  xlim([-pi,pi]);
  set(gca,'xtick',[-pi,0,pi],'xticklabel',{'\pi','0','\pi'});
  legend("Synthetic PPG","1^{st} Gaussian", "2^{nd} Gaussian");
  title("An example of generating a waveform using angle model");
  %%

  %CHAPTER 3
  
  % Chương lý thuyết

  %%

  %CHAPTER 4
   % Implementing the moving average using a simple for loop
   WindowSize = 4;
   Raw_Sig = sin(2*pi*0.01*(1:500)) + 0.2*randn(1, 500); % Tín hiệu sóng sin với nhiễu ngẫu nhiên
   figure, plot (Raw_Sig);
   for i=1:length(x)-WindowSize      
        Filtered_Sig(i) = 1/WindowSize *(Raw_Sig(i) + Raw_Sig(i+1) + Raw_Sig(i+2) + Raw_Sig(i+3));
   end
   figure, plot (Filtered_Sig);
   xlabel('Samples');
   ylabel('Amplitude');
   title('Moving Average')
   %%
   % Implementing the moving average using Convolution
   windowSize = 4;
   Raw_Sig = "your PPG signal";
   figure, plot (Raw_Sig);
   kernel = [0,0, 1/4, 1/4, 1/4, 1/4, 0, 0];
   Filtered_Sig = conv(Sig, kernel, 'same');
   figure, plot (Filtered_Sig);
   xlabel('Samples');
   ylabel('Amplitude');
   title('Moving Average')
   %%
    % Implementing the frequency response of Moving Average filter
    % Sampling frequency in Hz
    Fs = 200;
    windowSize = 5;
    num = (1/windowSize)*ones(1,windowSize);
    dend = 1;
    % Logarithimc scale
    L=logspace(0,2);
    % call the Freqency response function
    Z=freqz(num,dend,L,Fs);
    % Compute and display the magnitude response
    figure, semilogx(L,abs(Z),'K');
    grid;
    xlabel('Hz');
    ylabel('Gain');
    title('Moving Average')
    %%
     % Implementing the frequency response of Butterworth filter
     % Sampling frequency in Hz
     Fs=200;
     Fc=6/(Fs/2);
     m=6;
     % Butterworth filter
     [num dend]= butter(m,Fc);
     % Logarithimc scale
     L=logspace(0,2);
     % call the Freqency response function
     Z=freqz(num,dend,L,Fs);
     % Compute and display the magnitude response
     figure; semilogx(L,abs(Z),'K');
     grid;
     xlabel('Hz');
     ylabel('Gain');
     title('Butterworth')
     %%
      % Implementing the frequency response of Cheby I filter
      % Sampling frequency in Hz
      Fs=200;
      Fc=6/(Fs/2);
      m=6;
      Rs=18;
      % cheby1 filter
      [num, dend]=cheby1(m,Rs,Fc);
      % Logarithimc scale
      L=logspace(0,2);
      % call the Freqency response function
      Z=freqz(num,dend,L,Fs);
      % Compute and display the magnitude response
      figure, semilogx(L,abs(Z),'K');
      grid;
      xlabel('Hz');
      ylabel('Gain');
      title('Chebyshev 2')
   %%
   % Implementing the frequency response of Elliptic filter
   Fs=200;
   Fc=6/(Fs/2);
   m=6;
   Rp=0.5;
   Rc=20;
   % Butterworth filter
   [num, dend]=ellip(m,Rp,Rc,Fc);
   % Logarithimc scale
   L=logspace(0,2);
   % call the Freqency response function
   Z=freqz(num,dend,L,Fs);
   % Compute and display the magnitude response
   figure, semilogx(L,abs(Z),'K');
   grid;
   xlabel('Hz');
   ylabel('Gain');
   title('Elliptic')
   %% Parameters:
   % filter_type ———————————filter type
   %  order——————————————————filter order (level − Wavelet or point——Fir)
   % raw_data———————————————raw PPG signal
   % Fs—————————————————————sample frequency
   % fc—————————————————————Cutoff Frequency
   function [filtered_data] = PPG_Lowpass(raw_data, filter_type,order,Fs,fc)
   Fn = Fs/2;
   switch filter_type
       case 1
           [A,B,C,D] = butter(order,fc/Fn,'low');
           [filter_SOS,g] = ss2sos(A,B,C,D);
           filtered_data = filtfilt(filter_SOS,g, raw_data);
       case 2
           [A,B,C,D] = cheby1(order,0.1,fc/Fn,'low');
           [filter_SOS,g] = ss2sos(A,B,C,D);
           filtered_data = filtfilt(filter_SOS,g, raw_data);
       case 3
           [A,B,C,D] = cheby2(order,20,fc/Fn,'low');
           [filter_SOS,g] = ss2sos(A,B,C,D);
           filtered_data = filtfilt(filter_SOS,g, raw_data);
       case 4
           [A,B,C,D] = ellip(order,0.1,30,fc/Fn,'low');
           [filter_SOS,g] = ss2sos(A,B,C,D);
           filtered_data = filtfilt(filter_SOS,g, raw_data);
       case 5
           d = fir1(order,fc/Fn,'low');
           filtered_data = filtfilt(d,1,raw_data);
           filter_SOS = d;
       case 6
           d = designfilt('lowpassfir','FilterOrder', order, ...
               ' PassbandFrequency',fc,'Stopband Frequency',fc+0.2, ...
               'DesignMethod','ls', 'SampleRate',sample_freq);
           filtered_data = filtfilt(d,raw_data);
       case 7
           filtered_data = smooth(raw_data,order);
       case 8
           filtered_data = medfilt1(raw_data,order);
       case 9
            filtered_data= wden(raw_data,'modwtsqtwolog', 's','mln',order, ...
                'db2'); %Wavelet level: order
   end
   end


   %% Parameters:
   % filter_type ———————————filter type
   % order——————————————————filter order
   % raw_data———————————————raw PPG signal
   % Fs—————————————————————sampling frequency
   % fL—————————————————————lower cutoff frequency
   % fH—————————————————————higher cutoff frequency

   function [filtered_data] = PPG_Bandstop(raw_data, filter_type, order,Fs,fL,fH)
   Fn = Fs/2;
   switch filter_type
       case 1
           [A,B,C,D] = butter(order,[fL fH]/Fn, 'stop');
           [filter_SOS,g] = ss2sos(A,B,C,D);
           filtered_data = filtfilt(filter_SOS,g, raw_data);
       case 2
           [A,B,C,D] = cheby1(order,0.1,[fL fH]/Fn, 'stop');
           [filter_SOS,g] = ss2sos(A,B,C,D);
           filtered_data = filtfilt(filter_SOS,g, raw_data);
       case 3
           [A,B,C,D] = cheby2(order,20,[fL fH]/Fn, 'stop');
           [filter_SOS,g] = ss2sos(A,B,C,D);
           filtered_data = filtfilt(filter_SOS,g, raw_data);
       case 4
           [A,B,C,D] = ellip(order,0.1,30,[fL fH]/Fn, 'stop');
           [filter_SOS,g] = ss2sos(A,B,C,D);
           filtered_data = filtfilt(filter_SOS,g, raw_data);
       case 5
           d = fir1(order,[fL fH]/Fn,'stop');
           filtered_data = filtfilt(d,1,raw_data);
       case 6
   end
   end
   %%
       % Tạo tín hiệu mẫu và tín hiệu nhịp PPG
        t = 0:0.01:2*pi;               % Thời gian mẫu
        template = sin(t);             % Tín hiệu mẫu (template) - sóng sin
        beat = sin(t) + 0.2*rand(size(t)); % Tín hiệu nhịp PPG (beat) - sóng sin + nhiễu
        
        % Đặt giá trị ban đầu
        f = template;                  % Tín hiệu mẫu chất lượng cao
        g = beat;                      % Tín hiệu nhịp PPG cần cải thiện
        
        % Nội suy tín hiệu nhịp PPG để có cùng chiều dài với tín hiệu mẫu
        g = resample(beat, length(f), length(g));  
        
        % Tính tích chập (convolution) của hai tín hiệu
        conv_f_g = conv(f, g);         % Tích chập f với g
        conv_g_f = conv(g, f);         % Tích chập g với f
        
        % Vẽ các tín hiệu và kết quả tích chập
        figure;
        subplot(4,1,1);
        plot(f, 'k-'); 
        title('f: Template');
        
        subplot(4,1,2);
        plot(g, 'r-'); 
        title('g: Raw signal');
        
        subplot(4,1,3);
        plot(conv_f_g, 'b-'); 
        title('f * g');
        
        subplot(4,1,4);
        plot(conv_g_f, 'g-'); 
        title('g * f');
        
        suptitle('Convolution');

    %% CHAPTER 5

    % Tạo tín hiệu mẫu
    Fs = 125;                 % Tần số lấy mẫu (Hz)
    t = 0:1/Fs:10;            % Thời gian từ 0 đến 10 giây
    signal = sin(2*pi*1*t);   % Tín hiệu sóng sin 1Hz
    signal = signal + 0.1*randn(size(signal)); % Thêm nhiễu Gaussian

    zeroCrossingNum = 0;
    for i = 1:1:(length(signal)-1)
        if  signal(i)*signal(i+1)<=0
            zeroCrossingNum = zeroCrossingNum +1;
        end
    end
    Z_SQI = zeroCrossingNum/length(signal);
    %%
    % Tạo tín hiệu mẫu
    Fs = 125;                 % Tần số lấy mẫu (Hz)
    t = 0:1/Fs:10;            % Thời gian từ 0 đến 10 giây
    signal = sin(2*pi*1*t);   % Tín hiệu sóng sin 1Hz
    signal = signal + 0.1*randn(size(signal)); % Thêm nhiễu Gaussian

    signal=signal-mean(signal); %remove mean
    NFFT = max(256,2^nextpow2(length(signal)));
    Fs = 125;   % Sampling frequency
    %  Welch Method
    [pxx,f] = pwelch(signal,length(signal),length (signal)/2,(NFFT*2)-1,Fs);

    F1 = [1 2.25];
    F2 = [0 8];

    powerF1 = trapz(f(f>=F1(1)&f<=F1(2)),pxx(f>=F1(1) &f<=F1(2)));
    powerF2 = trapz(f(f>=F2(1)&f<=F2(2)),pxx(f>=F2(1) &f<=F2(2)));

    R_SQI = powerF1/powerF2;
    
    % Hiển thị kết quả
    disp(['R_SQI: ', num2str(R_SQI)]);
    
    % Vẽ tín hiệu và phổ công suất
    figure;
    subplot(2,1,1);
    plot(t, signal);
    xlabel('Thời gian (s)');
    ylabel('Biên độ');
    title('Tín hiệu mẫu');
    
    subplot(2,1,2);
    plot(f, pxx);
    xlabel('Tần số (Hz)');
    ylabel('Mật độ công suất');
    title('Phổ công suất (PSD)');
    %% CHAPTER 6
    A  = [0.57 0.59 0.60 0.1 0.59 0.58 0.57 0.58 0.3 0.61 0.62 0.60 0.62 0.58 0.57];
    TF = isoutlier(A,'mean')

    %%
    % Tạo giá trị mẫu cho intervals
    Fs = 125;                          % Tần số lấy mẫu (Hz)
    t = 0:1/Fs:60;                     % Thời gian từ 0 đến 60 giây
    heartRate = 60 + 5*sin(2*pi*0.1*t); % Nhịp tim dao động quanh 60 bpm
    intervals = 60 ./ heartRate;       % Khoảng thời gian giữa các nhịp (NN intervals)
    
    % Chuyển đổi từ giây sang mili giây (cho dễ đọc hơn)
    intervals = intervals / 1000;

    diffNN = diff(intervals);
    SD1  = sqrt(0.5*std(diffNN)^2)*1000;   
    SD2  = sqrt(2*(std(intervals)^2) - (0.5*std (diffNN)^2))*1000; % the unit of SD2 is ms
    ratio_SD2_SD1 = SD2/SD1;

    % draw Poincare plot
    poincare_x = intervals(1:end-1)*1000;  %convert to ms
    poincare_y = intervals(2:end)*1000;  %convert to ms
    figure;
    plot(poincare_x,poincare_y,'.');
    hold on;
    %draw ellipse
    phi  = pi/4; % The new coordinate system is established at 45 degree to the normal axis
    new_x=poincare_x./cos(phi);  %translatex to new coordinate
    center_new_x=mean(new_x);
    [cnx, cny]=deal(center_new_x*cos(phi),center_new_x*sin(phi));
    ellipse_width=SD2;
    ellipse_height=SD1;
    theta = 0:0.01:2*pi;
    x1 = ellipse_width*cos(theta);
    y1 = ellipse_height*sin(theta);
    X = cos(phi)*x1 - sin(phi)*y1;
    Y = sin(phi)*x1 + cos(phi)*y1;
    X = X + cnx;
    Y = Y + cny;
    plot(X,Y,'k-');
    %plot SD1 and SD2 inside the ellipse
    line_SD1=line([cnx cnx],[cny-ellipse_height cny+ellipse_height],'color','g');
    rotate(line_SD1,[0,0,1],45,[cnx,cny,0]);
    line_SD2=line([cnx-ellipse_width cnx+ellipse_width],[cny cny],'color','m');
    rotate(line_SD2,[0,0,1],45,[cnx,cny,0]);
    %% CHAPTER 7
    % Tạo giá trị mẫu cho AnnotateEvents và ExtractedEvents
    Fs = 1000;  % Tần số lấy mẫu (Hz)
    duration = 10; % Thời gian tín hiệu (giây)
    t = 0:1/Fs:duration; % Thời gian tín hiệu
    AnnotateEvents = [0.5, 2.1, 4.3, 6.8, 9.5]; % Các mốc sự kiện được chú thích
    ExtractedEvents = [0.48, 2.15, 4.31, 6.85, 9.45, 10.0]; % Các mốc sự kiện được trích xuất
    MaxDifference = 0.02; % Ngưỡng sai lệch tối đa (giây)

    x1 = AnnotateEvents;
    x2 = ExtractedEvents; 
    MaxDifference = 0.02;
    TP_inAnnotate = [];
    TP_inExtracted = [];
    FN = [];
    FP = [];
    for i = 1:length(x1)
        t = find(abs(x1(i) - x2) <MaxDifference);
        if ~ isempty(t)
            TP_inAnnotate = [TP_inAnnotate, i];
            TP_inExtracted = [TP_inExtracted, t];
        end
    end
    TP = length(TP_inAnnotate);
    FN_events = x1;
    FN_events(TP_inAnnotate) = [];
    FN = length(FN_events);

    FP_events = x2;
    FP_events(TP_inExtracted) = [];
    FP = length(FP_events);

    SE = TP/(TP + FN);
    PP = TP/(TP + FP); %+P
    %%
    n = 1;     % the iterations
    for F1 = F1_vector      % F1_vector is the selectable values of parameter F1
        for F2 =  F2_vector % F2_vector is the selectable values of parameter F2
            for W1  = W1_vector % W1_vector is the selectable values of parameter W1
                for W2  = W2_vector % W2_vector is the selectable values of parameter W2
                    for beta  = beta_vector % beta_vector is the selectable values of parameter beta
                         ExtractedEvents = DETECTOR (F1 , F2 ,W1,W2, beta ) ; % extract events by TERMA
                          [SE(n) ,PP(n) ] = calculateAccuracy (AnnotateEvents , ExtractedEvents ) ;    %calculate the accuracy SE and +P
                          n = n + 1;
                          parameters (n, :) = [F1,F2,W1,W2, beta];
                    end
                end
            end
        end
    end
    Results = [SE(:), PP(:), parameters];
    SortedResults = sortrows ( Results, 'descend');
    BestParameters = SortedResults (1, 3: end);

    %% CHAPTER 8
    for i=1:FeatSet
        meanOfFeature=mean(Features(i,:));
        stdOfFeature=std(Features(i,:));
        NormalizedFeatures(i,:)=((Features(i,:)-meanOfFeature)/stdOfFeature);
    end
    %%
    MinVal = 0;
    MaxVal = 1; 
    for i=1:numFeat
        theMin=min(Features(i,:));
        theMax=max(Features(i,:));
        NormalizedFeatures(i,:)=MinVal+((MaxVal-MinVal)*(Features(i,:)-theMin))/(theMax-theMin);
    end
    %%
    r = 0.5;
    Features=(1.0/r)*Features;
    NormalizedFeatures=1.0 ./(1.0+exp(-Features));
    %%
    N1=100;
    N2=40;
    mu1=3;
    mu2=0;
    var1=1;
    var2=1;
    Normtensive=mu1+var1*randn(1,N1);
    Hypertensive=mu2+var2*randn(1,N2);
    histogram(Normtensive)
    hold on, histogram (Hypertensive)
    xlabel('Feature value');
    ylabel('Frequency');
    %%
     u1=rand(100);
     u2=rand(100);
     u3=rand(100);
     U = [u1 u2 u3];
     anova1(U)
     M=meshgrid(1:100);
     V=M(:,1:3);
     W=U.*V;
     anova1(W)
    %%
    m_N = mean(F_N);
    m_H = mean(F_H);
    v_N = var(F_N);
    v_H = var(F_H);
    J_F = (m_N - m_H)^2 / (v_N + v_H);
    %% CHAPTER 9
    Feature1 = [1.12 0.11];
    Feature2 = [0.7 0.33];
    net1 = competlayer(2);

    feature = [Feature1; Feature2];
    net = train(net1,feature);
    outputs = net(feature);

    classes = vec2ind(outputs);
    %%
    Feature1=[1.12 0.11];
    Feature2=[0.7 0.33];
    Feature3=[1.01 0.10];
    Feature4=[0.6 0.3];
    feature=[Feature1;Feature2;Feature3;Feature4];
    Group=['Class 1';'Class 2';'Class 1';'Class 2'];
    [C,err,P,logp,coeff] = classify(feature,feature, Group,'linear')
    %%
    Feature1=[1.12 0.11];
    Feature2=[0.7 0.33];
    Feature3=[1.01 0.10];
    Feature4=[0.6 0.3];
    feature=[Feature1;Feature2;Feature3;Feature4];
    Group=['Class 1';'Class 2';'Class 1';'Class 2'];
     [C,err,P,logp,coeff] = classify(feature,feature, Group,'diagquadratic')



    









