function PlotData

% numero massimo di stime di Hmix per giorno
NH=24;

% Punta alla dir dei dati LiDAR
DIRLIDAR='../Data/LIDAR/';
DataFile=dir(strcat(DIRLIDAR, 'LD40sp*.dat'));

% Punta Allah dir dei dati radio-sodaggio
DIRRDS='../Data/RDS/';
RDSData=load(strcat(DIRRDS, 'RDS.dat'));
RDSDate=num2str(RDSData(:,1:3), '%04d%02d%02d\n');

% File di output
fid = fopen('../MH_LIDAR_ALL-CORR.dat','w');

% Matrice degli errori dei dati sperimentali
fid2 = fopen('../E','w');

for i=1:numel(DataFile)
    
    % Legge i dati
    [Rds, Grad, Std, ValDer]=GetData(DIRLIDAR, DataFile(i).name, ...
        RDSDate, RDSData);
    
    % Visualizza i dati
    [Hour, z, Sz, hp1]=PlotFig1(DataFile(i).name, NH, Grad, Std, ValDer, Rds, 'G1');
    
    NumElm=sum(sum(~isnan(Grad)));
    zdot=zeros(NumElm,2);
    zdot(:,1)=Hour(~isnan(Grad));
    zdot(:,2)=Grad(~isnan(Grad));
    
    % Stima
    [Y UpLow]=ClassifyData(zdot, z, Rds, Sz, DataFile(i).name, hp1, fid2);
    
    
    % Salva i dati nel file di output
    for k=1:24
        date=datestr(datenum(DataFile(i).name(8:17)), 'yyyy mm dd');
        fprintf(fid,'%10s %02d %9.2f %9.2f %9.2f %9.2f\n', ...
            date, k-1, z(k,2), Y(k), UpLow(k,1), UpLow(k,2));
    end
    
    
end

fclose(fid);
fclose(fid2);

end

function [Rds, Grad, Std, ValDer]=GetData(DIR, filename, RDSDate, RDSData)

x=load(strcat(DIR, filename)); x(25,:)=[];

Data=x(:,1:4); x(:,1:6)=[];
Data=num2str(Data, '%04d%02d%02d%02d\n');

Grad=x(:,1:7)';
Std=x(:,8:14)';
ValDer=x(:,15:21)';

SelRds=strmatch(Data(1,1:8), RDSDate);
Rds=RDSData(SelRds,[4 6]);
    


end


function [Hour, z, Sz, hp1]=PlotFig1(filename, NH, Grad, Std, ValDer, Rds, CASE)

Hour=repmat((0:23),size(Grad,1),1);
z=NaN*zeros(NH, 2);
Sz=NaN*zeros(NH, 1);

figure(1), clf, cla
hp1=plot(Hour(:), Grad(:), 'bo', Rds(:,1), Rds(:,2), 'go');
set(hp1(1), 'MarkerSize', 4)


hold on
for i=1:size(Grad,2)
    if any(~isnan(Grad(:,i)))
        switch CASE
            case 'G1'
                [~, I]=max(ValDer(:,i));
            case 'G2'
                [~, I]=min(Std(:,i));
            case 'G3'
                if ( i < 7 || i > 19 ) % Prima delle 6 e dopo le 18 UTC
                    [~, I] = min(isnan(Std(:,i)));
                else
                    [~, I]=min(Std(:,i));
                end
            otherwise
                error('No such case')
        end
        hp=plot(Hour(I,i), Grad(I,i), 'ro');
        set(hp, 'MarkerFaceColor', 'r', 'MarkerSize', 4)
        z(i,1)=Hour(I,i); z(i,2)=Grad(I,i);
        Sz(i)=max(37.0, Std(I,i));
    else
        z(i,1)=Hour(1,i);
    end
end

if ( isempty(Rds) )
    hp1=plot(z(:,1), z(:,2), 'ko', 1, 1, 'w.');
    set(hp1(2), 'Visible', 'off');
end

set(gca, 'XLim', [-1.0 24])
set(gca, 'YLim', [0 4000])
set(gca, 'XTick', (0:3:21))
set(gca, 'YTick', (500:500:4000))
% Tick=num2str([1; 0.5; (500:500:2500)']);
% set(gca, 'YTickLabel', Tick)
hx=xlabel('Time (UTC)');
set(hx, 'FontSize', 12);
hy=ylabel('Elevation (m)');
set(hy, 'FontSize', 12);
ht=title(filename(1:17));
set(ht, 'FontSize', 12);

shg

end

function yp=odefun(t,y,k)

% Calcola la derivata di Hmix
Ampl=k(1);
tStart=k(2);
tAmpl=k(3);
gamma=k(4);

wtheta0=Ampl*sin(2*pi*(t-tStart*3600)/(tAmpl*3600));
beta=0.4;

yp(1,1) =  beta*wtheta0/y(2);
yp(3,1) = (1  + beta)*wtheta0/y(1);
yp(2,1) = gamma*beta*wtheta0/y(2) - yp(3,1);

end

function [SS yout] = ssfun(k,data,Sigma)

% Calcola Hmix (out) e la somma delle deviazioni
% dei quadrati normalizzata (SS)

global y0

time = data.tdata;
yobs = data.ydata;

[T, ~, J]=unique(time);
y0(1)=k(5);
[t,y] = ode23s(@odefun,T,y0,[],k);

if ( not(isequal(t(end), T(end))) )
    [~]=unique(T);
    S1=eps*ones(size(yobs));
    SS=sum(log(S1));
    yout=zeros(size(yobs));
    return
end

yout=y(J,1);
yout=max(20.0, yout);

% sum-of-squares
SS=nansum(((yobs-yout)./Sigma).^2);

end

function y = f1(time,k)

global y0

[T, ~, J]=unique(time);
y0(1)=k(5);
[~, y] = ode23s(@odefun,T,y0,[],k);
y=y(J,1);

end

function J = Jacobian(F,x,b,db,varargin)

% Adattato da Haario et al.:
% DRAM: Efficient adaptive MCMC, Statistics and Computing,
% 2006, 16, 339-354

% keywords: Jacobian, derivatives
% call:    J = jacob(F,x,b,db,P1,...)
% The function computes the Jacobian matrix of the function
% given in F with respect to the components of the vector b.
%
% INPUT   F      string  giving the name of the function
%                whose derivatives are computed:
%                 function y = func(x,b,P1,P2,...)
%         x      argument matrix of F, one row for each observation
%         b      vector of parameters
%         y0     starting values for ODEs
%         db     (scalar) relative step size of numerical differentiation
%                or a 1x2 vector with
%                 db(1):  relative step size
%                 db(2):  minimal absolute step size
%                OPTIONAL, default db = [1e-6 1e-12];
%         P1,... optional parameters in F
%
% OUTPUT  J     the Jacobian, j:th component on i:th row given as
%
%                    d F(x(i,:),b) /d b(j)

% Copyright (c) 1994,2003 by ProfMath Ltd
% $Revision: 1.3 $  $Date: 2004/01/01 19:55:01 $

if nargin < 4 | isempty(db) %#ok<OR2>
  db = [1e-6 1e-12];
end
if length(db) == 1, db = [db 1e-12]; end;

b   = b(:);
nb  = length(b);
nx  = length(x(:,1)); %#ok<NASGU>
db1 = db(1); db2 = db(2);

for i = 1:nb
  db = zeros(nb,1); db(i) = max([db1*abs(b(i)) db2]);

  JJ = (feval(F,x,b+db,varargin{:}) - ...
        feval(F,x,b-db,varargin{:})) ./ (2*db(i));

  [m,ny]=size(JJ);
  if i == 1
    J = zeros(m*ny,nb);
  end
  if (ny==1)                 %single response
    J(:,i) = JJ;
  elseif ny>1                %multi response
    JJ=JJ';  JJ=JJ(:);
    J(:,i) = JJ;
  end
end

end



function [Y UpLow]=ClassifyData(zdot, z, Rds, Sz, filename, hp1, fid2)


global y0

warning('off', 'MATLAB:ode23s:IntegrationTolNotMet')

PRINT=true(1);

% define data structure for mcmcrun
data.tdata=z(:,1)*3600; data.ydata=z(:,2);
data.zdot=fliplr(zdot);

% First Guess and initial condition
Ampl=1.6e-4*nanmax(data.ydata(9+2:15+2));
if ( isnan(Ampl) ), Ampl=0.1; end

tStart=2.90;
tAmpl=9.0*2.0;

gamma=0.006;

y0(1,1)=max(100,min([200.0;data.ydata(1:3)]));   % HStart
y0(2,1)=1.12;                                    % jump
y0(3,1)=293.8;                                   % Tini

if ( isnan(y0(1,1)) )
    y0(1,1)=200.0 + rand(1)*10;
end

% add starting value to data struct
data.tdata=[0; data.tdata];
data.ydata=[y0(1); data.ydata];
k_fit=[Ampl, tStart, tAmpl, gamma, y0(1,1)]';

% First guess for boundary-layer height
[~, ydata]=ode23s(@odefun, unique(data.tdata), y0, [], k_fit);

% Number of data
n=numel(data.tdata)-1;

% Number of params to fit
p=numel(k_fit);

% figure(1)
% hp0=plot(t/3600, ydata(:,1), 'r.');
% set(hp0, 'Marker', 'o', 'MarkerSize', 2)
% 
SS=nansum(data.ydata(2:end)-ydata(:,1)).^2;
sigma2= SS./(sum(not(isnan(data.ydata(2:end))))-p);          % mean squared error

% Compute the approximative cov matrix for parameters
J = Jacobian(@f1,data.tdata,k_fit,[]);
cov_from_fit = sigma2*((J'*J)\eye(size(J,2)));

if ( sum(not(isnan(data.ydata)))<p )
    % Se il numero di dati e' insufficiente, ritorna senza
    % effettuare calcoli
    Y=NaN*ones(size(data.ydata,1),1);
    UpLow=NaN*zeros(24,2);
    return
end

%%
%%%%%%%%%%%% MCMC %%%%%%%%%%%%%%%%
model.ssfun   = @ssfun;
model.fun1    = @f1;

params.par0   = k_fit;
params.n0     = 1;
params.n      = n;

params.bounds = [zeros(p,1), ones(p,1)]';

% Ampl
params.bounds(1,1) = 0.01;
params.bounds(2,1) = params.par0(1)+0.3;
% tStart
params.bounds(1,2) = params.par0(2)-2.0;
params.bounds(2,2) = params.par0(2)+2.0;
% tAmpl
params.bounds(1,3) = params.par0(3)-2.0;
params.bounds(2,3) = params.par0(3)+2.0;
% gamma
params.bounds(1,4) = 0.0;
params.bounds(2,4) = 0.1;
% PBL init
params.bounds(1,5) = 50.0;
params.bounds(2,5) = 500.0;

nsamples      = 1000;
Ncycles       = 1;
options.nsimu = fix(nsamples/Ncycles);
options.qcov  = cov_from_fit.*2.4^2./p;
options.printint = 500;
options.adaptint = 25;
options.drscale  = 10;
options.verbosity = 0;

%%

% Load first guess standard deviation
Sxfilename=datestr(datenum(filename(8:14)), 'yyyymm');
Sx=load(strcat('../Data/B/B.',Sxfilename,'.txt')); Sx=[Sx(1,2); Sx(:,2)];

% Load climatology
Climafilename=datestr(datenum(filename(8:15)), 'yyyymm');
Clima=load(strcat('../Data/Clima/Clima.',Climafilename,'.txt'));
params.Clima=[Clima(1,2); Clima(:,2)];

% Initial bulk-model values
[~, yout] = feval(model.ssfun,params.par0,data,1000);
params.Ini=yout;

% In case you do not have a climatology, use the initial values
% de-commenting the subsequent instruction
% params.Clima=yout;



figure(1)
hp0=plot(data.tdata(2:end)/3600, yout(2:end), 'g-');
hp00=errorbar(data.tdata(2:end)/3600, params.Clima(2:end), ...
    min(1000.0, 2*Sx(2:end)), min(1000.0, 2*Sx(2:end)));
set(hp00, 'MarkerFaceColor', 'm')
set(hp00, 'MarkerSize', 4)


% DWT estimated error
params.Sz=[Sz(1); Sz].*ones(size(yout));
params.Sx=Sx;

TEST=false(1);
if ( TEST )
    Y=NaN*zeros(24,1);
    UpLow=NaN*zeros(24,2);
    for k=2:numel(params.Sz)
        fprintf(fid2, '%04s %02s %02s %02d %7.2f\n', ...
            filename(8:11), filename(13:14), filename(16:17), k-2, params.Sz(k));
    end
    return
end

% Idx tags un-reliable data
Idx=GetIdx(params.Clima, data.tdata, data.ydata, Sx);
params.Sz(Idx)=Inf;
params.Idx=Idx;

Inan=isnan(data.ydata);
data.ydata(Inan)=yout(Inan);
params.Sz(Inan)=Inf;

if ( sum(Inan) > 16 )
    Y=NaN*zeros(24,1);
    UpLow=NaN*zeros(24,2);
    return
end


%%
% Start a short simulation to optimize initial parameter choice
options.nsimu = 100;
params.Test=true(1);
[results,~,~,Yout,UpLow,PDF] = ...
    mcmcrun_DL(model,data,params,options,hp0,hp1);

%%
params.par0=results.mean;
params.Ini=Yout;
options.nsimu = fix(nsamples/Ncycles);
params.Test=false(1);

StoreY=zeros(size(data.tdata,1),1);
tic
for i=1:Ncycles
    [results,~,~,Yout,UpLow,PDF] = ...
        mcmcrun_DL(model,data,params,options,hp0,hp1);
    params.par0=results.mean;
    params.Ini=Yout;
    StoreY=StoreY + Yout;
end
StoreY=StoreY/Ncycles;
toc
data.ydata(Inan)=NaN;

set(hp0, 'XData', data.tdata(2:end)/3600, 'YData', StoreY(2:end))

%%
figure(2), clf, cla
hp=plot(data.zdot(:,2), data.zdot(:,1), 'bo');
set(hp, 'Marker', 'o', 'MarkerSize', 4)
hold on

Time=(0:23)';
SelectedTime=NaN*zeros(size(Time));
for i=2:numel(data.tdata)
    I=find(data.zdot(:,2)==i-2);
    [~, J]=max(PDF(I));
    if ( Inan(i) || (abs(data.zdot(I(J),1)-Clima(i-1,2)))>min(1000.0, 2.0*Sx(i)) )
        hp=plot(data.tdata(i)/3600, StoreY(i), 'bx');
    else
        hp=plot(data.zdot(I,2), data.zdot(I(J),1), 'bx');
        StoreY(i)=data.zdot(I(J),1);
        hp1=plot(data.zdot(I,2), StoreY(i), 'ro');
        if ( not(isempty(hp1)) )
            SelectedTime(i-1)=data.zdot(I(J),2);
            set(hp1, 'MarkerFaceColor', 'r', 'MarkerSize', 4)
        end
    end
    set(hp, 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'LineWidth', 2)
end

% plot(data.tdata(2:end)/3600, StoreY(2:end), 'b-');

% hp0=plot(data.tdata(2:end)/3600, StoreY(2:end), 'bo');
% set(hp0, 'MarkerFaceColor', 'b')
% set(hp0, 'MarkerSize', 4)
% 

plot(Rds(:,1), Rds(:,2), 'go');

set(gca, 'XLim', [-1.0 24])
set(gca, 'YLim', [0 4000])
set(gca, 'XTick', (0:3:21))
set(gca, 'YTick', (500:500:4000))
% Tick=num2str([1; 0.5; (500:500:2500)']);
% set(gca, 'YTickLabel', Tick)
hx=xlabel('Time (UTC)');
set(hx, 'FontSize', 12);
hy=ylabel('Elevation (m)');
set(hy, 'FontSize', 12);
ht=title(filename(1:17));
set(ht, 'FontSize', 12);

shg

hold off
%%

figure(1)
hold on
hp0=plot(data.tdata(2:end)/3600, StoreY(2:end), 'bx');
set(hp0, 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'LineWidth', 2)
hp1=plot(SelectedTime, StoreY(2:end), 'ro');
set(hp1, 'MarkerFaceColor', 'r', 'MarkerSize', 4)
plot(Rds(:,1), Rds(:,2), 'go');
hold off
%%

%%
if ( PRINT )
    figure(2)
    print('-depsc', strcat('../IMG/EPS/', filename(1:17)));
    print('-djpeg95', strcat('../IMG/JPG/', filename(1:17)));
%     figure(1)
%     print('-depsc', strcat('../IMG/EPS/', filename(1:17), 'AltFig'));
%     print('-djpeg95', strcat('../IMG/JPG/', filename(1:17), 'AltFig'));
end

Y=StoreY(2:end);
UpLow=UpLow(2:end,:);

%%

end

function Idx=GetIdx(Clima, tdata, ydata, Sx)

% check max distance
Idx=abs(ydata-Clima)>min(1000.0, 2.0*Sx);
Idx(1)=true(1);

Idx(isnan(ydata))=true(sum(isnan(ydata)),1);

figure(1)
X=tdata(Idx)/3600; Y=ydata(Idx);
hp=plot(X(2:end), Y(2:end), 'rx');
set(hp, 'MarkerSize', 12, 'LineWidth', 2)


end



