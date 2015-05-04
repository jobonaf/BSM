function [results,chain,s2chain,Y,UpLow,PDF]=mcmcrun_DL(model,data,params,options,hp0,hp1)
%MH (Metropolis-Hastings) MCMC run with options for adaptive delayed rejection (DRAM)
%

% Adattato da Haario et al.:
% DRAM: Efficient adaptive MCMC, Statistics and Computing,
% 2006, 16, 339-354


% The function generates MCMC chain for a model given together with a
% user supplied sum-of-squares function. Additive i.i.d. Gaussian
% error is supposed for the observations. The error variance sigma2 may be updated
% using the conjugate inverse gamma distribution.
%
% INPUT:
%
% model.ssfun =    ; % scaled sum-of-squares function, ss=ssfun(par,data,S),
%
% model.priorfun = ; % scaled prior "sum-of-squares", priorfun(par,params),
%                    % default: inline('0','x','params')
%
% data    = ;        % extra argument for ssfun (to pass the data etc.)
% S       = ;        % standard deviation
%
% params.par0   =  ; % initial parameter vector (a row vector)
% params.sigma2 =  1;% initial/prior value for the Gaussian error variance
% params.n0     = -1;% precision of sigma2 as imaginative observations
%                    %   if n0<0, no sigma2 update
% params.n      = ;  % number of actual observations (for sigma2 update)
% params.bounds = ;  % 2*npar matrix of parameter bounds
%                    % default: [-Inf,Inf]
%
% options.nsimu  = 10000;   % length of the chain
% options.qcov   = ;        % proposal covariance matrix. Initial, if adaptation used
%
% parameters for DR and AM
% options.adaptint = 10;  % how often to adapt.            if zero, no adaptation
% options.drscale  = 3;   % scale for the second proposal. if zero, no DR
%
% OUTPUT:
%
% results  structure that contains info about the run
% chain    the MCMC chain of size nsimu x npar
% s2chain  the sigma chain (if generated)
% Y        median value of modeled boundary layer growth
% UpLow    upper and lower interquartile range

% calls covupd.m for covariance update and (optionally) gammar_mt.m for
% gamma variates

global chaincov chainmean wsum lasti
%covariance update uses these to store previous values
chaincov = []; chainmean = []; wsum = []; lasti = 0;

[nsimu,par0,npar,bounds,ssfun,~,adaptint,drscale,adascale,                ...
    qcoveps,~,~,sigma2,qcov,dodr,s20,s2chain,R2,iR,printint,verbosity] =  ...
    optionfun(model,params,options);

Y = NaN*zeros(size(data.tdata,1),nsimu);

Sx=params.Sx;
Sz=params.Sz;

Idx=params.Idx;
NotIdx=not(Idx);

PDF=zeros(size(data.zdot,1),1);
NPDF=0;

Clima=params.Clima;

chain       = zeros(nsimu,npar);           % store the chain here
oldpar      = par0(:)';                    % first row of the chain
acce        = 1;                           % how many accepted moves
chain(1,:)  = oldpar;
R           = chol(qcov);                  % Cholesky factor of proposal covariance

set(hp0, 'XData', data.tdata/3600, 'YData', params.Ini);
shg

Temper=0.01; LenTempering=0.3;

Sx=chol(diag(Sx.*Sx));

[oldss yout] = feval(ssfun,oldpar,data,Sz);  % first sum-of-squares
if ( params.Test ) 
    oldprior = 0.0;
else
    oldprior = Inf;
end

%%% the simulation loop
for isimu=2:nsimu
    
    Temper=min(1.0, Temper + 0.99/nsimu/LenTempering);
    
    if isimu/printint == fix(isimu/printint) && isimu<nsimu % info on every printint iteration
        fprintf('isimu=%d, %d%% done, accepted until now: %d\n',...
            isimu,fix(isimu/nsimu*100),acce);
        figure(1), shg
    end
    
    
    % check until new probed parameters are within admissible bounds
    NotConv=true(1);
    while ( NotConv )
        newpar = oldpar+randn(1,npar)*R;  % a new proposal
        if all(newpar>bounds(1,:)) && all(newpar<bounds(2,:))
            NotConv=false(1);
        end
    end
    
    accept = 0;
    if ( 1 ) % inside bounds, check if accepted
        
        [newss yout] = feval(ssfun,newpar,data,Sz);
        if ( params.Test ) 
            newprior = oldprior;
        else
            newprior = (yout-Clima)'*(Sx\(Sx'\(yout-Clima)));
        end
        alpha = min(1,exp(-0.5*(newss-oldss)-0.5*(newprior-oldprior)).^Temper);
        if ( sum(yout) < eps )
            alpha=0.0;
        end
        if rand < alpha % we accept
            accept   = 1;
            acce     = acce+1;
            oldpar   = newpar;
            oldss    = newss;
            oldprior = newprior;
        end
    end
    
    %%%%%%%%%%%%%%%DR STEP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if accept == 0 && dodr    % step was rejected, but make a new try (DR)
        [oldpar,oldss,oldprior,acce,yout,accept]=...
            drfun(data,params,bounds,ssfun,R2,iR,alpha,acce,...
            oldprior,newprior,oldss,newss,oldpar,newpar,yout,Temper,Sz,Sx);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    chain(isimu,:) = oldpar; % save the present sample choice
    
    %%%%% update the error variance sigma2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if s20 > 0
        sigma2  = 1./gammar_mt(1,1,0.5*(numel(yout)-sum(NotIdx)),2./(sum(yout(NotIdx)-data.ydata(NotIdx)).^2));
        s2chain(isimu,:) = mean(sigma2);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%%%%%%%%%%%%%%AM adaptation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if adaptint>0 && fix(isimu/adaptint) == isimu/adaptint
        [R,R2,iR]=amfun(chain,isimu,npar,qcoveps,adascale,verbosity,dodr,drscale);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if ( accept )
      set(hp0, 'XData', data.tdata/3600, 'YData', yout, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g');
      set(hp1(2), 'XData', data.tdata(NotIdx)/3600, 'YData', data.ydata(NotIdx), ...
          'Marker', 'o', 'MarkerSize', 4, 'Color', 'r', 'MarkerFaceColor', 'r', 'visible', 'on');
      shg
    end
    
    Y(:,isimu)=yout;
    if ( Temper >= 1.0 )
        NPDF=NPDF+1;
        for i=0:numel(data.tdata)-2
            I=data.zdot(:,2)==i;
%             [~,J] = min(abs(data.zdot(I,1)-yout(i+2)));
%             JJ=find(I); PDF(JJ(J))=PDF(JJ(J)) + 1;
%             PDF(I)=PDF(I) + normpdf(data.zdot(I,1), yout(i+2), sqrt(sigma2(i+2)));
            Sig=sqrt(1./(1./Sz(i+2).^2+1./Sx(i+2,i+2).^2));
            PDF(I)=PDF(I) + normpdf(data.zdot(I,1), yout(i+2), Sig);
        end
    end
    
end

% Final calculation of covariance and mean of the chain
[chaincov,chainmean,wsum] = covupd(chain((lasti+1):isimu,:), ...
    1,chaincov,chainmean,wsum);


% Collect the results
results.class = 'MCMC results';
results.accepted=acce./nsimu;              % acceptance ratio
results.mean = chainmean;
results.cov  = chaincov;
results.qcov = R'*R;
results.R = R;
results.nsimu = nsimu;
results.drscale = drscale;
results.adascale = adascale;
results.adaptint = adaptint;

Y(:,1)=[];
Y(:,1:end-NPDF)=[];
s2chain(1:end-NPDF)=[];

UpLow=prctile(Y,[25 50 75],2);
Y=UpLow(:,2); UpLow(:,1)=Y-UpLow(:,1); UpLow(:,3)=UpLow(:,3)-Y; UpLow(:,2)=[];

PDF=PDF/NPDF;

fprintf('isimu=%d, %d%% done, Totally accepted: %d\n',...
    isimu,fix(isimu/nsimu*100),acce);

return



%%%%%%%%%%%%%%%%%%%auxiliary functions %%%%%%%%%%%%%%%%%%%%%%%%%%%

function y=getpar(options,par,default)
% GETPAR get parameter value from a struct
% options   options struct
% par       parameter value to extract from the struct
% default   default value if par is not a member of the options struct

if isfield(options,par)
    y = getfield(options,par); %#ok<GFLD>
elseif nargin>2
    y = default;
else
    error(sprintf('Need value for option: %s',par)); %#ok<SPERR>
end

%%% function for initial OPTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nsimu,par0,npar,bounds,ssfun,priorfun,adaptint,drscale,adascale, ...
    qcoveps,n0,n,sigma2,qcov,dodr,s20,s2chain,R2,iR,printint,verbosity] =  ...
    optionfun(model,params,options)

%% get values from the input structs
nsimu  = getpar(options,'nsimu',10000);
% initial parameter vector
par0   = getpar(params,'par0'); par0=par0(:)'; % row vector
% number of parameters
npar   = length(par0);
% 2*npar matrix of parameter bounds
bounds = getpar(params,'bounds',(ones(npar,2)*diag([-Inf,Inf]))');
% sum-of-squares function, ssfun(par,data),  -2*log(p(y|theta))
ssfun  = getpar(model,'ssfun');
% prior "sum-of-squares", -2*log(p(theta))
priorfun = getpar(model,'priorfun',inline('0','x','params'));

%%% parameters for DRAM
% how often to adapt, if zero, no adaptation
adaptint = getpar(options,'adaptint',100);
% scale for the second proposal, if zero, no DR
drscale  = getpar(options,'drscale',3);
% scale for adapting the propsal
adascale = getpar(options,'adascale',2.4/sqrt(npar));
% blow factor for covariace update
qcoveps  = getpar(options,'qcoveps',1e-5);

% precision of sigma2 as imaginative observations
%  if n0<0, no sigma2 update
n0  = getpar(params,'n0',-1);
% initial/prior value for the Gaussian error variance
sigma2 = getpar(params,'sigma2',1);
% number of observations (needed for sigma2 update)
n=[]; if n0>=0, n = getpar(params,'n'); end

qcov = getpar(options,'qcov'); % proposal covariance

% to DR or not to DR
if drscale<=0, dodr=0; else dodr=1;end

R = chol(qcov);  % Cholesky factor of proposal covariance
if dodr
    R2 = R./drscale;     % second proposal for DR try
    iR = inv(R);
else
    R2 = []; iR=[];
end

s20 = 0;
if n0>=0
    s2chain = zeros(nsimu,1);   % the sigma2 chain
    s20 = sigma2;
    if s20>0
        s2chain(1,:) = sigma2;
    end
else
    s2chain = [];
end

printint  = getpar(options,'printint',500);
verbosity = getpar(options,'verbosity',0);


%%%function for a one-stage DELAYED REJECTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [oldpar,oldss,oldprior,acce,yout,accept]=...
    drfun(data,params,bounds,ssfun,...
    R2,iR,alpha12,acce,...
    oldprior,newprior,oldss,newss,oldpar,newpar,yin,Temper,Sz,Sx)

npar    = length(oldpar);
% newpar2 = oldpar+randn(1,npar)*R2;  % a new try

accept=0; yout=yin;

NotConv=true(1);
while ( NotConv )
    % check until new probed parameters are within bounds
    newpar2 = oldpar+randn(1,npar)*R2;   % a new proposal
    if all(newpar2>bounds(1,:)) && all(newpar2<bounds(2,:))
        NotConv=false(1);
    end
end


if ( 1 ) % inside bounds
    
    [newss2 yout] = feval(ssfun,newpar2,data,Sz);
        
%     newprior2 = feval(priorfun,newpar2,params);
    if ( params.Test )
        newprior2 = oldprior;
    else
        newprior2 = (yout-params.Clima)'*(Sx\(Sx'\(yout-params.Clima)));
    end
    
    alpha32 = min(1,exp(-0.5*(newss-newss2) -0.5*(newprior-newprior2)).^Temper);
    l2 = exp(-0.5*(newss2-oldss) - 0.5*(newprior2-oldprior)).^Temper;
    q1 = exp(-0.5*(norm((newpar2-newpar)*iR)^2-norm((oldpar-newpar)*iR)^2)).^Temper;
    alpha13 = l2*q1*(1-alpha32)/(1-alpha12);
    
    if ( sum(yout) < eps )
        alpha13=0.0;
    end
    if rand < alpha13 % we accept
        accept = 1;
        acce     = acce+1;
        oldpar   = newpar2;
        oldss    = newss2;
        oldprior = newprior2;
    end
end


%%%function for VARIANCE/COVARIANCE SAMPLING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% not implemented
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%function for adaptation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [R,R2,iR]=amfun(chain,isimu,npar,qcoveps,adascale,verbosity,dodr,drscale)
global chaincov chainmean wsum lasti
%persistent chaincov chainmean wsum lasti

if verbosity, fprintf('adapting\n'); end
if isempty(lasti);lasti = 0; end

[chaincov,chainmean,wsum] = covupd(chain((lasti+1):isimu,:),1, ...
    chaincov,chainmean,wsum);
lasti   = isimu;
[Ra,is] = chol(chaincov + eye(npar)*qcoveps);
if is % singular cmat
    fprintf('Warning cmat singular, not adapting\n');
else
    R = Ra*adascale;R2 = [];iR=[];
    if dodr
        R2 = R./drscale;     % second proposal for DR try
        iR = inv(R);
    end
end





