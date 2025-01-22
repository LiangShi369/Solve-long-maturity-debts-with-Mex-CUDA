
clear

format compact

%Calibrated parameters
phi0 = -0.18819;
phi1 = 0.24558;
rstar = 0.01;   
theta= 0.0385;  
sigg = 2; 
betta = 0.954; 
coup = 0.03; %0.03, long-term bond, coupon rate
eta = 0.05; %0.05 long-term bond, average maturity

ny = 1225; % number of transitory shock grids
nb = 300; % # of grid points for debt 

% for transitory shocks
rhoy = 0.9485; %mean reversion of log prod  %% 0.86 from data
sdy = 0.027092;  %stdev of log prod shock 
width = 3.7;  % 3.7 (two-tail) sigma both sides covers 99.98% of the distribution
muy = 0; %long run mean of log prod

[ygrid,pdfy] = ar1_mex(ny,muy,rhoy,sdy,width);
y = exp(ygrid);
ny = numel(ygrid); %number of grid points for log of ouput

ya =  y - max(0,phi0*y + phi1*y.^2); %output in autarky

%debt grid
bupper = 1.4;    blower = 0;
b = blower:(bupper-blower)/(nb-1):bupper;
b = b(:);

n = ny*nb; %total number of states 

ua = (ya.^(1-sigg)-1) / (1-sigg);

%% Mex serial

cfg = coder.config('mex');
cfg.GenerateReport = false;

tic
codegen -config cfg solver_loasMex_serial -args {y,b,pdfy,ua} -o solver_loasMex_serial 
toc

[v,vd,q,bp] = solver_loasMex_serial(y,b,pdfy,ua) ;


%% Mex parfor

codegen -report solver_loasMex -args {y,b,pdfy,ua} 

cfg = coder.config('mex');
cfg.GenerateReport = false;

tic
codegen -config cfg solver_loasMex -args {y,b,pdfy,ua} -o solver_loasMex
toc

[v,vd,q,bp] = solver_loasMex(y,b,pdfy,ua) ;


%% Mex cuda
reset(gpuDevice)

cfg = coder.gpuConfig('mex');
cfg.GenerateReport = true;

b = gpuArray(b);
y = gpuArray(y);
pdfy = gpuArray(pdfy);
ua = gpuArray(ua);
tic
codegen -config cfg solver_gpu -args {y,b,pdfy,ua} -o solver_gpu
toc

[v,vd,q,bp] = solver_gpu(y,b,pdfy,ua) ;


