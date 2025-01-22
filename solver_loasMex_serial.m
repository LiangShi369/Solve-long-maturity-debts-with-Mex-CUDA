function [vp,vd,q,bp] = solver_loasMex_serial(y,b,pdfy,ua)

rstar = 0.01;   
theta= 0.0385;  
betta = 0.954; 
coup = 0.03; %0.03, long-term bond, coupon rate
eta = 0.05; %long-term bond, average maturity

ny = length(y);
nb = length(b);

vp = zeros(ny,nb); 
vd = zeros(ny,1); 
vo = zeros(ny,1); 
vpnew = zeros(ny,nb); 

bp = zeros(ny,nb); %debt policy function (expressed in indices)  
q = ones(ny,nb)/(1+rstar); %q is price of debt; it is a function of  (y_t, d_{t+1}) 
qnew = zeros(ny,nb);
W = zeros(1,nb,'double') ;
% WW = zeros(ny,nb,nb,'double') ;
% ww = zeros(1,nb);

%%%%%%%
diff = 1;
tol = 1e-7;
its = 1;
maxits = 2000;

% to incorporate taste shocks
epsi = 10e-16;
sigg_bp = 0.0002;
sigg_defp = 0.0001;
cv_bp = sigg_bp*log(epsi); % critical value

smctime   = tic;
totaltime = 0;

while diff > tol && its< maxits

    evg = betta*pdfy*vp; 

for iy = 1:ny  % change the loop to y first d second

    % W = zeros(1,nb,'double') ;
    qv = q(iy,:) ;
    probbp = zeros(nb,nb);

    for ib = 1:nb
        bib = b(ib);
        for i = 1: nb
            c = y(iy) - (eta+(1-eta)*coup)*bib + (b(i) - (1-eta)*bib).*qv(i)  ;
            if c <= 0
                W(i) = - Inf;
            else
                W(i) = 1 - c.^(-1) + evg(iy,i) ;
            end
        end
        vpnew(iy,ib) = max( W ) ; 
        indix = W - vpnew(iy,ib) - cv_bp > 0 ;           
        theExp = exp( (W(indix)- vpnew(iy,ib)) / sigg_bp ) ;
        probbp(ib,indix) = theExp ./ sum( theExp, 2) ;            
    end
    qnew(iy,:) =  (eta + (1-eta)*(coup + sum(probbp.*qv,2) )) ;
end

vdnew = ua + betta*pdfy*(theta*vo + (1-theta)*vd);
defp = 1./(1 + exp((vpnew - vdnew)/sigg_defp)) ;
qnew = (1 - defp).* qnew / (1+rstar) ;

qnew = pdfy*qnew  ;

vp_1 = max(repmat(vdnew,1,nb), vpnew);

diff = max(max(abs(qnew-q))) + max(max(abs(vp_1-vp))) + max(max(abs(vdnew-vd)));

vo = vp_1(:,1) ;
vp = vp_1 ; 
vd = vdnew ;
q = qnew ;

totaltime = totaltime + toc(smctime);
avgtime   = totaltime/its;
if mod(its, 40) == 0 || diff<=tol; fprintf('%8.0f ~%8.8f ~%8.5fs ~%8.5fs \n', its, diff, totaltime, avgtime); end
its = its+1;
smctime = tic; % re-start clock

end % end while

end % end function


