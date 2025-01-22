function [vp,vd,q,bp] = solver_gpu(y,b,pdfy,ua)

coder.gpu.kernelfun;

rstar = 0.01;   
theta= 0.0385;  
betta = 0.954; 
coup = 0.03; %0.03, long-term bond, coupon rate
eta = 0.05; %long-term bond, average maturity

ny = length(y);
nb = length(b);

vp = coder.nullcopy(zeros(ny,nb));  %continue repaying
vd = coder.nullcopy(zeros(ny,1)) ; 
vo = coder.nullcopy(zeros(ny,1));
vpnew = coder.nullcopy(zeros(ny,nb)) ;
WW = coder.nullcopy(zeros(nb,nb,ny)) ;
theExp = coder.nullcopy( zeros(1,nb) ) ;
% theExpQ = coder.nullcopy( zeros(1,nb) ) ;

bp = coder.nullcopy(zeros(ny,nb)) ; %debt policy function (expressed in indices)  
q = ones(ny,nb)/(1+rstar) ; %q is price of debt; it is a function of  (y_t, d_{t+1}) 
qnew = coder.nullcopy(zeros(ny,nb)) ;

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

    evp = betta*pdfy*vp ;
  
  coder.gpu.kernel()
  for iy = 1:ny  % change the loop to y first d second

      qv = q(iy,:) ;
      yiy = y(iy);

    for ib = 1:nb
        
        bib = b(ib) ;

        for i = 1: nb

            c = yiy - (eta+(1-eta)*coup) * bib + (b(i)-(1-eta)*bib) * qv(i) ;

            if c <= 0
                WW(i, ib, iy) = - Inf;
            else
                WW(i, ib, iy) = 1 - c^(-1) + evp(iy,i) ;
            end

        end
        vpnew(iy, ib)  = max( WW(:, ib, iy) ) ;
    end
  end


coder.gpu.kernel()
for iy = 1:ny 
    for ib = 1:nb
        for i = 1:nb

            temp = WW(i,ib,iy) - vpnew(iy,ib) - cv_bp ;
            
            if temp > 0
                theExp(i) = exp( (temp + cv_bp) / sigg_bp ) ;
                % theExpQ(i) = theExp(i) * q(iy,i) ;
            else
                theExp(i) = 0 ;
                % theExpQ(i) = 0 ;
            end           
        end
        qnew(iy,ib) = (eta + (1-eta)*(coup + sum(theExp.*q(iy,:))/sum(theExp) ) ) ;
        % qnew(iy,ib) = (eta + (1-eta)*(coup + sum(theExpQ)/sum(theExp) ) ) ;
    end
end

vdnew = ua + betta * pdfy * ( theta*vo + (1-theta)*vd );
defp = 1./( 1 + exp( (vpnew - vdnew)/sigg_defp ) ) ;
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


