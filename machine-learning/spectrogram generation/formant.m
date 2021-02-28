
function ffreq=formant(x,fs)
ncoeff=2+fs/1000;           % rule of thumb for formant estimation
a=lpc(x,ncoeff);
        % find frequencies by root-solving
r=roots(a);                  % find roots of polynomial a
r=r(imag(r)>0.01);           % only look for roots >0Hz up to fs/2
ffreq=sort(atan2(imag(r),real(r))*fs/(2*pi));
end