% Esharp() -  Calculate sharpness of extrema in a timeseries
% Usage:
%  >> sharpness = Esharp(rawsignal, extremaInds, width, threshold, analyticAmp);
%
% Inputs:
%   x             = (array) 1-D signal; this signal should be as raw as possible
%   Es            = (array) time points of oscillatory peaks or troughs
%   widthS        = (int) Number of samples in each direction around extrema to use for sharpness estimation
%   ampPC         = (double) voltage threshold, determined using analytic amplitude 
%                   of oscillation of interest; only evaluate extrema above this threshold
%                   this threshold
%   amps          = (array) analytic amplitude of narrow bandpassed x
% Outputs:
%   sharpness     = (array) sharpness of each extrema is Es

function sharps = Esharp(x, Es, widthS, ampPC, amps)
E = numel(Es);
sharps = nan(E,1);
for e = 1:E
    Edata = x(Es(e)-widthS:Es(e)+widthS);
    sharps(e) = mean(abs(diff(Edata)));
end

if ampPC > 0
    amps = amps(Es);    
    sharps = sharps(amps>=ampPC);
end