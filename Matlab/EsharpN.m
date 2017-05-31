% EsharpN() -  Calculate sharpness of extrema in a timeseries
% Usage:
%  >> sharpness = Esharp(rawsignal, extremaInds, width, threshold, analyticAmp);
%
% Inputs:
%   x             = (array) 1-D signal; this signal should be as raw as possible
%   Es            = (array) time points of oscillatory peaks or troughs
%   EsOpp         = (array) time points of the opposite extrema
%   widthS        = (int) Number of samples in each direction around extrema to use for sharpness estimation
%   ampPC         = (double) voltage threshold, determined using analytic amplitude 
%                   of oscillation of interest; only evaluate extrema above this threshold
%                   this threshold
%   amps          = (array) analytic amplitude of narrow bandpassed x
% Outputs:
%   sharpness     = (array) sharpness of each extrema in Es

function sharps = EsharpN(x, Es, EsOpp, widthS, ampPC, amps)
% Add offset if the first extrema is the opposite one
if Es(1) < EsOpp(1)
    offset = 1;
else
    offset = 0;
end
E = numel(Es) - 2;
sharps = nan(E,1);
for epre = 1:E
    e = epre + 1;
    sharprise = (x(Es(e)) - x(Es(e)-widthS)) / (x(Es(e)) - x(EsOpp(e-offset)));
    sharpdeca = (x(Es(e)) - x(Es(e)+widthS)) / (x(Es(e)) - x(EsOpp(e-offset+1)));
    sharps(epre) = mean([sharprise,sharpdeca]);
end

if ampPC > 0
    amps = amps(Es(2:end-1));
    sharps = sharps(amps>=ampPC);
end