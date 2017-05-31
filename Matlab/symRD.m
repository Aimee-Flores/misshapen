% symRD() -  Measure of asymmetry between the rise and subsequent decay in an oscillation
% Usage:
%  >> symmetry = symRD(rawsignal, peakInds, troughInds, windowSize, riseordecay, threshold, analyticAmp);
%
% Inputs:
%   x             = (array) 1-D signal; this signal should be as raw as possible
%   Ps            = (array) time points of oscillatory peaks
%   Ts            = (array) time points of oscillatory troughs
%   winsz         = (int) Number of samples in each direction around
%                   extrema to use for symmetry window
%   rthend        = (boolean) if true, oscillation is rise to decay; else,
%                   decay to rise
%   ampPC         = (double) voltage threshold, determined using analytic amplitude 
%                   of oscillation of interest; only evaluate extrema above this threshold
%                   this threshold
%   amps          = (array) analytic amplitude of narrow bandpassed x
% Outputs:
%   sym           = (array) measure of symmetry between each rise-decay pair
%                   Result of 0 means the rise and decay are perfectly symmetric

function sym = symRD(x, Ps, Ts, winsz, rthend, ampPC, amps)
if rthend
    Ps2 = Ts;
    Ts2 = Ps;
    Ps = Ps2;
    Ts = Ts2;
end
% Compare rise and subsequent decay, so trough should be first
if Ps(1) < Ts(1)
    Ps = Ps(2:end);
end
if numel(Ps) == numel(Ts)
    P = numel(Ps) - 1;
else
    P = numel(Ps);
end
sym = nan(P,1);
for p = 1:P
    if Ps(p) <= winsz || Ps(p) > numel(x) - winsz
        continue
    end
    % Find regions for the rise and the decay
    rise = x(Ps(p)-winsz:Ps(p)) - x(Ps(p));
    decay = x(Ps(p):Ps(p)+winsz) - x(Ps(p));
    rise = flipud(rise);
    
    % Compare the two measures
    riseenergy = sum(rise.^2);
    decayenergy = sum(decay.^2);
    energy = max([riseenergy,decayenergy]);
    diffenergy = sum((rise-decay).^2);
    sym(p) = diffenergy ./ energy;
end

if ampPC > 0
    amps = amps(Ps(1:P));
    sym = sym(amps>=ampPC);
end