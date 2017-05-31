% symPT() -  Measure of asymmetry between the trough and subsequent peak of an oscillation
% Usage:
%  >> symmetry = symPT(rawsignal, peakInds, troughInds, windowSize, peakortrough, threshold, analyticAmp);
%
% Inputs:
%   x             = (array) 1-D signal; this signal should be as raw as possible
%   Ps            = (array) time points of oscillatory peaks
%   Ts            = (array) time points of oscillatory troughs
%   winsz         = (int) Number of samples in each direction around
%                   extrema to use for symmetry window
%   pthent        = (boolean) if true, oscillation is peak to trough; else,
%                   trough to peak
%   ampPC         = (double) voltage threshold, determined using analytic amplitude 
%                   of oscillation of interest; only evaluate extrema above this threshold
%                   this threshold
%   amps          = (array) analytic amplitude of narrow bandpassed x
% Outputs:
%   sym           = (array) measure of symmetry between each trough-peak pair
%                   Result of 0 means the peak and trough are perfectly symmetric

function sym = symPT(x, Ps, Ts, winsz, pthent, ampPC, amps)
if pthent
    Ps2 = Ts;
    Ts2 = Ps;
    Ps = Ps2;
    Ts = Ts2;
end
% Compare trough and its subsequent peak, so trough should be first
if Ps(1) < Ts(1)
    Ps = Ps(2:end);
end

% numel(Ps)
% numel(Ts)
% 
% Ps = Ps(Ps > winsz);
% Ts = Ts(Ts > winsz);
% Ps = Ps(Ps <= numel(x) - winsz);
% Ts = Ts(Ts <= numel(x) - winsz);
% 
% numel(Ps)
% numel(Ts)

E = numel(Ps);
sym = nan(E,1);
for e = 1:E
    if Ps(e) <= winsz || Ps(e) > numel(x) - winsz || Ts(e) <= winsz || Ts(e) > numel(x) - winsz
        continue
    end
    % Find region around each peak and trough. Make extrema be 0
    peak = x(Ps(e)-winsz:Ps(e)+winsz) - x(Ps(e));
    peak = -peak;
    trough = x(Ts(e)-winsz:Ts(e)+winsz) - x(Ts(e));
    
    % Compare the two measures
    peakenergy = sum(peak.^2);
    troughenergy = sum(trough.^2);
    energy = max([peakenergy,troughenergy]);
    diffenergy = sum((peak-trough).^2);
    sym(e) = diffenergy ./ energy;
end
if numel(Ts) > 0 && numel(Ps) > 0
    if ampPC > 0
        if Ts(1) < Ps(1)
            amps = amps(Ps(1:numel(sym)));
        else
            amps = amps(Ps(2:1+numel(sym)));
        end
        sym = sym(amps>=ampPC);
    end
end