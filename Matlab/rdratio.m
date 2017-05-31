% rdratio() - Calculate the ratio between rise time and decay time for oscillations
%    Define an oscillation as trough-to-trough, so we're comparing a rise to the
%    subsequent decay
% Usage:
%  >> risedecayratio = rdratio(peakInds, troughInds, threshold, analyticAmp);
%
% Inputs:
%   Ps            = (array) time points of oscillatory peaks
%   Ts            = (array) time points of oscillatory troughs
%   ampPC         = (double) voltage threshold, determined using analytic amplitude 
%                   of oscillation of interest; only evaluate extrema above this threshold
%                   this threshold
%   amps          = (array) analytic amplitude of narrow bandpassed x
% Outputs:
%   rdr           = (array) rise-decay ratios for each oscillation

function rdr = rdratio(Ps, Ts, ampPC, amps)
if Ts(1) < Ps(1)
    if numel(Ts) == numel(Ps) + 1
        riset = Ps - Ts(1:end - 1);
        decayt = Ts(2:end) - Ps;
    elseif numel(Ps) == numel(Ts)
        riset = Ps(1:end - 1) - Ts(1:end-1);
        decayt = Ts(2:end) - Ps(1:end-1);
    end
else
    if numel(Ps) == numel(Ts) + 1
        riset = Ps(2:end) - Ts;
        decayt = Ts - Ps(1:end-1);
    elseif numel(Ps) == numel(Ts)
        riset = Ps(2:end) - Ts(1:end-1);
        decayt = Ts(2:end) - Ps(2:end);
    end
end
rdr = riset./decayt;

if ampPC > 0
    if Ts(1) < Ps(1)
        amps = amps(Ps(1:numel(rdr)));
    else
        amps = amps(Ps(2:1+numel(rdr)));
    end
    rdr = rdr(amps>=ampPC);
end

