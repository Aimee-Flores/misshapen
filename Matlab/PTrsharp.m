% PTrsharp() -  Calculate peak-trough sharpness ratio
% Usage:
%  >> peakTroughRatio = PTrsharp(rawsignal, peakInds, troughInds, width, normalize, peakortrough, threshold, analyticAmp);
%
% Inputs:
%   x             = (array) 1-D signal; this signal should be as raw as possible
%   Ps            = (array) time points of oscillatory peaks
%   Ts            = (array) time points of oscillatory troughs
%   widthS        = (int) Number of samples in each direction around extrema to use for sharpness estimation
%   normalize     = (boolean) if true, use normalized sharpness
%                   measure (EsharpN); else, use nonnormalized (Esharp)
%   pthent        = (boolean) if true, oscillation is peak to trough; else,
%                   trough to peak
%   ampPC         = (double) voltage threshold, determined using analytic amplitude 
%                   of oscillation of interest; only evaluate extrema above this threshold
%                   this threshold
%   amps          = (array) analytic amplitude of narrow bandpassed x
% Outputs:
%   ptr           = (array) peak-trough ratio of each oscillation

function ptr = PTrsharp(x, Ps, Ts, widthS, normalize, pthent, ampPC, amps)
% Calculate sharpness of peaks and troughs
if normalize
    Psharp = EsharpN(x, Ps, Ts, widthS, 0, amps);
    Tsharp = EsharpN(x, Ts, Ps, widthS, 0, amps);
    Ps = Ps(2:end-1);
    Ts = Ts(2:end-1);
else
    Psharp = Esharp(x, Ps, widthS, 0, amps);
    Tsharp = Esharp(x, Ts, widthS, 0, amps);
end

if numel(Ts) == 0 || numel(Ps) == 0
    ptr = nan;
% Align peak and trough arrays to one another
else
    if pthent
        if Ts(1) < Ps(1)
            Tsharp = Tsharp(2:end);
            Ts = Ts(2:end);
        end
        if numel(Psharp) == numel(Tsharp)+1
            Psharp = Psharp(1:end - 1);
            Ps = Ps(1:end - 1);
        end
    else
        if Ps(1) < Ts(1)
            Psharp = Psharp(2:end);
            Ps = Ps(2:end);
        end
        if numel(Tsharp) == numel(Psharp)+1
            Tsharp = Tsharp(1:end - 1);
            Ts = Ts(1:end - 1);
        end
    end

    ptr = Psharp./Tsharp;

    if ampPC > 0
        if pthent
            amps = amps(Ps);
        else
            amps = amps(Ts);
        end
        ptr = ptr(amps>=ampPC);
    end
end