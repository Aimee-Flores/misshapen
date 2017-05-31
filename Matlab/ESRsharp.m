% ESRsharp() -  Calculate extrema sharpness ratio: the peak/trough sharpness ratio
%               but fixed to be above 1.
%               Pairs are peaks and subsequent troughs
% Usage:
%  >> extremaSharpnessRatio = ESRsharp(rawsignal, peakInds, troughInds, width, normalize, peakortrough, threshold, analyticAmp);
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
%   esr           = (array) extrema sharpness ratio for each period
function esr = ESRsharp(x, Ps, Ts, widthS, normalize, pthent, ampPC, amps)
PTr = PTrsharp(x, Ps, Ts, widthS, normalize, pthent, ampPC, amps);
PTr = PTr(:);
esr = max([PTr, 1./PTr], [], 2);
