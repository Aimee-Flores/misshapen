% findpt() -  Calculate peaks and troughs over time series
% Usage:
%  >> [peaks, troughs] = findpt(filteredsignal);
%
% Inputs:
%   x             = (array) raw voltage time series
%   xn            = (array) narrowband-filtered voltage time series
% Outputs:
%   Ps            = (array) indices of peaks in the input signal xn
%   Ts            = (array) indices of troughs in the input signal xn

function [Ps, Ts] = findpt(x, xn)
% Find zero crosses
fzerofall = @(data) find((diff(sign(data))) < 0);
fzerorise = @(data) find((diff(sign(data))) > 0);

zeroriseN = fzerorise(xn);
zerofallN = fzerofall(xn);

if numel(zeroriseN) == 0 || numel(zerofallN) == 0
    Ps = [];
    Ts = [];
else
    % Calculate number of peaks and troughs
    if zeroriseN(end) > zerofallN(end)
        P = numel(zeroriseN) - 1;
        T = numel(zerofallN);
    else
        P = numel(zeroriseN);
        T = numel(zerofallN) - 1;
    end
    
    % Calculate peak samples
    Ps = nan(P,1);
    for p = 1:P
        % Calculate the sample range between the most recent zero rise
        % and the next zero fall
        mrzerorise = zeroriseN(p);
        nfzerofall = zerofallN(zerofallN > mrzerorise);
        nfzerofall = nfzerofall(1);
        [~,argmax] = max(x(mrzerorise:nfzerofall));
        Ps(p) = argmax + mrzerorise - 1;
    end
    
    % Calculate trough samples
    Ts = nan(T, 1);
    for tr = 1:T
        % Calculate the sample range between the most recent zero fall
        % and the next zero rise
        mrzerofall = zerofallN(tr);
        nfzerorise = zeroriseN(zeroriseN > mrzerofall);
        nfzerorise = nfzerorise(1);
        [~,argmin] = min(x(mrzerofall:nfzerorise));
        Ts(tr) = argmin + mrzerofall - 1;
    end
end