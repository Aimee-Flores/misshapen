% remove_nonextrema() -  removes peak/trough that are not local extrema,
%                        also removing the corresponding trough/peak
% Usage:
%  >> [newPeaks, newTroughs] = remove_nonextrema(rawsignal, peakInds, troughInds, peakortrough);
%
% Inputs:
%   x              = (array) voltage timeseries; this signal should be as raw as possible
%   Ps             = (array) time points of oscillatory peaks
%   Ts             = (array) time points of oscillatory troughs
%   pthend         = if true, oscillation is peak and subsequent trough;
%                    trough and subsequent peak otherwise
% Outputs:
%   Pnew           = indices of valid peaks
%   Tnew           = indices of valid troughs
%

function [Pnew, Tnew] = remove_nonextrema(x, Ps, Ts, pthend)
Pnew = Ps;
Tnew = Ts;

if Ps(1) < Ts(1)
    pfirst = true;
else
    pfirst = false;
end

extrema = {Ps, Ts};

% go through each peak, then each trough
for eType = 1:numel(extrema)
    eTypeCurr = extrema{eType};
    E = numel(eTypeCurr);
    
    for e = 1:E
        
        eCurr = eTypeCurr(e);
        remove = false;
        
        % designate indices that are too close to boundary as nonextrema by
        % default
        if eCurr == 1 || eCurr == numel(x)
            remove = true;
            
            % non extrema if signal is decreasing
        elseif x(eCurr-1) >= x(eCurr) && x(eCurr+1) <= x(eCurr)
            remove = true;
            
            % non extrema if signal is increasing
        elseif x(eCurr-1) <= x(eCurr) && x(eCurr+1) >= x(eCurr)
            remove = true;
        end
        
        if remove
            %             disp([num2str(eType) ' ' num2str(eCurr)])
            % find corresponding peak or trough to remove
            %             if pthend && eType == 1
            %                 eRemove = find(Ts > eCurr, 1);
            %             elseif ~pthend && eType == 1
            %                 eRemove = find(Ts < eCurr, 1, 'last');
            %             elseif pthend && eType == 2
            %                 eRemove = find(Ps < eCurr, 1, 'last');
            %             else
            %                 eRemove = find(Ps > eCurr, 1);
            %             end
            
            eRemove = e;
            
            % mark peaks and troughs to remove
            if eType == 1
                Pnew(e) = -1;
                Tnew(eRemove) = -1;
            else
                Tnew(e) = -1;
                Pnew(eRemove) = -1;
            end
        end
    end
end

% remove nonextrema
Pnew = Pnew(Pnew ~= -1);
Tnew = Tnew(Tnew ~= -1);