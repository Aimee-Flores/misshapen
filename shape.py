"""
This library contains metrics to quantify the shape of a waveform
1. threshold_amplitude - only look at a metric while oscillatory amplitude is above a set percentile threshold
2. rdratio - Ratio of rise time and decay time
3. pt_duration - Peak and trough durations and their ratio
3. symPT - symmetry between peak and trough
4. symRD - symmetry between rise and decay
5. pt_sharp - calculate sharpness of oscillatory extrema
6. rd_steep - calculate rise and decay steepness
7. ptsr - calculate extrema sharpness ratio
8. rdsr - calculate rise-decay steepness ratio
9. average_waveform_trigger - calculate the average waveform of an oscillation by triggering on peak or trough
10. gips_swm - identify a repeated waveform in the signal
11. rd_diff - normalized difference between rise and decay time
12. compute_shape_by_cycle - make a dataframe of shape features for each cycle
13. define_true_oscillating_cycles - determine which cycles are part of an oscillating period
"""

from __future__ import division
import numpy as np
import pandas as pd
from misshapen.nonshape import ampT, bandpass_default, findpt, findzerox


def threshold_amplitude(x, metric, samples, percentile, frange, Fs, filter_fn=None, filter_kwargs=None):
    """
    Exclude from analysis the samples in which the amplitude falls below a defined percentile

    Parameters
    ----------
    x : numpy array
        raw time series
    metric : numpy array
        series of measures corresponding to time samples in 'samples' (e.g. peak sharpness)
    samples : numpy array
        time samples at which metric was computer (e.g. peaks)
    percentile : float
        percentile cutoff for exclusion (e.g. 10 = bottom 10% excluded)
    frange : [lo, hi]
        frequency range of interest for calculating amplitude
    Fs : float
        Sampling rate (Hz)

    Returns
    -------
    metric_new : numpy array
        same as input 'metric' but only for samples above the amplitude threshold
    samples_new : numpy array
        samples above the amplitude threshold
    """

    # Do nothing if threshold is 0
    if percentile == 0:
        return metric, samples

    # Default filter function
    if filter_fn is None:
        filter_fn = bandpass_default
    if filter_kwargs is None:
        filter_kwargs = {}

    # Calculate amplitude time series and threshold
    amp = ampT(x, frange, Fs, rmv_edge=False,
               filter_fn=filter_fn, filter_kwargs=filter_kwargs)
    amp = amp[samples]
    amp_threshold = np.percentile(amp, percentile)

    # Update samples used
    samples_new = samples[amp >= amp_threshold]
    metric_new = metric[amp >= amp_threshold]

    return metric_new, samples_new


def rdratio(Ps, Ts):
    """
    Calculate the ratio between rise time and decay time for oscillations

    Note: must have the same number of peaks and troughs
    Note: the final rise or decay is unused

    Parameters
    ----------
    Ps : numpy arrays 1d
        time points of oscillatory peaks
    Ts : numpy arrays 1d
        time points of osillatory troughs

    Returns
    -------
    rdr : array-like 1d
        rise-decay ratios for each oscillation
    """

    # Assure input has the same number of peaks and troughs
    if len(Ts) != len(Ps):
        raise ValueError('Length of peaks and troughs arrays must be equal')

    # Assure Ps and Ts are numpy arrays
    if type(Ps) == list or type(Ts) == list:
        print('Converted Ps and Ts to numpy arrays')
        Ps = np.array(Ps)
        Ts = np.array(Ts)

    # Calculate rise and decay times
    if Ts[0] < Ps[0]:
        riset = Ps[:-1] - Ts[:-1]
        decayt = Ts[1:] - Ps[:-1]
    else:
        riset = Ps[1:] - Ts[:-1]
        decayt = Ts[:-1] - Ps[:-1]

    # Calculate ratio between each rise and decay time
    rdr = riset / decayt.astype(float)

    return riset, decayt, rdr


def pt_duration(Ps, Ts, zeroxR, zeroxD):
    """
    Calculate the ratio between peak and trough durations

    NOTE: must have the same number of peaks and troughs
    NOTE: the durations of the first and last extrema will be estimated by using the only zerox they have

    Parameters
    ----------
    Ps : numpy arrays 1d
        time points of oscillatory peaks
    Ts : numpy arrays 1d
        time points of osillatory troughs
    zeroxR : array-like 1d
        indices at which oscillatory rising zerocrossings occur
    zeroxD : array-like 1d
        indices at which oscillatory decaying zerocrossings occur

    Returns
    -------
    Ps_dur : array-like 1d
        peak-trough duration ratios for each oscillation
    Ts_dur : array-like 1d
        peak-trough duration ratios for each oscillation
    ptr : array-like 1d
        peak-trough duration ratios for each oscillation
    """

    # Assure input has the same number of peaks and troughs
    if len(Ts) != len(Ps):
        raise ValueError('Length of peaks and troughs arrays must be equal')

    # Assure Ps and Ts are numpy arrays
    if type(Ps) == list or type(Ts) == list:
        print('Converted Ps and Ts to numpy arrays')
        Ps = np.array(Ps)
        Ts = np.array(Ts)

    # Calculate the duration of each peak and trough until last
    Ps_dur = np.zeros(len(Ps))
    Ts_dur = np.zeros(len(Ts))
    if Ps[0] < Ts[0]:
        # treat first extrema differently
        Ps_dur[0] = 2 * (zeroxD[0] - Ps[0])
        # duration of each peak
        for i in range(1, len(Ps) - 1):
            Ps_dur[i] = (zeroxD[i] - zeroxR[i - 1])
        # duration of each trough
        for i in range(len(Ts) - 1):
            Ts_dur[i] = (zeroxR[i] - zeroxD[i])
    else:
        Ts_dur[0] = 2 * (zeroxR[0] - Ts[0])
        for i in range(len(Ps) - 1):
            Ps_dur[i] = (zeroxD[i] - zeroxR[i])
        # duration of each trough
        for i in range(1, len(Ts) - 1):
            Ts_dur[i] = (zeroxR[i] - zeroxD[i - 1])

    # Treat last extrema differently
    if Ps[-1] < Ts[-1]:
        Ps_dur[-1] = (zeroxD[-1] - zeroxR[-1])
        Ts_dur[-1] = 2 * (Ts[-1] - zeroxD[-1])
    else:
        Ps_dur[-1] = 2 * (Ps[-1] - zeroxR[-1])
        Ts_dur[-1] = (zeroxR[-1] - zeroxD[-1])

    ptr = Ps_dur / Ts_dur
    return Ps_dur, Ts_dur, ptr


def symPT(x, Ps, Ts, window_half):
    """
    Measure of asymmetry between oscillatory peaks and troughs

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : array-like 1d
        time points of oscillatory peaks
    Ts : array-like 1d
        time points of oscillatory troughs
    window_half : int
        Number of samples around extrema to analyze, in EACH DIRECTION

    Returns
    -------
    sym : array-like 1d
        measure of symmetry between each trough-peak pair
        Result of 0 means the peak and trough are perfectly symmetric

    Notes
    -----
    Opt 2: Roemer; The metric should be between 0 and 1
    Inner product of Peak and Trough divided by the squareroot of the product of SSQ_peak and SSQ_trough

    I'll need to fine tune this to make it more complicated and less susceptible to noise
    """

    # Assure input has the same number of peaks and troughs
    if len(Ts) != len(Ps):
        raise ValueError('Length of peaks and troughs arrays must be equal')

    E = len(Ps)
    sym = np.zeros(E)
    for e in range(E):
        # Find region around each peak and trough. Make extrema be 0
        peak = x[Ps[e] - window_half:Ps[e] + window_half + 1] - x[Ps[e]]
        peak = -peak
        trough = x[Ts[e] - window_half:Ts[e] + window_half + 1] - x[Ts[e]]

        # Compare the two measures
        peakenergy = np.sum(peak**2)
        troughenergy = np.sum(trough**2)
        energy = np.max((peakenergy, troughenergy))
        diffenergy = np.sum((peak - trough)**2)
        sym[e] = diffenergy / energy

    return sym


def symRD(x, Ts, window_full):
    """
    Measure of asymmetry between oscillatory peaks and troughs

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ts : array-like 1d
        time points of oscillatory troughs
    window_full : int
        Number of samples after peak to analyze for decay and before peak to analyze for rise

    Returns
    -------
    sym : array-like 1d
        measure of symmetry between each rise and decay
    """

    T = len(Ts)
    sym = np.zeros(T)
    for t in range(T):
        # Find regions for the rise and the decay
        rise = x[Ts[t]:Ts[t] + window_full + 1] - x[Ts[t]]
        decay = x[Ts[t] - window_full:Ts[t] + 1] - x[Ts[t]]

        # Ensure the minimum value is 0
        rise[rise < 0] = 0
        decay[decay < 0] = 0

        # Make rises and decays go the same direction
        rise = np.flipud(rise)

        # Calculate absolute difference between each point in the rise and
        # decay
        diffenergy = np.sum(np.abs(rise - decay))

        # Normalize this difference by the max voltage value at each point
        rise_decay_maxes = np.max(np.vstack((rise, decay)), axis=0)
        energy = np.sum(rise_decay_maxes)

        # Compare the two measures
        sym[t] = diffenergy / energy

    return sym


def pt_sharp(x, Ps, Ts, window_half, method='diff'):
    """
    Calculate the sharpness of extrema

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : array-like 1d
        time points of oscillatory peaks
    Ts : array-like 1d
        time points of oscillatory troughs
    window_half : int
        Number of samples in each direction around extrema to use for sharpness estimation

    Returns
    -------
    Psharps : array-like 1d
        sharpness of peaks
    Tsharps : array-like 1d
        sharpness of troughs

    """

    # Assure input has the same number of peaks and troughs
    if len(Ts) != len(Ps):
        raise ValueError('Length of peaks and troughs arrays must be equal')

    # Calculate the sharpness of each peak
    P = len(Ps)
    Psharps = np.zeros(P)
    for e in range(P):
        if method == 'deriv':
            Edata = x[Ps[e] - window_half: Ps[e] + window_half + 1]
            Psharps[e] = np.mean(np.abs(np.diff(Edata)))
        elif method == 'diff':
            Psharps[e] = np.mean(
                (x[Ps[e]] - x[Ps[e] - window_half], x[Ps[e]] - x[Ps[e] + window_half]))

    T = len(Ts)
    Tsharps = np.zeros(T)
    for e in range(T):
        if method == 'deriv':
            Edata = x[Ts[e] - window_half: Ts[e] + window_half + 1]
            Tsharps[e] = np.mean(np.abs(np.diff(Edata)))
        elif method == 'diff':
            Tsharps[e] = np.mean(
                (x[Ts[e] - window_half] - x[Ts[e]], x[Ts[e] + window_half] - x[Ts[e]]))

    return Psharps, Tsharps


def rd_steep(x, Ps, Ts):
    """
    Calculate the max steepness of rises and decays

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : array-like 1d
        time points of oscillatory peaks
    Ts : array-like 1d
        time points of oscillatory troughs

    Returns
    -------
    risesteep : array-like 1d
        max steepness in each period for rise
    decaysteep : array-like 1d
        max steepness in each period for decay
    """

    # Assure input has the same number of peaks and troughs
    if len(Ts) != len(Ps):
        raise ValueError('Length of peaks and troughs arrays must be equal')

    # Calculate rise and decay steepness
    E = len(Ps) - 1
    risesteep = np.zeros(E)
    for t in range(E):
        if Ts[0] < Ps[0]:
            rise = x[Ts[t]:Ps[t] + 1]
        else:
            rise = x[Ts[t]:Ps[t + 1] + 1]
        risesteep[t] = np.max(np.diff(rise))

    decaysteep = np.zeros(E)
    for p in range(E):
        if Ts[0] < Ps[0]:
            decay = x[Ps[p]:Ts[p + 1] + 1]
        else:
            decay = x[Ps[p]:Ts[p] + 1]
        decaysteep[p] = -np.min(np.diff(decay))

    return risesteep, decaysteep


def ptsr(Psharp, Tsharp, log=True, polarity=True):
    if polarity:
        sharpnessratio = Psharp / Tsharp
    else:
        sharpnessratio = np.max((Psharp / Tsharp, Tsharp / Psharp))
    if log:
        sharpnessratio = np.log10(sharpnessratio)
    return sharpnessratio


def rdsr(Rsteep, Dsteep, log=True, polarity=True):
    if polarity:
        steepnessratio = Rsteep / Dsteep
    else:
        steepnessratio = np.max((Rsteep / Dsteep, Dsteep / Rsteep))
    if log:
        steepnessratio = np.log10(steepnessratio)
    return steepnessratio


def average_waveform_trigger(x, f_range, Fs, avgwave_halflen, trigger='trough'):
    """
    Calculate the average waveform of a signal by triggering on the peaks or troughs

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    f_range : (low, high), Hz
        frequency range for narrowband signal of interest
    Fs : float
        The sampling rate
    avgwave_halflen : float
        length of time for the averaged signal to be recorded in the positive and negative direction
    trigger : str
        'trough' to trigger the averaging on each trough
        'peak' to trigger the averaging on each peak

    Returns
    -------
    avg_wave : array-like 1d
        the average waveform in 'x' in the frequency 'f_range' triggered on 'trigger'

    """
    # Set up the parameters for averaging
    dt = 1 / float(Fs)
    t_avg_wave = np.arange(-avgwave_halflen, avgwave_halflen + dt, dt)
    N_samples_halflen = int(avgwave_halflen * Fs)

    # Find the trigger points for averaging
    Ps, Ts = findpt(x, f_range, Fs, boundary=N_samples_halflen + 1)
    if trigger == 'trough':
        trig_samps = Ts
    elif trigger == 'peak':
        trig_samps = Ps
    else:
        raise ValueError('Trigger not implemented')

    # Do the averaging at each trigger
    avg_wave = np.zeros(int(N_samples_halflen * 2 + 1))
    N_triggers = len(trig_samps)
    for i in range(N_triggers):
        avg_wave += x[trig_samps[i] -
                      N_samples_halflen:trig_samps[i] + N_samples_halflen + 1]
    avg_wave = avg_wave / N_triggers
    return t_avg_wave, avg_wave


def gips_swm(x, Fs, L, G,
             max_iterations=100, T=1, window_starts_custom=None):
    """
    Sliding window matching methods to find recurring patterns in a time series
    using the method by Bart Gips in J Neuro Methods 2017.
    See matlab code at: https://github.com/bartgips/SWM

    Calculate the average waveform of a signal by triggering on the peaks or troughs

    Note should high-pass if looking at high frequency activity so that it does not converge on a low frequency motif

    L and G should be chosen to be about the size of the motif of interest, and the N derived should be about the number of occurrences

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Fs : float
        The sampling rate (samples per second)
    L : float
        Window length (seconds)
    G : float
        Minimum window spacing (seconds)
    T : float
        temperature parameter. Controls acceptance probability
    max_iterations : int
        Maximum number of iterations for the pattern finder
    window_starts_custom : np.ndarray (1d)
        Pre-set locations of initial windows (instead of evenly spaced by 2G)

    Returns
    -------
    avg_wave : np.ndarray (1d)
        the average waveform in 'x' in the frequency 'f_range' triggered on 'trigger'
    window_starts : np.ndarray (1d)
        indices at which each window begins for the final set of windows
    J : np.ndarray (1d)
        History of costs
    """

    # Initialize window positions, separated by 2*G
    L_samp = int(L * Fs)
    G_samp = int(G * Fs)
    if window_starts_custom is None:
        window_starts = np.arange(0, len(x) - L_samp, 2 * G_samp)
    else:
        window_starts = window_starts_custom

    # Calculate the total number of windows
    N_windows = len(window_starts)

    # Calculate initial cost
    J = np.zeros(max_iterations)
    J[0] = _gips_compute_J(x, window_starts, L_samp)

    # Randomly sample windows with replacement
    random_window_idx = np.random.choice(range(N_windows), size=max_iterations)

    # Optimize X
    iter_num = 1
    while iter_num < max_iterations:
        print(iter_num)

        # Pick a random window position
        window_idx_replace = random_window_idx[iter_num]

        # Find a new allowed position for the window
        # OH. CHANGE IT IN THE WINDOW ARRAY. at the end have all windows
        window_starts_temp = np.copy(window_starts)
        window_starts_temp[window_idx_replace] = _gips_find_new_windowidx(
            window_starts, G_samp, L_samp, len(x) - L_samp)

        # Calculate the cost
        J_temp = _gips_compute_J(x, window_starts_temp, L_samp)

        # Calculate the change in cost function
        deltaJ = J_temp - J[iter_num - 1]

        # Calculate the acceptance probability
        p_accept = np.exp(-deltaJ / float(T))

        # Accept update to J with a certain probability
        if np.random.rand() < p_accept:
            # Update J
            J[iter_num] = J_temp
            # Update X
            window_starts = window_starts_temp
        else:
            # Update J
            J[iter_num] = J[iter_num - 1]

        # Update iteration number
        iter_num += 1

    # Calculate average wave
    avg_wave = np.zeros(L_samp)
    for w in range(N_windows):
        avg_wave = avg_wave + x[window_starts[w]:window_starts[w] + L_samp]
    avg_wave = avg_wave / float(N_windows)

    return avg_wave, window_starts, J


def _gips_compute_J(x, window_starts, L_samp):
    """Compute the cost, which is the average distance between all windows"""

    # Get all windows and zscore them
    N_windows = len(window_starts)
    windows = np.zeros((N_windows, L_samp))
    for w in range(N_windows):
        temp = x[window_starts[w]:window_starts[w] + L_samp]
        windows[w] = (temp - np.mean(temp)) / np.std(temp)

    # Calculate distances for all pairs of windows
    d = []
    for i in range(N_windows):
        for j in range(i + 1, N_windows):
            window_diff = windows[i] - windows[j]
            d_temp = 1 / float(L_samp) * np.sum(window_diff**2)
            d.append(d_temp)
    # Calculate cost
    J = 1 / float(2 * (N_windows - 1)) * np.sum(d)
    return J


def _gips_find_new_windowidx(window_starts, G_samp, L_samp, N_samp):
    """Find a new sample for the starting window"""

    found = False
    while found is False:
        # Generate a random sample
        new_samp = np.random.randint(N_samp)
        # Check how close the sample is to other window starts
        dists = np.abs(window_starts - new_samp)
        if np.min(dists) > G_samp:
            return new_samp


def rd_diff(Ps, Ts):
    """
    Calculate the normalized difference between rise and decay times,
    as Gips, 2017 refers to as the "skewnwss index"
    SI = (T_up-T_down)/(T_up+T_down)

    Parameters
    ----------
    Ps : numpy arrays 1d
        time points of oscillatory peaks
    Ts : numpy arrays 1d
        time points of osillatory troughs

    Returns
    -------
    rdr : array-like 1d
        rise-decay ratios for each oscillation
    """

    # Assure input has the same number of peaks and troughs
    if len(Ts) != len(Ps):
        raise ValueError('Length of peaks and troughs arrays must be equal')

    # Assure Ps and Ts are numpy arrays
    if type(Ps) == list or type(Ts) == list:
        print('Converted Ps and Ts to numpy arrays')
        Ps = np.array(Ps)
        Ts = np.array(Ts)

    # Calculate rise and decay times
    if Ts[0] < Ps[0]:
        riset = Ps[:-1] - Ts[:-1]
        decayt = Ts[1:] - Ps[:-1]
    else:
        riset = Ps[1:] - Ts[:-1]
        decayt = Ts[:-1] - Ps[:-1]

    # Calculate ratio between each rise and decay time
    rdr = (riset - decayt) / float(riset + decayt)
    return riset, decayt, rdr


def compute_shape_by_cycle(x, f_range, Fs,
                           findpt_kwargs=None,
                           define_true_oscillating_periods_kwargs=None):
    """
    Calculate several features of an oscillation's waveform
    shape for each cycle in the recording.

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    f_range : (low, high), Hz
        frequency range for narrowband signal of interest
    Fs : float
        The sampling rate (default = 1000Hz)
    findpt_kwargs : dict or None
        Keyword arguments for function to find peaks and
        troughs (nonshape.findpt)
    define_true_oscillating_periods_kwargs : dict or None
        Keyword arguments for function to find label cycles
        as in or not in an oscillation

    Returns
    -------
    df_P : pandas DataFrame
        features of the waveform shape of each peak:
        sample : sample of 'x' at which the peak occurs
        sample_zeroxD : sample of the decaying zerocrossing
        sample_zeroxR : sample of the rising zerocrossing
        sample_lastE : sample of the last trough
        sample_nextE : sample of the next trough
        period : period of the cycle
        half_decay_time : time between peak and decay zerocross
        half_rise_time : time rise zerocross and peak
        whole_decay_time : time between peak and next trough
        whole_rise_time : time rise zerocross and previous trough
        peak_time : time between rise and decay zerocrosses
        half_decay_volt : voltage change between peak and decay zerocross
        half_rise_volt : voltage change between peak and rise zerocross
        whole_decay_volt : voltage change between peak and next trough
        whole_rise_volt : voltage change between peak and previous trough
        peak_volt : voltage at the peak
        half_decay_sharp : steepness between peak and decay zerocross
        half_rise_sharp : steepness between peak and rise zerocross
        whole_decay_sharp : steepness between peak and next trough
        whole_rise_sharp : steepness between peak and previous trough
        peak_sharp : sharpness of peak
        rdsym_time : asymmetry between the whole rise and decay times
        rdsym_volt : asymmetry between the whole rise and decay voltages
        rdsym_sharp : asymmetry between the whole rise and decay steepnesses

    df_T : pandas DataFrame
        features of the waveform shape of each trough.
        Similar features as df_P

    .. note:: First extrema is peak
              The first extrema analyzed will be a peak,
              and the final one a trough. In order to switch
              the preference, simply invert the polarity of x.
    """

    # Set defaults if user input is None
    if findpt_kwargs is None:
        findpt_kwargs = {}
    else:
        # Raise warning if switch from peak start to trough start
        if 'forcestart' in findpt_kwargs.keys():
            if findpt_kwargs['forcestart'] == 'trough':
                print('WARNING: This function has been designed to assume that\
                       the first extrema identified will be a peak. This has\
                       been overwritten. Proceed with caution.')
    if define_true_oscillating_periods_kwargs is None:
        define_true_oscillating_periods_kwargs = {}

    # Find peak and trough locations in the signal
    Ps, Ts = findpt(x, f_range, Fs, boundary=None, **findpt_kwargs)

    # Find zero-crossings
    zeroxR, zeroxD = findzerox(x, Ps, Ts)

    # Compute stats on peak-to-peak cycles (trough-centered)
    N_p2p = len(Ps) - 1
    trough_stats = {}

    # Define important samples
    trough_stats['sample'] = Ts[:-1]
    trough_stats['sample_zeroxD'] = zeroxD[:-1]
    trough_stats['sample_zeroxR'] = zeroxR
    trough_stats['sample_lastE'] = Ps[:-1]
    trough_stats['sample_nextE'] = Ps[1:]

    # Compute other statistics of the cycle centered around the trough
    trough_stats['period'] = trough_stats['sample_nextE'] - \
        trough_stats['sample_lastE']
    trough_stats['trough_volt'] = x[Ts[:-1]]

    trough_stats['half_decay_time'] = (Ts[:-1] - zeroxD[:-1])
    trough_stats['half_rise_time'] = (zeroxR - Ts[:-1])
    trough_stats['trough_time'] = trough_stats['half_decay_time'] + \
        trough_stats['half_rise_time']

    trough_stats['half_decay_volt'] = x[zeroxD[:-1]] - x[Ts[:-1]]
    trough_stats['half_rise_volt'] = x[zeroxR] - x[Ts[:-1]]

    trough_stats['half_decay_sharp'] = trough_stats['half_decay_volt'] / \
        trough_stats['half_decay_time']
    trough_stats['half_rise_sharp'] = trough_stats['half_rise_volt'] / \
        trough_stats['half_rise_time']
    trough_stats['trough_sharp'] = (
        trough_stats['half_decay_sharp'] + trough_stats['half_rise_sharp']) / 2.

    trough_stats['whole_decay_time'] = (Ts[:-1] - Ps[:-1])
    trough_stats['whole_rise_time'] = (Ps[1:] - Ts[:-1])
    trough_stats['rdsym_time'] = trough_stats['whole_decay_time'] - \
        trough_stats['whole_rise_time']

    trough_stats['whole_decay_volt'] = x[Ps[:-1]] - x[Ts[:-1]]
    trough_stats['whole_rise_volt'] = x[Ps[1:]] - x[Ts[:-1]]
    trough_stats['rdsym_volt'] = trough_stats['whole_decay_volt'] - \
        trough_stats['whole_rise_volt']

    trough_stats['whole_decay_sharp'] = trough_stats['whole_decay_volt'] / \
        trough_stats['whole_decay_time']
    trough_stats['whole_rise_sharp'] = trough_stats['whole_rise_volt'] / \
        trough_stats['whole_rise_time']
    trough_stats['rdsym_sharp'] = trough_stats['whole_decay_sharp'] - \
        trough_stats['whole_rise_sharp']

    # Compute stats on trough-to-trough cycles (peak-centered)
    N_t2t = len(Ts) - 1
    peak_stats = {}
    peak_stats['sample'] = Ps[1:]
    peak_stats['sample_zeroxD'] = zeroxD[1:]
    peak_stats['sample_zeroxR'] = zeroxR
    peak_stats['sample_lastE'] = Ts[:-1]
    peak_stats['sample_nextE'] = Ts[1:]
    peak_stats['period'] = peak_stats['sample_nextE'] - \
        peak_stats['sample_lastE']
    peak_stats['peak_volt'] = x[Ps[1:]]

    peak_stats['half_decay_time'] = (zeroxD[1:] - Ps[1:])
    peak_stats['half_rise_time'] = (Ps[1:] - zeroxR)
    peak_stats['peak_time'] = peak_stats['half_decay_time'] + \
        peak_stats['half_rise_time']

    peak_stats['half_decay_volt'] = x[Ps[1:]] - x[zeroxD[1:]]
    peak_stats['half_rise_volt'] = x[Ps[1:]] - x[zeroxR]

    peak_stats['half_decay_sharp'] = peak_stats['half_decay_volt'] / \
        peak_stats['half_decay_time']
    peak_stats['half_rise_sharp'] = peak_stats['half_rise_volt'] / \
        peak_stats['half_rise_time']
    peak_stats['peak_sharp'] = (peak_stats['half_decay_sharp'] +
                                peak_stats['half_rise_sharp']) / 2.

    peak_stats['whole_decay_time'] = (Ts[1:] - Ps[1:])
    peak_stats['whole_rise_time'] = (Ps[1:] - Ts[:-1])
    peak_stats['rdsym_time'] = peak_stats['whole_decay_time'] - \
        peak_stats['whole_rise_time']

    peak_stats['whole_decay_volt'] = x[Ps[1:]] - x[Ts[1:]]
    peak_stats['whole_rise_volt'] = x[Ps[1:]] - x[Ts[:-1]]
    peak_stats['rdsym_volt'] = peak_stats['whole_decay_volt'] - \
        peak_stats['whole_rise_volt']

    peak_stats['whole_decay_sharp'] = peak_stats['whole_decay_volt'] / \
        peak_stats['whole_decay_time']
    peak_stats['whole_rise_sharp'] = peak_stats['whole_rise_volt'] / \
        peak_stats['whole_rise_time']
    peak_stats['rdsym_sharp'] = peak_stats['whole_decay_sharp'] - \
        peak_stats['whole_rise_sharp']

    # Compute features of peak-trough symmetry
    # Peak relative to previous trough
    peak_stats['ptsym_volt'] = peak_stats['peak_volt'] + trough_stats['trough_volt']
    peak_stats['ptsym_time'] = peak_stats['peak_time'] - \
        trough_stats['trough_time']
    peak_stats['ptsym_sharp'] = peak_stats['peak_sharp'] - trough_stats['trough_sharp']
    trough_stats['ptsym_volt'] = peak_stats['ptsym_volt']
    trough_stats['ptsym_time'] = peak_stats['ptsym_time']
    trough_stats['ptsym_sharp'] = peak_stats['ptsym_sharp']

    # Compute amplitude features
    amp = ampT(x, f_range, Fs)
    trough_stats['amp_mean'] = [
        np.mean(amp[zeroxD[i]:zeroxR[i]]) for i in range(len(zeroxR))]
    peak_stats['amp_mean'] = [
        np.mean(amp[zeroxR[i]:zeroxD[i + 1]]) for i in range(len(zeroxR))]

    # Convert stats into a DataFrame
    df_P = pd.DataFrame.from_dict(peak_stats)
    df_T = pd.DataFrame.from_dict(trough_stats)

    # Define whether or not each cycle is part of an oscillation
    df_P, df_T = define_true_oscillating_periods(df_P, df_T, **define_true_oscillating_periods_kwargs)

    return df_P, df_T


def define_true_oscillating_periods(dfP, dfT, ampdiff_th=.7, timediff_th=.6):
    """
    Append two columns to cycle-by-cycle dataframes to label
    which periods are in true oscillatory modes

    Parameters
    ----------
    df_P : pandas DataFrame
        waveform features for each peak-centered cycle of a recording
    df_T : pandas DataFrame
        waveform features for each trough-centered cycle of a recording
    ampdiff_th : float between 0 and 1
        tolerance of amplitude

    Returns
    -------
    df_P : pandas DataFrame
        waveform features for each peak-centered cycle of a recording.
        Now including oscillatory mode features:
        oscillating_amp: are the amplitudes of the two flanks (rise and
            decay) within `ampdiff_th` fraction of one another?
        oscillating_amp_time: are the amplitudes of the two flanks within
            `ampdiff_th` fraction of one another? AND are the durations
            of the two flanks within `timediff_th` fraction of one another?
    df_T : pandas DataFrame
        waveform features for each trough-centered cycle of a recording.
        Now including oscillatory mode features
    """

    # Make a binary array to indicate if a peak is good to analyze
    P = len(dfP)
    cycle_good = np.zeros(P, dtype=bool)

    # Loop through each peak (skip first and last) and determine if meets
    # criteria for good
    rises = dfP['whole_rise_volt']
    decays = dfP['whole_decay_volt']
    for p in range(1, P - 1):
        frac1 = np.min([rises[p], decays[p]]) / np.max([rises[p], decays[p]])
        frac2 = np.min([rises[p], decays[p - 1]]) / \
            np.max([rises[p], decays[p - 1]])
        frac3 = np.min([rises[p + 1], decays[p]]) / \
            np.max([rises[p + 1], decays[p]])
        if np.min([frac1, frac2, frac3]) >= ampdiff_th:
            cycle_good[p] = True
    # Add oscillating feature to the dataframe
    dfP['oscillating_amp'] = cycle_good

    # Repeat process for troughs
    T = len(dfT)
    cycle_good = np.zeros(T, dtype=bool)
    rises = dfT['whole_rise_volt']
    decays = dfT['whole_decay_volt']
    for p in range(1, P - 1):
        frac1 = np.min([rises[p], decays[p]]) / np.max([rises[p], decays[p]])
        frac2 = np.min([rises[p - 1], decays[p]]) / \
            np.max([rises[p - 1], decays[p]])
        frac3 = np.min([rises[p], decays[p + 1]]) / \
            np.max([rises[p], decays[p + 1]])
        if np.min([frac1, frac2, frac3]) >= ampdiff_th:
            cycle_good[p] = True
    dfT['oscillating_amp'] = cycle_good

    # Make a binary array to indicate if a peak is good to analyze
    cycle_good_P = np.copy(dfP['oscillating_amp'])
    cycle_good_T = np.copy(dfT['oscillating_amp'])

    # Loop through each peak (skip first and last) and determine if meets
    # criteria for good
    P_times = dfP['sample'].values
    T_times = dfT['sample'].values

    for p in range(1, P - 1):
        if cycle_good_P[p]:
            p1 = P_times[p] - P_times[p - 1]
            p2 = P_times[p + 1] - P_times[p]
            frac = np.min([p1, p2]) / np.max([p1, p2])
            if frac < timediff_th:
                cycle_good_P[p] = False
                cycle_good_T[p] = False
                cycle_good_T[p + 1] = False

        if cycle_good_T[p]:
            p1 = T_times[p] - T_times[p - 1]
            p2 = T_times[p + 1] - T_times[p]
            frac = np.min([p1, p2]) / np.max([p1, p2])
            if frac < timediff_th:
                cycle_good_T[p] = False
                cycle_good_P[p] = False
                cycle_good_P[p - 1] = False

    dfP['oscillating_amp_time'] = cycle_good_P
    dfT['oscillating_amp_time'] = cycle_good_T

    return dfP, dfT
