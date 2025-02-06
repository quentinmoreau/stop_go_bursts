import copy
import math
import numpy as np
from mne.filter import filter_data
from scipy.signal import hilbert, argrelextrema
from scipy.stats import linregress


def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    """
    Two-dimensional gaussian function
    :param x: x grid
    :param y: y grid
    :param mx: mean in x dimension
    :param my: mean in y dimension
    :param sx: standard deviation in x dimension
    :param sy: standard deviation in y dimension
    :return: Two-dimensional Gaussian distribution
    """
    return np.exp(-((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))


def overlap(a, b):
    """
    Find if two ranges overlap
    :param a: first range [low, high]
    :param b: second range [low, high]
    :return: True if ranges overlap, false otherwise
    """
    return a[0] <= b[0] <= a[1] or b[0] <= a[0] <= b[1]


def fwhm_burst_norm(tf, peak):
    """
    Find two-dimensional FWHM
    :param tf: TF spectrum
    :param peak: peak of activity [freq, time]
    :return: right, left, up, down limits for FWM
    """
    right_loc = np.nan
    # Find right limit (values to right of peak less than half value at peak)
    cand = np.where(tf[peak[0], peak[1]:] <= tf[peak] / 2)[0]
    # If any found, take the first one
    if len(cand):
        right_loc = cand[0]

    up_loc = np.nan
    # Find up limit (values above peak less than half value at peak)
    cand = np.where(tf[peak[0]:, peak[1]] <= tf[peak] / 2)[0]
    # If any found, take the first one
    if len(cand):
        up_loc = cand[0]

    left_loc = np.nan
    # Find left limit (values below peak less than half value at peak)
    cand = np.where(tf[peak[0], :peak[1]] <= tf[peak] / 2)[0]
    # If any found, take the last one
    if len(cand):
        left_loc = peak[1] - cand[-1]

    down_loc = np.nan
    # Find down limit (values below peak less than half value at peak)
    cand = np.where(tf[:peak[0], peak[1]] <= tf[peak] / 2)[0]
    # If any found, take the last one
    if len(cand):
        down_loc = peak[0] - cand[-1]

    # Set arms equal if only one found
    if down_loc is np.nan:
        down_loc = up_loc
    if up_loc is np.nan:
        up_loc = down_loc
    if left_loc is np.nan:
        left_loc = right_loc
    if right_loc is np.nan:
        right_loc = left_loc

    # Use the minimum arm in each direction (forces Gaussian to be symmetric in each dimension)
    horiz = np.nanmin([left_loc, right_loc])
    vert = np.nanmin([up_loc, down_loc])
    right_loc = horiz
    left_loc = horiz
    up_loc = vert
    down_loc = vert
    return right_loc, left_loc, up_loc, down_loc


def extract_bursts_single_trial(raw_trial, tf, times, search_freqs, band_lims, aperiodic_spectrum, sfreq, w_size=.26):
    """
    Extract bursts from epoched data
    :param raw_trial: raw data for trial (time)
    :param tf: time-frequency decomposition for trial (freq x time)
    :param times: time steps
    :param search_freqs: frequency limits to search within for bursts (should be wider than band_lims)
    :param band_lims: keep bursts whose peak frequency falls within these limits
    :param aperiodic_spectrum: aperiodic spectrum
    :param sfreq: sampling rate
    :param w_size: window size to extract burst waveforms
    :return: disctionary with waveform, peak frequency, relative peak amplitude, absolute peak amplitude, peak
            time, peak adjustment, FWHM in frequency, FWHM in time, and polarity for each detected burst
    """
    bursts = {
        'waveform': [],
        'peak_freq': [],
        'peak_amp_iter': [],
        'peak_amp_base': [],
        'peak_time': [],
        'peak_adjustment': [],
        'fwhm_freq': [],
        'fwhm_time': [],
        'polarity': [],
        'waveform_times': []
    }

    # Grid for computing 2D Gaussians
    x_idx, y_idx = np.meshgrid(range(len(times)), range(len(search_freqs)))

    # Window size in points
    wlen = int(w_size * sfreq)
    half_wlen = int(wlen * .5)

    # Subtract 1/f
    trial_tf = tf - aperiodic_spectrum
    trial_tf[trial_tf < 0] = 0

    # Skip trial if no peaks above aperiodic
    if (trial_tf == 0).all():
        print("All values equal 0 after aperiodic subtraction")
        return bursts

    # TF for iterating
    trial_tf_iter = copy.copy(trial_tf)

    while True:
        # Compute noise floor
        thresh = 2 * np.std(trial_tf_iter)

        # Find peak
        [peak_freq_idx, peak_time_idx] = np.unravel_index(np.argmax(trial_tf_iter), trial_tf.shape)
        peak_freq = search_freqs[peak_freq_idx]
        peak_amp_iter = trial_tf_iter[peak_freq_idx, peak_time_idx]
        peak_amp_base = trial_tf[peak_freq_idx, peak_time_idx]
        # Stop if no peak above threshold
        if peak_amp_iter < thresh:
            break

        # Fit 2D Gaussian and subtract from TF
        rloc, lloc, uloc, dloc = fwhm_burst_norm(trial_tf_iter, (peak_freq_idx, peak_time_idx))

        # Detect degenerate Gaussian (limits not found)
        vert_isnan = any(np.isnan([uloc, dloc]))
        horiz_isnan = any(np.isnan([rloc, lloc]))
        if vert_isnan:
            v_sh = int((search_freqs.shape[0] - peak_freq_idx) / 2)
            if v_sh <= 0:
                v_sh = 1
            uloc = v_sh
            dloc = v_sh
        elif horiz_isnan:
            h_sh = int((times.shape[0] - peak_time_idx) / 2)
            if h_sh <= 0:
                h_sh = 1
            rloc = h_sh
            lloc = h_sh
        hv_isnan = any([vert_isnan, horiz_isnan])

        # Compute FWHM and convert to SD
        fwhm_f_idx = uloc + dloc
        fwhm_f = (search_freqs[1] - search_freqs[0]) * fwhm_f_idx
        fwhm_t_idx = lloc + rloc
        fwhm_t = (times[1] - times[0]) * fwhm_t_idx
        sigma_t = fwhm_t_idx / 2.355
        sigma_f = fwhm_f_idx / 2.355
        # Fitted Gaussian
        z = peak_amp_iter * gaus2d(x_idx, y_idx, mx=peak_time_idx, my=peak_freq_idx, sx=sigma_t, sy=sigma_f)
        # Subtract fitted Gaussian for next iteration
        new_trial_tf_iter = trial_tf_iter - z

        # If detected peak is within band limits and not degenerate
        if all([peak_freq >= band_lims[0], peak_freq <= band_lims[1], not hv_isnan]):
            # Bandpass filter within frequency range of burst
            freq_range = [
                np.max([0, peak_freq_idx - dloc]),
                np.min([len(search_freqs) - 1, peak_freq_idx + uloc])
            ]
            filtered = filter_data(raw_trial.reshape(1, -1), sfreq, search_freqs[freq_range[0]], search_freqs[freq_range[1]],
                                   verbose=False)

            # Hilbert transform
            analytic_signal = hilbert(filtered)
            # Get phase
            instantaneous_phase = np.unwrap(np.angle(analytic_signal)) % math.pi

            # Find local phase minima with negative deflection closest to TF peak
            # If no minimum is found, the error is caught and no burst is added
            min_phase_pts = argrelextrema(instantaneous_phase.T, np.less)[0]
            new_peak_time_idx = peak_time_idx
            try:
                new_peak_time_idx = min_phase_pts[np.argmin(np.abs(peak_time_idx - min_phase_pts))]
                adjustment = (new_peak_time_idx - peak_time_idx) * 1 / sfreq
            except:
                adjustment = 1

            # Keep if adjustment less than 30ms
            if np.abs(adjustment) < .03:

                # If burst won't be cutoff
                if new_peak_time_idx >= half_wlen and new_peak_time_idx + half_wlen <= len(times):
                    peak_time = times[new_peak_time_idx]

                    overlapped = False
                    # Check for overlap
                    for b_idx in range(len(bursts['peak_time'])):
                        o_t = bursts['peak_time'][b_idx]
                        o_fwhm_t = bursts['fwhm_time'][b_idx]
                        if overlap([peak_time - .5 * fwhm_t, peak_time + .5 * fwhm_t],
                                   [o_t - .5 * o_fwhm_t, o_t + .5 * o_fwhm_t]):
                            overlapped = True
                            break

                    if not overlapped:
                        # Get burst
                        burst = raw_trial[new_peak_time_idx - half_wlen:new_peak_time_idx + half_wlen]
                        # Remove DC offset
                        burst = burst - np.mean(burst)
                        bursts['waveform_times'] = times[new_peak_time_idx - half_wlen:new_peak_time_idx + half_wlen] - \
                                                   times[new_peak_time_idx]

                        # Flip if positive deflection
                        peak_dists = np.abs(argrelextrema(filtered.T, np.greater)[0] - new_peak_time_idx)
                        trough_dists = np.abs(argrelextrema(filtered.T, np.less)[0] - new_peak_time_idx)

                        polarity = 0
                        if len(trough_dists) == 0 or (
                                len(peak_dists) > 0 and np.min(peak_dists) < np.min(trough_dists)):
                            burst *= -1.0
                            polarity = 1

                        bursts['waveform'].append(burst)
                        bursts['peak_freq'].append(peak_freq)
                        bursts['peak_amp_iter'].append(peak_amp_iter)
                        bursts['peak_amp_base'].append(peak_amp_base)
                        bursts['peak_time'].append(peak_time)
                        bursts['peak_adjustment'].append(adjustment)
                        bursts['fwhm_freq'].append(fwhm_f)
                        bursts['fwhm_time'].append(fwhm_t)
                        bursts['polarity'].append(polarity)

        trial_tf_iter = new_trial_tf_iter

    bursts['waveform'] = np.array(bursts['waveform'])
    bursts['peak_freq'] = np.array(bursts['peak_freq'])
    bursts['peak_amp_iter'] = np.array(bursts['peak_amp_iter'])
    bursts['peak_amp_base'] = np.array(bursts['peak_amp_base'])
    bursts['peak_time'] = np.array(bursts['peak_time'])
    bursts['peak_adjustment'] = np.array(bursts['peak_adjustment'])
    bursts['fwhm_freq'] = np.array(bursts['fwhm_freq'])
    bursts['fwhm_time'] = np.array(bursts['fwhm_time'])
    bursts['polarity'] = np.array(bursts['polarity'])

    return bursts


def extract_bursts(raw_trials, tf, times, search_freqs, band_lims, aperiodic_spectrum, sfreq, w_size=.26):
    """
    Extract bursts from epoched data
    :param raw_trials: raw data for each trial (trial x time)
    :param tf: time-frequency decomposition for each trial (trial x freq x time)
    :param times: time steps
    :param search_freqs: frequency limits to search within for bursts (should be wider than band_lims)
    :param band_lims: keep bursts whose peak frequency falls within these limits
    :param aperiodic_spectrum: aperiodic spectrum
    :param sfreq: sampling rate
    :param w_size: window size to extract burst waveforms
    :return: disctionary with trial, waveform, peak frequency, relative peak amplitude, absolute peak amplitude, peak
            time, peak adjustment, FWHM in frequency, FWHM in time, and polarity for each detected burst
    """
    bursts = {
        'trial': [],
        'waveform': [],
        'peak_freq': [],
        'peak_amp_iter': [],
        'peak_amp_base': [],
        'peak_time': [],
        'peak_adjustment': [],
        'fwhm_freq': [],
        'fwhm_time': [],
        'polarity': [],
        'waveform_times': []
    }

    # Compute event-related signal
    erf = np.mean(raw_trials, axis=0)

    # Iterate through trials
    for t_idx, tr_tf in enumerate(tf):

        # Regress out ERF
        slope, intercept, r, p, se = linregress(erf, raw_trials[t_idx, :])
        raw_trial = raw_trials[t_idx, :] - (intercept + slope * erf)

        trial_bursts=extract_bursts_single_trial(raw_trial, tr_tf, times, search_freqs, band_lims, aperiodic_spectrum,
                                                 sfreq, w_size=w_size)

        n_trial_bursts=len(trial_bursts['peak_time'])
        bursts['trial'].extend([int(t_idx) for i in range(n_trial_bursts)])
        bursts['waveform'].extend(trial_bursts['waveform'])
        bursts['peak_freq'].extend(trial_bursts['peak_freq'])
        bursts['peak_amp_iter'].extend(trial_bursts['peak_amp_iter'])
        bursts['peak_amp_base'].extend(trial_bursts['peak_amp_base'])
        bursts['peak_time'].extend(trial_bursts['peak_time'])
        bursts['peak_adjustment'].extend(trial_bursts['peak_adjustment'])
        bursts['fwhm_freq'].extend(trial_bursts['fwhm_freq'])
        bursts['fwhm_time'].extend(trial_bursts['fwhm_time'])
        bursts['polarity'].extend(trial_bursts['polarity'])
        if len(trial_bursts['waveform_times']):
            bursts['waveform_times'] = trial_bursts['waveform_times']

    bursts['trial'] = np.array(bursts['trial'])
    bursts['waveform'] = np.array(bursts['waveform'])
    bursts['peak_freq'] = np.array(bursts['peak_freq'])
    bursts['peak_amp_iter'] = np.array(bursts['peak_amp_iter'])
    bursts['peak_amp_base'] = np.array(bursts['peak_amp_base'])
    bursts['peak_time'] = np.array(bursts['peak_time'])
    bursts['peak_adjustment'] = np.array(bursts['peak_adjustment'])
    bursts['fwhm_freq'] = np.array(bursts['fwhm_freq'])
    bursts['fwhm_time'] = np.array(bursts['fwhm_time'])
    bursts['polarity'] = np.array(bursts['polarity'])

    return bursts
