import numpy as np
from mne.time_frequency import psd_array_multitaper
from meegkit.dss import dss_line
import matplotlib.pylab as plt
import copy

def nan_basic_interp(array):
    nans, ix = np.isnan(array), lambda x: x.nonzero()[0]
    array[nans] = np.interp(ix(nans), ix(~nans), array[~nans])
    return array

def zapline_until_gone(data, target_freq, sfreq, win_sz=10, spot_sz=2.5, max_iter=None, viz=False, prefix="zapline_iter"):
    """
    Returns: clean data, number of iterations

    Function iteratively applies the Zapline algorithm.

    data: assumed that the function is a part of the MNE-Python pipeline,
    the input should be an output of {MNE object}.get_data() function. The shape 
    should be Trials x Sensors x Time for epochs.
    target_freq: frequency + harmonics that comb-like approach will be applied
    with Zapline
    sfreq: sampling frequency, the output of {MNE object}.info["sfreq"]
    win_sz: 2x win_sz = window around the target frequency
    spot_sz: 2x spot_sz = width of the frequency peak to remove
    max_iter: maximum number of iterations before stopping (None for unlimited until convergence)
    viz: produce a visual output of each iteration,
    prefix: provide a path and first part of the file 
    "{prefix}_{iteration number}.png"
    """
    
    iterations = 0
    aggr_resid = []

    freq_rn = [target_freq - win_sz, target_freq + win_sz]
    freq_sp = [target_freq - spot_sz, target_freq + spot_sz]

    norm_vals = []
    resid_lims = []
    
    while True:
        if iterations > 0:
            data, art = dss_line(data.transpose(), target_freq, sfreq, nremove=1)
            del art
            data = data.transpose()
        
        psd, freq = psd_array_multitaper(data, sfreq, verbose=False)
        
        freq_rn_ix = [
            np.where(freq >= freq_rn[0])[0][0], 
            np.where(freq <= freq_rn[1])[0][-1]
        ]
        freq_used = freq[freq_rn_ix[0]:freq_rn_ix[1]]
        freq_sp_ix = [
            np.where(freq_used >= freq_sp[0])[0][0], 
            np.where(freq_used <= freq_sp[1])[0][-1]
        ]

        norm_psd = np.mean(psd, axis=0)[:, freq_rn_ix[0]:freq_rn_ix[1]]
        for ch_idx in range(norm_psd.shape[0]):
            if iterations == 0:
                norm_val = np.max(norm_psd[ch_idx, :])
                norm_vals.append(norm_val)
            else:
                norm_val = norm_vals[ch_idx]
            norm_psd[ch_idx, :] = norm_psd[ch_idx, :] / norm_val
        
        mean_psd = np.mean(norm_psd, axis=0)
        mean_psd_wospot = copy.copy(mean_psd)
        mean_psd_wospot[freq_sp_ix[0]: freq_sp_ix[1]] = np.nan
        mean_psd_tf = nan_basic_interp(mean_psd_wospot)
        
        pf = np.polyfit(freq_used, mean_psd_tf, 3)
        p = np.poly1d(pf)
        clean_fit_line = p(freq_used)
        residuals = mean_psd - clean_fit_line
        aggr_resid.append(np.mean(residuals))
        tf_ix = np.where(freq_used <= target_freq)[0][-1]
        print("Iteration:", iterations, "Power above the fit:", residuals[tf_ix])

        if viz:
            f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12,6), facecolor="gray", gridspec_kw={"wspace":0.2})
            for sensor in range(psd.shape[1]):
                ax1.plot(freq_used, norm_psd[sensor, :])
            ax1.set_title("Normalized mean PSD \nacross trials")

            ax2.plot(freq_used, mean_psd_tf, c="gray")
            ax2.plot(freq_used, mean_psd, c="blue")
            ax2.plot(freq_used, clean_fit_line, c="red")
            ax2.set_title("Mean PSD across \ntrials and sensors")

            ax3.set_title("Residuals")
            ax3.plot(residuals, freq_used)
            scat_color = "green" if residuals[tf_ix] > 0 else "red"
            ax3.scatter(residuals[tf_ix], freq_used[tf_ix], c=scat_color)
            
            if iterations == 0:
                resid_lims = ax3.get_xlim()
            else:
                ax3.set_xlim(resid_lims)

            ax4.set_title("Iterations")
            ax4.scatter(np.arange(iterations + 1), aggr_resid)
            plt.savefig("{}_{}.png".format(prefix, str(iterations).zfill(3)))
            plt.close("all")

        if (iterations > 0 and residuals[tf_ix] <= 0) or (max_iter is not None and iterations >= max_iter):
            break

        iterations += 1
    
    return [data, iterations]