import tempfile
import logging
import numpy as np
import matplotlib.pyplot as plt
from pyOMA.core.PreProcessingTools import PreProcessSignals
from tests.system_ambient_ifrf import frequency_response, ambient_ifrf, ambient_spectral
logger = logging.getLogger('core.PreProcessingTools')
logger.setLevel(level=logging.DEBUG)


def verify_functionality():
    fname = '/vegas/users/staff/womo1998/git/pyOMA/tests/files/prepsignals.npz'
    prep_signals_compat = PreProcessSignals.load_state(fname)
    
    signals = prep_signals_compat.signals
    sampling_rate = prep_signals_compat.sampling_rate
    headers = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    
    prep_signals = PreProcessSignals(signals, sampling_rate, channel_headers=headers)
    
    print(prep_signals.total_time_steps, prep_signals.duration, prep_signals.dt, 
          prep_signals.t.shape, prep_signals.t, np.diff(prep_signals.t))
    
    print(prep_signals.total_time_steps, prep_signals.duration, prep_signals.dt, 
          prep_signals.t.shape, prep_signals.t, np.diff(prep_signals.t))
    
    print(prep_signals.signal_power)
    prep_signals.correct_offset()
    print(prep_signals.signal_rms)
    
    ref_channels = ['c', 'h', 'i']
    accel_channels = ['a', 'b', 'c', 'j']
    velo_channels = [3, 4, 5]
    disp_channels = ['g', 'h', 8]
    
    print(prep_signals.ref_channels, prep_signals.accel_channels)
    
    prep_signals.ref_channels = ref_channels
    prep_signals.disp_channels = disp_channels
    prep_signals.velo_channels = velo_channels
    prep_signals.accel_channels = accel_channels
    
    print(prep_signals.ref_channels, ref_channels)
    print(prep_signals.accel_channels, accel_channels)
    print(prep_signals.disp_channels, disp_channels)
    print(prep_signals.velo_channels, velo_channels)
    
    print(prep_signals.num_ref_channels, prep_signals.num_analised_channels)
    
    with tempfile.NamedTemporaryFile(suffix='.npz') as f:
        prep_signals.save_state(f.name)
        prep_signals_load = PreProcessSignals.load_state(f.name)
    
    print(prep_signals_load.total_time_steps, prep_signals_load.duration, prep_signals_load.dt, 
          prep_signals_load.t.shape, prep_signals_load.t, np.diff(prep_signals_load.t))
    
    print(prep_signals_load.signal_power)
    print(prep_signals_load.signal_rms)
    print(prep_signals_load.ref_channels, ref_channels)
    print(prep_signals_load.accel_channels, accel_channels)
    print(prep_signals_load.disp_channels, disp_channels)
    print(prep_signals_load.velo_channels, velo_channels)
    
    print(prep_signals_load.num_ref_channels, prep_signals_load.num_analised_channels)
    
    # Test basic plotting capabilities with selected channels
    example_channels = np.random.choice(accel_channels + velo_channels + disp_channels, 6)
    example_channel_numbers, _ = prep_signals._channel_numbers(example_channels)
    
    _, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    for channel, ax in zip(example_channels, axes.flat):
        prep_signals.plot_timeseries(channel, ax)
    plt.figure()
    prep_signals.plot_timeseries(**{'ls': 'dashed', 'alpha': 0.5})
        
    # Test decimation and plotting to same axes
    decimate_factor = 2
    nyq_rat = 10
    highpass = 0.1
    filter_type = 'brickwall'
    prep_signals.decimate_signals(decimate_factor, nyq_rat, highpass, None, filter_type)
    
    for channel, ax in zip(example_channel_numbers, axes.flat):
        prep_signals.plot_timeseries(channel, ax, alpha=0.5)
    
    n_lines = 3072
    method = 'welch'
    window = 'hanning'
    
    _, axesc = plt.subplots(2, 3, sharex=True, sharey=True)

    #pre compute correlations
    prep_signals.correlation(n_lines // 2 + 1, method, refs_only=True, window=window)
    for channel, ax in zip(example_channels, axesc.flat):
        # should not trigger recompuation of correlations, only ref_channels are used
        prep_signals.plot_correlation(None, channel, ax, method=method)
    plt.figure()
    # should trigger recomputation of correlations, due to 'auto' all channels have to be used
    _ = prep_signals.plot_correlation(None, example_channels, refs='auto', window=window, method=method)
        
    # test psd plotting in different scales
    _, axesf = plt.subplots(2, 3, sharex=True, sharey=False)

    #pre compute psd
    prep_signals.psd(n_lines, method, refs_only=False, window=window)
    for channel, ax, scale in zip(example_channels, axesf.flat, ['db', 'power', 'rms', 'phase', 'db', 'phase']):
        # should not trigger recompuation of psds, only ref_channels are used
        prep_signals.plot_psd(None, channel, ax, scale)
    plt.figure()
    # should not trigger recomputation of psds,only ref_channels are used
    # multiple channels by index list and not named channels
    axf = prep_signals.plot_psd(None, example_channel_numbers, plot_kwarg_dict={'marker': 'x', 'alpha': 0.5})
    
    n_lines = 3072
    method = 'blackman-tukey'
    window = 'ones'
    #pre compute correlations
    prep_signals.correlation(n_lines // 2 + 1, method, refs_only=True)
    for channel, ax in zip(example_channels, axesc.flat):
        # should not trigger recompuation of correlations, only ref_channels are used
        prep_signals.plot_correlation(None, channel, ax, method=method)
    #pre compute psd
    prep_signals.psd(n_lines, method, refs_only=True)
    for channel, ax, scale in zip(example_channels, axesf.flat, ['db', 'power', 'rms', 'phase', 'db', 'phase']):
        # should not trigger recompuation of psds, only ref_channels are used
        prep_signals.plot_psd(None, channel, ax, scale, plot_kwarg_dict={'alpha': 0.5})
    # should not trigger recomputation of psds,only ref_channels are used
    # multiple channels by index list and not named channels
    prep_signals.plot_psd(None, example_channel_numbers, scale='svd', ax=axf)
    
    prep_signals.plot_signals(None, True)
    prep_signals.plot_signals(None, True, psd_scale='svd')
    prep_signals.plot_signals(example_channels, False, timescale='lags', psd_scale='svd', plot_kwarg_dict={'alpha': 0.5})
    plt.show()
    
    
def shift_plot(freq, spec):
    return np.fft.fftshift(freq), np.fft.fftshift(spec)


def validate_spectral():
    '''
    Generates the ambient acceleration response of a fixed-free rod and
    theoretical solutions for the PSD and correlation functions.
    
    Compares PSDs and correlation function visually and by estimated vs.
    theoretically signal power values.
    
    Theoretical and estimated solutions should match approximately, regardless
    of the input parameters defined below and prior to the method calls
    
    in particular the following settings should match well
    
    N = 16384
    n_nodes = 10
    inp_nodes = [9]
    fs = 140
    f_scale = 10
    n_modes = 1
    n_lines = N // 4
    method = 'blackman-tukey'
    window = 'bartlett'
    num_blocks = 1
    '''
    
    def print_powers():
        
        plt.figure()
        psd_est = prep_signals.psd_matrix
        corr_est = prep_signals.corr_matrix
        n_comb = n_nodes*len(inp_nodes)
        powers = np.zeros((n_comb, 5))
        i=0
        xticklabels=[]
        for node in range(n_nodes):
            for ref_ind, ref_node in enumerate(inp_nodes):
                print(f'\n Node: {node}, Ref. Node: {ref_node}')
                power = np.sum(np.abs(sig[:, node] * sig[:, ref_node]))
                powers[i,0]=power
                print(f'Signal power time-domain: {power}')
                power_psd = np.mean(np.abs(psd_ana[node, ref_ind, :]))
                powers[i,1]=power_psd
                assert np.isclose(power_psd.imag, 0)
                print(f'Theoretical PSD power: {power_psd.real}')
                power_corr = np.abs(corr_ana[node, ref_ind, 0] * N)
                powers[i,3]=power_corr
                print(f'Theoretical 0-lag correlation (power): {power_corr}')
                power_psd = np.mean(np.abs(psd_est[node, ref_ind, :]))
                powers[i,2]=power_psd
                assert np.isclose(power_psd.imag, 0)
                print(f'Estimated PSD power: {power_psd.real}')
                if prep_signals._last_meth == 'welch':
                    norm_fact = prep_signals.n_lines
                elif prep_signals._last_meth == 'blackman-tukey':
                    norm_fact = prep_signals.total_time_steps
                power_corr = np.abs(corr_est[node, ref_ind, 0] * norm_fact)
                powers[i,4]=power_corr
                print(f'Estimated 0-lag correlation (power): {power_corr}')
                i+=1
                xticklabels.append(f'{node}-{ref_node}')
        
        x= np.arange(n_comb)
        for i, label in enumerate(['psig', 'psdana', 'psdest', 'corrana',  'correst']):
            if i==0:
                plt.bar(x, powers[:,i], label=label, width=2/3, color='none', edgecolor='grey', ls='dashed')
            else:
                plt.bar(x+5/12-(i/6), powers[:,i], label=label, width=1/6)
        plt.legend()
        plt.gca().set_xticks(x)
        plt.gca().set_xticklabels(xticklabels)
        # plt.show()
            
    N = 16384
    n_nodes = 12
    # too many input nodes make the plots hard to read
    inp_nodes = [9]
    fs = 140
    f_scale = 10
    n_modes = 1
    
    # print theoretical FRFs
    if False:
        omegas, frf = frequency_response(N, n_nodes, inp_nodes[0], fs)
        plt.figure()
        plt.plot(omegas, np.abs(frf))
        plt.show()
    
    time_values, sig = ambient_ifrf(N, n_nodes, inp_nodes, fs, f_scale, num_modes=n_modes, snr_db=np.infty)
    
    # print synthesized ambient vibration response data
    if False:
        _, axes = plt.subplots(3, 3)
        for i in range(9):
            axes.flat[i].plot(time_values, sig[:, i])
        plt.show()
    
    # theoretical solutions
    omegas, taus, psd_ana, corr_ana = ambient_spectral(N, n_nodes, inp_nodes, fs, f_scale, num_modes=n_modes)
    
    # initialize PreProcessSignals object
    prep_signals = PreProcessSignals(sig, fs, ref_channels=inp_nodes)
    
    # Estimation parameters (output should approximately match theoretical
    # solution regardless of these parameters )
    n_lines = N // 4
    method = 'blackman-tukey'
    window = 'bartlett'
    num_blocks = 1
    
    axest, axesf = prep_signals.plot_signals(per_channel_axes=True, n_lines=n_lines, timescale='lags', psd_scale='power',
                                             method=method, window=window, plot_kwarg_dict={'alpha': 0.5, }, num_blocks=num_blocks)
    
    print_powers()
    
    # Estimation parameters (output should approximately match theoretical
    # solution regardless of these parameters )
    n_lines = N // 4
    method = 'welch'
    window = 'hamming'
    
    axest, axesf = prep_signals.plot_signals(per_channel_axes=True, n_lines=n_lines, timescale='lags', psd_scale='power',
                                             axest=axest, axesf=axesf,
                                             method=method, window=window, plot_kwarg_dict={'alpha': 0.5, })
    print_powers()
    
    # plot theoretical solution
    for i in range(n_nodes):
        for j in range(len(inp_nodes)):
            axest[i].plot(taus, corr_ana[i, j, :] * N, alpha=0.5, marker='+')
            axesf[i].plot(*shift_plot(omegas / 2 / np.pi, np.abs(psd_ana[i, j, :])), alpha=0.5, marker='x')
    plt.show()


if __name__ == '__main__':
    # verify_functionality()
    validate_spectral()
    