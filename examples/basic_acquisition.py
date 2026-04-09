"""
RTO2024 basic acquisition example — simulation mode.

Demonstrates both acquisition modes:
  - measure: scope-side amplitude extraction (fast)
  - waveform: full waveforms transferred, PC finds pulses
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rto2024 import RTO2024Controller

def run_mode(acq_mode, save_waveforms=False):
    print(f"\n--- Mode: {acq_mode} {'(+waveforms)' if save_waveforms else ''} ---")
    ctrl = RTO2024Controller(address="192.168.0.2", mode="simulation")

    # Realistic cold SiPM: low DCR, 20 mV SPE from CR200
    for ch in [1, 2]:
        ctrl.sim_set_dark_rate(ch, 300.0)
        ctrl.sim_set_spe_amplitude(ch, 0.020)

    ctrl.connect()
    ctrl.configure_channels([1, 2], scale_v=0.05)
    ctrl.configure_record_window(pre_us=2.0, post_us=10.0)
    ctrl.configure_trigger(source=1, level_v=0.010, slope="POS")
    ctrl.configure_measurement(function="MAX")
    ctrl.configure_acquisition_mode(acq_mode, save_waveforms=save_waveforms)
    ctrl.configure_pulse_finding(threshold_v=0.010)

    ctrl.on_progress = lambda a, t: print(f"\r  {a}/{t}", end="", flush=True)

    result = ctrl.run(n_waveforms=2000)
    print()

    for ch in result.channel_ids:
        amps = result.amplitudes[ch]
        if len(amps) == 0:
            print(f"  CH{ch}: 0 pulses found")
        else:
            print(f"  CH{ch}: {len(amps)} pulses | "
                  f"mean={amps.mean()*1000:.1f} mV | "
                  f"max={amps.max()*1000:.1f} mV")
        if ch in result.waveforms and len(result.waveforms[ch]) > 0:
            print(f"         waveforms stored: {result.waveforms[ch].shape}")

    print(f"  source={result.source}")
    ctrl.disconnect()


if __name__ == "__main__":
    run_mode("measure", save_waveforms=False)
    run_mode("measure", save_waveforms=True)
    run_mode("waveform")
    print("\nDone.")
