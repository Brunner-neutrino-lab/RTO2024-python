"""
rto2024/controller.py

High-level controller for the R&S RTO2024 oscilloscope.

Two acquisition modes:
  "waveform"    — Fill history buffer, transfer all waveforms to PC,
                  run pulse finding on the PC (same as VX2740 path).
  "measure"     — Fill history buffer, step through each entry, read
                  the scope's built-in measurement result (peak amplitude,
                  area, etc.) per entry. Faster, no raw waveform storage.
                  Optionally also saves the raw waveforms alongside.

Both modes produce the same AcquisitionResult structure, so downstream
analysis is identical regardless of which instrument was used.

Usage (headless):

    from rto2024.controller import RTO2024Controller

    with RTO2024Controller("192.168.0.2", mode="simulation") as rto:
        rto.configure_channel(1, scale_v=0.05, offset_v=0.0)
        rto.configure_record_window(pre_us=2.0, post_us=10.0)
        rto.configure_trigger(source=1, level_v=0.010, slope="POS")
        rto.configure_measurement(function="MAX")

        # Measure mode: scope measures each entry, PC gets amplitude + timestamp
        result = rto.run(n_waveforms=1000, acquisition_mode="measure")

        # Waveform mode: full waveforms transferred, PC finds pulses
        result = rto.run(n_waveforms=1000, acquisition_mode="waveform")
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from .driver import (
    RTO2024Driver,
    DEFAULT_PRE_TIME_S, DEFAULT_POST_TIME_S,
    SIM_SPE_AMPLITUDE_V, SIM_DARK_RATE_HZ, SIM_NOISE_V,
    MEASUREMENT_FUNCTIONS,
)


# ---------------------------------------------------------------------------
# Data structures (mirrors VX2740 AcquisitionResult for DAQ compatibility)
# ---------------------------------------------------------------------------

@dataclass
class AcquisitionResult:
    """
    Result of one RTO2024 acquisition block.

    Compatible with vx2740.controller.AcquisitionResult.

    Pulse-level data (both acquisition modes):
        amplitudes[ch]   -> np.ndarray of pulse amplitudes (V)
        timestamps[ch]   -> np.ndarray of arrival times (s, relative to first waveform)

    Waveform-level data (waveform mode, or measure mode with save_waveforms=True):
        waveforms[ch]    -> np.ndarray shape (N, n_samples) float32, in volts
        time_axis        -> np.ndarray shape (n_samples,) float32, seconds

    Source info:
        source           -> "rto_measure" or "rto_waveform" or "rto_simulation"
        measurement_function -> e.g. "MAX", "AREA"
    """
    waveforms:           dict  = field(default_factory=dict)
    amplitudes:          dict  = field(default_factory=dict)
    timestamps:          dict  = field(default_factory=dict)
    time_axis:           object = None
    channel_ids:         list  = field(default_factory=list)
    n_waveforms:         int   = 0
    bias_voltage_V:      float = 0.0
    temperature_K:       float = 0.0
    run_timestamp:       float = field(default_factory=time.time)
    source:              str   = "rto_waveform"
    measurement_function: str  = "MAX"


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class RTO2024Controller:
    """
    High-level controller for the R&S RTO2024.

    Parameters
    ----------
    address : str
        IP address of the oscilloscope.
    mode : str
        "hardware" or "simulation".
    """

    # ------------------------------------------------------------------
    # Plugin interface
    # ------------------------------------------------------------------
    MODULE_NAME  = "RTO2024"
    DEVICE_NAME  = "R&S RTO2024 Oscilloscope"
    CONFIG_FIELDS = [
        {"key": "address",       "label": "IP Address",              "type": "str",    "default": "192.168.0.2"},
        {"key": "mode",          "label": "Mode",                    "type": "choice", "default": "simulation",
         "choices": ["simulation", "hardware"]},
        {"key": "channels",      "label": "Active channels (e.g. 1,2)", "type": "str", "default": "1"},
        {"key": "pre_us",        "label": "Pre-trigger (µs)",        "type": "float",  "default": 2.0},
        {"key": "post_us",       "label": "Post-trigger (µs)",       "type": "float",  "default": 10.0},
        {"key": "acq_mode",      "label": "Acquisition mode",        "type": "choice", "default": "measure",
         "choices": ["measure", "waveform"]},
        {"key": "meas_function", "label": "Measure function",        "type": "choice", "default": "MAX",
         "choices": list(MEASUREMENT_FUNCTIONS.keys())},
        {"key": "save_waveforms","label": "Save waveforms (measure mode)", "type": "bool", "default": False},
        {"key": "trigger_level_mv","label": "Trigger level (mV)",   "type": "float",  "default": 10.0},
        {"key": "trigger_slope", "label": "Trigger slope",           "type": "choice", "default": "POS",
         "choices": ["POS", "NEG"]},
    ]
    DEFAULTS = {
        "address":          "192.168.0.2",
        "mode":             "simulation",
        "channels":         "1",
        "pre_us":           2.0,
        "post_us":          10.0,
        "acq_mode":         "measure",
        "meas_function":    "MAX",
        "save_waveforms":   False,
        "trigger_level_mv": 10.0,
        "trigger_slope":    "POS",
    }

    @staticmethod
    def test(config: dict) -> tuple[bool, str]:
        try:
            ctrl = RTO2024Controller(
                address=config.get("address", "192.168.0.2"),
                mode=config.get("mode", "simulation"),
            )
            ctrl.connect()
            idn = ctrl.identify()
            ctrl.disconnect()
            return True, f"OK — {idn}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    @staticmethod
    def read(config: dict) -> dict:
        return {
            "address": config.get("address", ""),
            "mode":    config.get("mode", "simulation"),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, address: str = "192.168.0.2", mode: str = "simulation"):
        self._driver = RTO2024Driver(address=address, mode=mode)

        # Channel configuration
        self._channels:    list[int]        = [1]
        self._scale_v:     dict[int, float] = {}    # V/div per channel
        self._offset_v:    dict[int, float] = {}    # offset per channel

        # Record window
        self._pre_s   = DEFAULT_PRE_TIME_S
        self._post_s  = DEFAULT_POST_TIME_S

        # Trigger
        self._trigger_source: int   = 1
        self._trigger_level:  float = 0.010   # V
        self._trigger_slope:  str   = "POS"
        self._trigger_mode_ext: bool = False  # True = external trigger

        # Measurement
        self._meas_function:  str  = "MAX"
        self._meas_slot:      int  = 1        # scope measurement slot (1–8)

        # Acquisition
        self._acquisition_mode: str  = "measure"  # "measure" or "waveform"
        self._save_waveforms:   bool = False       # in measure mode: also save raw

        # Simulation tuning
        self._sim_dark_rates: dict[int, float] = {}
        self._sim_spe_amps:   dict[int, float] = {}

        # Pulse finding (waveform mode)
        self._pf_threshold_v: float = 0.010   # minimum peak above baseline (V)

        # Progress callback: fn(acquired, total)
        self.on_progress: Callable | None = None

    def connect(self):
        self._driver.connect()

    def disconnect(self):
        self._driver.disconnect()

    def identify(self) -> str:
        if self._driver.mode == "simulation":
            return f"R&S RTO2024 [simulation] @ {self._driver._address}"
        return f"R&S RTO2024 [hardware] @ {self._driver._address}"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_channel(self, channel: int, scale_v: float = 0.05,
                          offset_v: float = 0.0, coupling: str = "DC"):
        """
        Configure a scope channel.

        Parameters
        ----------
        channel : int
            Channel number (1–4).
        scale_v : float
            Vertical scale in V/div.
        offset_v : float
            Vertical offset in V.
        coupling : str
            "DC" or "AC".
        """
        self._scale_v[channel]  = scale_v
        self._offset_v[channel] = offset_v
        if channel not in self._channels:
            self._channels.append(channel)
            self._channels.sort()

        if self._driver.mode == "hardware":
            d = self._driver
            d.write(f"CHANnel{channel}:STATe ON")               # VERIFY
            d.write(f"CHANnel{channel}:SCALe {scale_v}")        # VERIFY: V/div
            d.write(f"CHANnel{channel}:OFFSet {offset_v}")      # VERIFY
            d.write(f"CHANnel{channel}:COUPling {coupling}")    # VERIFY

    def configure_channels(self, channels: list[int], scale_v: float = 0.05,
                           offset_v: float = 0.0):
        """Configure multiple channels with the same scale and offset."""
        for ch in channels:
            self.configure_channel(ch, scale_v, offset_v)

    def configure_record_window(self, pre_us: float = 2.0, post_us: float = 10.0):
        """
        Set the record window around the trigger.

        Parameters
        ----------
        pre_us : float
            Time before trigger in microseconds.
        post_us : float
            Time after trigger in microseconds.
        """
        self._pre_s  = pre_us  * 1e-6
        self._post_s = post_us * 1e-6
        total_s = self._pre_s + self._post_s

        if self._driver.mode == "hardware":
            d = self._driver
            # Time scale: total_s / 10 divisions
            d.write(f"TIMebase:SCALe {total_s / 10:.2e}")       # VERIFY: s/div
            # Reference point: pre_s / total_s * 100% from left
            ref_pct = (self._pre_s / total_s) * 100.0
            d.write(f"TIMebase:REFerence {ref_pct:.1f}PCT")     # VERIFY

        # Update driver geometry for time axis
        self._driver._x_start_s = -self._pre_s
        self._driver._x_stop_s  =  self._post_s

    def configure_trigger(self, source: int = 1, level_v: float = 0.010,
                          slope: str = "POS", mode: str = "self"):
        """
        Configure the scope trigger.

        Parameters
        ----------
        source : int
            Source channel for edge trigger (1–4).
        level_v : float
            Trigger threshold in volts.
        slope : str
            "POS" (rising edge) or "NEG" (falling edge).
        mode : str
            "self"     — self-trigger (edge trigger on signal)
            "external" — external trigger input
        """
        self._trigger_source   = source
        self._trigger_level    = level_v
        self._trigger_slope    = slope
        self._trigger_mode_ext = (mode == "external")

        if self._driver.mode == "hardware":
            d = self._driver
            if mode == "external":
                d.write("TRIGger:A:SOURce EXTernal")            # VERIFY
            else:
                d.write(f"TRIGger:A:SOURce CH{source}")         # VERIFY
            d.write(f"TRIGger:A:LEVel{source} {level_v:.4f}")  # VERIFY
            d.write(f"TRIGger:A:EDGE:SLOPe {slope}")           # VERIFY

    def configure_measurement(self, function: str = "MAX", channel: int | None = None):
        """
        Configure the scope's built-in measurement for measure mode.

        Parameters
        ----------
        function : str
            Measurement function. One of: MAX, MIN, PEAK, AMPL, MEAN, RMS, AREA.
            For SiPM positive pulses from CR200: use "MAX" (peak above baseline)
            or "AREA" (proportional to charge).
        channel : int, optional
            Source channel. Defaults to first configured channel.
        """
        if function not in MEASUREMENT_FUNCTIONS:
            raise ValueError(
                f"Unknown measurement function {function!r}. "
                f"Choose from: {list(MEASUREMENT_FUNCTIONS)}"
            )
        self._meas_function = function
        ch = channel if channel is not None else (self._channels[0] if self._channels else 1)
        self._driver.configure_measurement(self._meas_slot, ch, function)

    def configure_acquisition_mode(self, mode: str, save_waveforms: bool = False):
        """
        Set the acquisition mode.

        Parameters
        ----------
        mode : str
            "measure"  — scope measures each history entry; PC receives amplitude +
                         timestamp only. Fast, low bandwidth.
            "waveform" — full waveforms transferred to PC; PC runs pulse finding.
                         Same path as VX2740.
        save_waveforms : bool
            In "measure" mode: also transfer and save raw waveforms alongside the
            measurement results. Useful for cross-checking.
        """
        if mode not in ("measure", "waveform"):
            raise ValueError(f"mode must be 'measure' or 'waveform', got {mode!r}")
        self._acquisition_mode = mode
        self._save_waveforms   = save_waveforms

    def configure_pulse_finding(self, threshold_v: float = 0.010):
        """
        Set pulse finding threshold for waveform mode.

        Parameters
        ----------
        threshold_v : float
            Minimum peak amplitude above baseline (V) to count as a pulse.
        """
        self._pf_threshold_v = threshold_v

    # Simulation helpers
    def sim_set_dark_rate(self, channel: int, rate_hz: float):
        self._sim_dark_rates[channel] = rate_hz

    def sim_set_spe_amplitude(self, channel: int, amplitude_v: float):
        self._sim_spe_amps[channel] = amplitude_v

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def run(self, n_waveforms: int, timeout_s: float = 120.0) -> AcquisitionResult:
        """
        Acquire n_waveforms and return results.

        Automatically selects the correct acquisition path based on
        the configured acquisition_mode ("measure" or "waveform").
        Arms, acquires, and disarms the scope.

        Parameters
        ----------
        n_waveforms : int
            Number of waveforms / history segments to acquire.
        timeout_s : float
            Maximum time to wait for history buffer to fill.

        Returns
        -------
        AcquisitionResult
        """
        if self._driver.mode == "simulation":
            return self._run_simulation(n_waveforms)
        elif self._acquisition_mode == "measure":
            return self._run_measure_mode(n_waveforms, timeout_s)
        else:
            return self._run_waveform_mode(n_waveforms, timeout_s)

    # ------------------------------------------------------------------
    # Measure mode (scope measures each history entry)
    # ------------------------------------------------------------------

    def _run_measure_mode(self, n_waveforms: int,
                          timeout_s: float) -> AcquisitionResult:
        """
        Fill the history buffer, then step through each entry reading
        the scope's measurement result and optionally the raw waveform.
        """
        d = self._driver
        result = AcquisitionResult(
            channel_ids=list(self._channels),
            source="rto_measure",
            measurement_function=self._meas_function,
            run_timestamp=time.time(),
        )
        for ch in self._channels:
            result.amplitudes[ch]  = []
            result.timestamps[ch]  = []
            if self._save_waveforms:
                result.waveforms[ch] = []

        # Configure history and run
        d.configure_history(n_waveforms)
        d.run_until_full(timeout_s=timeout_s)
        n_acquired = d.get_segment_count()

        # Update time axis from scope header
        hdr = d.read_waveform_header(self._channels[0])
        result.time_axis = d.time_axis()

        # Step through history entries (oldest first)
        for i in range(n_acquired):
            # RTO: 0 = newest, -(n-1) = oldest → entry i (0-based oldest) = -(n-1-i)
            rto_idx = -(n_acquired - 1 - i)
            d.navigate_to_segment(rto_idx)

            ts = d.read_segment_timestamp()

            for ch in self._channels:
                amp = d.read_measurement(self._meas_slot)
                result.amplitudes[ch].append(amp)
                result.timestamps[ch].append(ts)
                if self._save_waveforms:
                    wfm = d.read_waveform(ch)
                    result.waveforms[ch].append(wfm)

            if self.on_progress:
                self.on_progress(i + 1, n_acquired)

        # Consolidate
        for ch in self._channels:
            result.amplitudes[ch] = np.array(result.amplitudes[ch], dtype=np.float32)
            result.timestamps[ch] = np.array(result.timestamps[ch], dtype=np.float64)
            if self._save_waveforms and result.waveforms[ch]:
                result.waveforms[ch] = np.stack(result.waveforms[ch])

        result.n_waveforms = n_acquired
        d.disable_history()
        return result

    # ------------------------------------------------------------------
    # Waveform mode (full waveforms transferred, PC finds pulses)
    # ------------------------------------------------------------------

    def _run_waveform_mode(self, n_waveforms: int,
                           timeout_s: float) -> AcquisitionResult:
        """
        Fill the history buffer, transfer all waveforms to PC,
        run pulse finding identically to the VX2740 path.
        """
        d = self._driver
        result = AcquisitionResult(
            channel_ids=list(self._channels),
            source="rto_waveform",
            measurement_function="PC",
            run_timestamp=time.time(),
        )
        for ch in self._channels:
            result.amplitudes[ch]  = []
            result.timestamps[ch]  = []
            result.waveforms[ch]   = []

        d.configure_history(n_waveforms)
        d.run_until_full(timeout_s=timeout_s)
        n_acquired = d.get_segment_count()

        hdr = d.read_waveform_header(self._channels[0])
        result.time_axis = d.time_axis()
        pre_samples = int(round(self._pre_s / hdr["dt_s"]))

        for i in range(n_acquired):
            rto_idx = -(n_acquired - 1 - i)
            d.navigate_to_segment(rto_idx)
            ts = d.read_segment_timestamp()

            for ch in self._channels:
                wfm = d.read_waveform(ch)
                result.waveforms[ch].append(wfm)

                # PC pulse finding
                amp, t_pulse = self._find_pulse(wfm, pre_samples, hdr["dt_s"], ts)
                if amp is not None:
                    result.amplitudes[ch].append(amp)
                    result.timestamps[ch].append(t_pulse)

            if self.on_progress:
                self.on_progress(i + 1, n_acquired)

        for ch in self._channels:
            result.amplitudes[ch] = np.array(result.amplitudes[ch], dtype=np.float32)
            result.timestamps[ch] = np.array(result.timestamps[ch], dtype=np.float64)
            if result.waveforms[ch]:
                result.waveforms[ch] = np.stack(result.waveforms[ch])

        result.n_waveforms = n_acquired
        d.disable_history()
        return result

    def _find_pulse(self, wfm: np.ndarray, pre_samples: int,
                    dt_s: float, wfm_timestamp: float) -> tuple:
        """
        Find the dominant pulse in a single waveform.

        Returns (amplitude_V, arrival_time_s) or (None, None) if no pulse found.
        """
        if len(wfm) == 0:
            return None, None
        baseline  = float(wfm[:pre_samples].mean())
        corrected = wfm - baseline
        peak_val  = float(corrected.max())
        if peak_val < self._pf_threshold_v:
            return None, None
        peak_idx = int(corrected.argmax())
        arrival  = wfm_timestamp + peak_idx * dt_s
        return peak_val, arrival

    # ------------------------------------------------------------------
    # Simulation path
    # ------------------------------------------------------------------

    def _run_simulation(self, n_waveforms: int) -> AcquisitionResult:
        """Generate synthetic waveforms and process them."""
        dark_rates = {ch: self._sim_dark_rates.get(ch, SIM_DARK_RATE_HZ)
                      for ch in self._channels}
        spe_amps   = {ch: self._sim_spe_amps.get(ch, SIM_SPE_AMPLITUDE_V)
                      for ch in self._channels}

        raw = self._driver.sim_generate_waveforms(
            n_waveforms, self._channels, dark_rates, spe_amps
        )

        result = AcquisitionResult(
            channel_ids=list(self._channels),
            source="rto_simulation",
            measurement_function=self._meas_function,
            run_timestamp=time.time(),
            n_waveforms=n_waveforms,
        )
        result.time_axis = self._driver.time_axis()
        pre_samples = int(round(self._pre_s / self._driver._dt_s))

        for ch in self._channels:
            waves = raw["waveforms"][ch]
            timestamps = raw["timestamps"]
            amps = []
            ts_out = []
            wfm_store = []

            for i, wfm in enumerate(waves):
                if self._acquisition_mode == "measure":
                    # Simulate scope measurement: peak of post-trigger window
                    baseline  = float(wfm[:pre_samples].mean())
                    corrected = wfm - baseline
                    amp = float(corrected[pre_samples:].max())
                    if amp >= self._pf_threshold_v:
                        amps.append(amp)
                        ts_out.append(float(timestamps[i]))
                    if self._save_waveforms:
                        wfm_store.append(wfm)
                else:
                    # Waveform mode: PC pulse finding
                    amp, t_p = self._find_pulse(wfm, pre_samples,
                                                self._driver._dt_s,
                                                float(timestamps[i]))
                    if amp is not None:
                        amps.append(amp)
                        ts_out.append(t_p)
                    wfm_store.append(wfm)

                if self.on_progress and (i % 100 == 0):
                    self.on_progress(i + 1, n_waveforms)

            result.amplitudes[ch] = np.array(amps, dtype=np.float32)
            result.timestamps[ch] = np.array(ts_out, dtype=np.float64)
            if wfm_store:
                result.waveforms[ch] = np.stack(wfm_store)

        if self.on_progress:
            self.on_progress(n_waveforms, n_waveforms)

        return result

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
