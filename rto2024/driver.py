"""
rto2024/driver.py

Low-level interface to the R&S RTO2024 oscilloscope.

Handles:
  - VISA Ethernet (LXI) connection
  - SCPI command / query
  - Binary waveform block decode (REAL,32, little-endian, already in volts)
  - History (fast segmentation) buffer management
  - Measurement function readout from history entries

Two modes:
  "hardware"   — connects to real scope via pyvisa
  "simulation" — generates synthetic CR200-shaped SiPM waveforms in volts

Waveform data is always float32, in volts, matching the scope's own export
format (verified against RTO-wfmparser/scopedata reference files).
"""

import struct
import time
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default record window (matches VX2740 defaults for consistency)
DEFAULT_SAMPLE_RATE_HZ   = 1e9          # 1 GS/s (RTO2024 max is 10 GS/s; 1G typical)
DEFAULT_PRE_TIME_S       = 2e-6         # 2 µs
DEFAULT_POST_TIME_S      = 10e-6        # 10 µs
DEFAULT_RECORD_TIME_S    = DEFAULT_PRE_TIME_S + DEFAULT_POST_TIME_S

# Simulation pulse parameters (CR112 CSP + CR200-1µs shaper → Gaussian ~1µs FWHM)
SIM_NOISE_V              = 0.0005       # 0.5 mV RMS baseline noise
SIM_SPE_AMPLITUDE_V      = 0.020        # 20 mV per SPE (typical after CR200)
SIM_PULSE_SIGMA_S        = 4e-7         # σ ≈ 0.4 µs → FWHM ≈ 1 µs
SIM_DARK_RATE_HZ         = 500.0

# Supported measurement functions (SCPI MEASurement:MAIN values)
MEASUREMENT_FUNCTIONS = {
    "MAX":   "MAX",      # Maximum value in waveform window → pulse peak for positive pulses
    "MIN":   "MIN",      # Minimum value
    "PEAK":  "PEAK",     # Peak-to-peak amplitude
    "AMPL":  "AMPL",     # Amplitude (high level - low level)
    "MEAN":  "MEAN",     # Mean value
    "RMS":   "RMS",      # RMS value
    "AREA":  "AREA",     # Area (integral) — proportional to charge
}


class RTO2024Driver:
    """
    Low-level SCPI driver for the R&S RTO2024.

    Parameters
    ----------
    address : str
        IP address of the oscilloscope, e.g. "192.168.0.2"
    mode : str
        "hardware" or "simulation"
    """

    def __init__(self, address: str = "192.168.0.2", mode: str = "simulation"):
        if mode not in ("hardware", "simulation"):
            raise ValueError(f"mode must be 'hardware' or 'simulation', got {mode!r}")

        self._address  = address
        self._mode     = mode
        self._visa     = None       # pyvisa resource
        self._rm       = None       # pyvisa ResourceManager
        self._connected = False

        # Current waveform geometry (populated by configure / header query)
        self._x_start_s  = -DEFAULT_PRE_TIME_S
        self._x_stop_s   = DEFAULT_POST_TIME_S
        self._n_samples  = int(DEFAULT_RECORD_TIME_S * DEFAULT_SAMPLE_RATE_HZ)
        self._dt_s       = 1.0 / DEFAULT_SAMPLE_RATE_HZ

        # Simulation state
        self._sim_channels:     list[int]       = [1]
        self._sim_dark_rates:   dict[int, float] = {}
        self._sim_spe_amps:     dict[int, float] = {}
        self._sim_rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        if self._connected:
            return
        if self._mode == "hardware":
            self._connect_hardware()
        else:
            self._connect_simulation()
        self._connected = True

    def disconnect(self):
        if not self._connected:
            return
        if self._mode == "hardware":
            self._disconnect_hardware()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # SCPI primitives (hardware mode)
    # ------------------------------------------------------------------

    def write(self, cmd: str):
        """Send a SCPI command."""
        if self._mode != "hardware":
            return
        self._visa.write(cmd)

    def query(self, cmd: str) -> str:
        """Send a SCPI query and return the response string."""
        if self._mode != "hardware":
            return ""
        return self._visa.query(cmd).strip()

    def query_binary(self, cmd: str) -> bytes:
        """
        Send a SCPI query and return the raw binary response bytes.
        Used for waveform data in REAL,32 format.
        """
        if self._mode != "hardware":
            return b""
        return self._visa.query_binary_values(cmd, datatype='f', is_big_endian=False,
                                              container=np.ndarray)

    def opc_wait(self):
        """Block until all pending operations complete (*OPC?)."""
        if self._mode == "hardware":
            self._visa.query("*OPC?")

    # ------------------------------------------------------------------
    # Waveform geometry
    # ------------------------------------------------------------------

    def read_waveform_header(self, channel: int) -> dict:
        """
        Query waveform scaling parameters from the scope.

        Returns dict with keys:
            x_start_s, x_stop_s, n_samples, dt_s
        """
        if self._mode == "simulation":
            return {
                "x_start_s": self._x_start_s,
                "x_stop_s":  self._x_stop_s,
                "n_samples": self._n_samples,
                "dt_s":      self._dt_s,
            }

        # CHANnel<N>:DATA:HEADer? returns: <XStart>,<XStop>,<RecordLength>,<ValuesPerSample>
        raw = self.query(f"CHANnel{channel}:DATA:HEADer?")  # VERIFY syntax
        parts = raw.split(",")
        x_start  = float(parts[0])
        x_stop   = float(parts[1])
        n_points = int(parts[2])
        dt       = (x_stop - x_start) / max(n_points - 1, 1)

        self._x_start_s = x_start
        self._x_stop_s  = x_stop
        self._n_samples = n_points
        self._dt_s      = dt

        return {
            "x_start_s": x_start,
            "x_stop_s":  x_stop,
            "n_samples": n_points,
            "dt_s":      dt,
        }

    def time_axis(self) -> np.ndarray:
        """Return time axis array in seconds."""
        return np.linspace(self._x_start_s, self._x_stop_s, self._n_samples,
                           dtype=np.float32)

    # ------------------------------------------------------------------
    # History (fast segmentation) control
    # ------------------------------------------------------------------

    def configure_history(self, n_segments: int):
        """
        Enable fast segmentation and set the number of segments.

        Parameters
        ----------
        n_segments : int
            Number of waveforms to acquire per history fill.
            Maximum is scope-dependent (typically up to 50,000).
        """
        if self._mode == "hardware":
            self.write(f"ACQuire:SEGMented:MAX {n_segments}")   # VERIFY
            self.write("ACQuire:SEGMented:STATe ON")            # VERIFY
            self.write("ACQuire:HISTory:STATe ON")              # VERIFY

    def disable_history(self):
        """Disable fast segmentation / history mode."""
        if self._mode == "hardware":
            self.write("ACQuire:SEGMented:STATe OFF")           # VERIFY
            self.write("ACQuire:HISTory:STATe OFF")             # VERIFY

    def run_until_full(self, timeout_s: float = 120.0):
        """
        Start acquisition and wait until the history buffer is full.

        The scope stops automatically when the configured number of
        segments has been acquired (SEGMented:STATe ON behaviour).
        """
        if self._mode == "hardware":
            self.write("RUN")                                   # VERIFY: RUN or SINGle
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                # Query acquisition state
                state = self.query("ACQuire:STATe?")            # VERIFY
                if state.upper() in ("STOP", "0"):
                    return
                time.sleep(0.1)
            self.write("STOP")
            raise TimeoutError(
                f"History buffer not full after {timeout_s:.0f}s"
            )

    def get_segment_count(self) -> int:
        """Return number of segments currently in the history buffer."""
        if self._mode == "hardware":
            return int(self.query("ACQuire:SEGMented:COUNt?"))  # VERIFY
        return 0

    def navigate_to_segment(self, index: int):
        """
        Navigate history to a specific segment index.

        The RTO uses 0 = most recent, negative = older (0, -1, -2, ...).
        Here we use 0-based positive indexing from oldest; conversion is done internally.

        Parameters
        ----------
        index : int
            0-based index from oldest entry (0 = first acquired).
        n_total : int
            Total number of segments in buffer (for index conversion).
        """
        if self._mode == "hardware":
            # RTO convention: 0 = newest, -(n-1) = oldest
            # We store oldest-first so convert: rto_idx = -(n_total - 1 - index)
            # Caller must pass n_total; stored in controller, passed via read_waveform
            # Here we accept the raw RTO-style index (caller converts)
            self.write(f"ACQuire:HISTory:CURRent {index}")      # VERIFY

    def read_segment_timestamp(self) -> float:
        """
        Read the relative timestamp of the currently selected history segment.

        Returns time in seconds relative to the first segment.
        """
        if self._mode == "hardware":
            raw = self.query("ACQuire:HISTory:TSRelative?")     # VERIFY
            return float(raw)
        return 0.0

    # ------------------------------------------------------------------
    # Waveform data readout
    # ------------------------------------------------------------------

    def read_waveform(self, channel: int) -> np.ndarray:
        """
        Read waveform from the currently selected history segment.

        Returns float32 array in volts, length n_samples.
        The scope outputs pre-scaled float values — no additional scaling needed.
        This matches the file export format verified in RTO-wfmparser.
        """
        if self._mode == "hardware":
            # Set binary float format (REAL,32, little-endian matches scope default)
            self.write("FORMat REAL,32")                        # VERIFY
            self.write("FORMat:BORDer LSBFirst")                # VERIFY: scope default
            raw = self._visa.query_binary_values(
                f"CHANnel{channel}:DATA?",                      # VERIFY
                datatype='f',
                is_big_endian=False,
                container=np.ndarray,
            )
            return raw.astype(np.float32)
        return np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # Measurement readout
    # ------------------------------------------------------------------

    def configure_measurement(self, meas_index: int, channel: int,
                               function: str = "MAX"):
        """
        Configure a scope measurement slot.

        Parameters
        ----------
        meas_index : int
            Measurement slot number (1–8 on RTO2024).
        channel : int
            Source channel (1–4).
        function : str
            Measurement function key from MEASUREMENT_FUNCTIONS.
        """
        if function not in MEASUREMENT_FUNCTIONS:
            raise ValueError(
                f"Unknown measurement function {function!r}. "
                f"Choose from: {list(MEASUREMENT_FUNCTIONS)}"
            )
        scpi_func = MEASUREMENT_FUNCTIONS[function]
        if self._mode == "hardware":
            self.write(f"MEASurement{meas_index}:SOURce CH{channel}")  # VERIFY
            self.write(f"MEASurement{meas_index}:MAIN {scpi_func}")    # VERIFY
            self.write(f"MEASurement{meas_index}:ENABle ON")           # VERIFY

    def read_measurement(self, meas_index: int) -> float:
        """
        Read the result of a configured measurement for the current history segment.

        Returns the measurement value in SI units (V for amplitude, V·s for area, etc.).
        Returns NaN if the measurement is not valid.
        """
        if self._mode == "hardware":
            raw = self.query(f"MEASurement{meas_index}:RESult:ACTual?")  # VERIFY
            try:
                return float(raw)
            except ValueError:
                return float("nan")
        return float("nan")

    # ------------------------------------------------------------------
    # Hardware connect / disconnect
    # ------------------------------------------------------------------

    def _connect_hardware(self):
        try:
            import pyvisa
        except ImportError as e:
            raise ImportError("pyvisa not installed. Run: pip install pyvisa pyvisa-py") from e

        self._rm   = pyvisa.ResourceManager()
        resource   = f"TCPIP0::{self._address}::inst0::INSTR"   # VERIFY LXI resource string
        self._visa = self._rm.open_resource(resource)
        self._visa.timeout = 30_000  # ms

        # Identify
        idn = self._visa.query("*IDN?").strip()
        if "RTO" not in idn:                                     # VERIFY IDN substring
            self._visa.close()
            raise RuntimeError(
                f"IDN query returned unexpected response: {idn!r}\n"
                "Check IP address and that the scope is in remote mode."
            )
        # Reset to known state
        self._visa.write("*RST")
        self.opc_wait()

    def _disconnect_hardware(self):
        if self._visa is not None:
            try:
                self.disable_history()
                self._visa.write("STOP")
                self._visa.close()
            except Exception:
                pass
        if self._rm is not None:
            try:
                self._rm.close()
            except Exception:
                pass

    def _connect_simulation(self):
        pass  # Nothing to open

    # ------------------------------------------------------------------
    # Simulation data generation
    # ------------------------------------------------------------------

    def sim_generate_waveforms(self,
                                n: int,
                                channels: list[int],
                                dark_rates: dict[int, float],
                                spe_amps: dict[int, float]) -> dict:
        """
        Generate n synthetic waveforms per channel.

        Returns
        -------
        dict with keys:
            "waveforms"  : {ch: np.ndarray shape (n, n_samples) float32, in volts}
            "timestamps" : np.ndarray shape (n,) float64, relative seconds
        """
        rng      = self._sim_rng
        n_samp   = self._n_samples
        x_start  = self._x_start_s
        dt       = self._dt_s
        t_axis   = np.linspace(x_start, x_start + (n_samp - 1) * dt, n_samp,
                               dtype=np.float32)

        # Relative timestamps: uniform spacing based on expected dark rate
        # (in real acquisition they come from the scope)
        mean_rate = np.mean([dark_rates.get(ch, SIM_DARK_RATE_HZ) for ch in channels])
        timestamps = np.cumsum(rng.exponential(1.0 / max(mean_rate, 1.0), n)).astype(np.float64)
        timestamps -= timestamps[0]

        waveforms = {}
        for ch in channels:
            dark_rate = dark_rates.get(ch, SIM_DARK_RATE_HZ)
            spe_amp   = spe_amps.get(ch, SIM_SPE_AMPLITUDE_V)
            waves     = np.zeros((n, n_samp), dtype=np.float32)

            for i in range(n):
                # Gaussian baseline noise
                wave = rng.normal(0.0, SIM_NOISE_V, n_samp).astype(np.float32)

                # Dark pulses in this window
                window_s    = n_samp * dt
                n_pulses    = rng.poisson(dark_rate * window_s)
                for _ in range(n_pulses):
                    peak_t = rng.uniform(x_start, x_start + window_s)
                    n_pe   = rng.choice([1, 2, 3], p=[0.85, 0.12, 0.03])
                    amp    = n_pe * spe_amp * rng.normal(1.0, 0.05)
                    pulse  = amp * np.exp(
                        -0.5 * ((t_axis - peak_t) / SIM_PULSE_SIGMA_S) ** 2
                    )
                    wave += pulse

                waves[i] = wave

            waveforms[ch] = waves

        return {"waveforms": waveforms, "timestamps": timestamps}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
