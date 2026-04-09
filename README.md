# rto2024-python

Python driver and GUI for the R&S RTO2024 oscilloscope.

## Acquisition Modes

| Mode | Who measures | Output | Use case |
|------|-------------|--------|----------|
| `measure` | Scope (built-in) | Amplitude + timestamp per pulse | Fast; low data volume |
| `waveform` | PC (pulse finding) | Full waveforms + amplitude + timestamp | Matches VX2740 path; full analysis flexibility |

In `measure` mode, set `save_waveforms=True` to also transfer raw waveforms
alongside the scope-measured amplitudes.

## Quick Start

```bash
pip install -r requirements.txt

# Standalone GUI
python -m rto2024.gui

# Headless example
python examples/basic_acquisition.py
```

## API

```python
from rto2024 import RTO2024Controller

with RTO2024Controller("192.168.0.2", mode="simulation") as rto:
    rto.configure_channels([1, 2], scale_v=0.05)
    rto.configure_record_window(pre_us=2.0, post_us=10.0)
    rto.configure_trigger(source=1, level_v=0.010, slope="POS")
    rto.configure_measurement(function="MAX")        # scope-side measurement
    rto.configure_acquisition_mode("measure")        # or "waveform"

    result = rto.run(n_waveforms=10_000)

# result.amplitudes[ch]  -> np.ndarray (V)
# result.timestamps[ch]  -> np.ndarray (s)
# result.waveforms[ch]   -> np.ndarray (N, n_samples) float32, V  [if stored]
# result.time_axis       -> np.ndarray (n_samples,) float32, s
# result.source          -> "rto_measure" | "rto_waveform" | "rto_simulation"
```

## Measurement Functions

| Key | SCPI | Description |
|-----|------|-------------|
| `MAX` | MAX | Maximum voltage — peak for positive SiPM pulses |
| `MIN` | MIN | Minimum voltage |
| `PEAK` | PEAK | Peak-to-peak |
| `AMPL` | AMPL | Amplitude (high − low level) |
| `MEAN` | MEAN | Mean value |
| `RMS` | RMS | RMS |
| `AREA` | AREA | Waveform area (V·s) — proportional to charge |

## Waveform Data Format

Waveform data is float32 in **volts** (pre-scaled by scope), matching the
`.Wfm.csv` export format. Time axis in seconds, with 0 at the trigger point.

Verified against reference data in `RTO-wfmparser/` using parameters:
- `HardwareXStart` / `SignalResolution` / `SignalHardwareRecordLength`
- `SignalFormat: FLOAT`, `ByteOrder: LSB first`

## Hardware SCPI Notes

All hardware-specific SCPI commands are marked `# VERIFY` in `rto2024/driver.py`.
Confirm against the RTO2024 User Manual (provided in `RTO-wfmparser/`) before
first hardware run. Key areas to verify:

| Feature | SCPI used | Section to check |
|---------|-----------|-----------------|
| Channel scale | `CHANnel<N>:SCALe` | Channel commands |
| Time base | `TIMebase:SCALe` | Timebase commands |
| Trigger level | `TRIGger:A:LEVel<N>` | Trigger commands |
| History enable | `ACQuire:SEGMented:STATe` | Fast segmentation |
| History size | `ACQuire:SEGMented:MAX` | Fast segmentation |
| History navigate | `ACQuire:HISTory:CURRent` | History commands |
| Timestamp | `ACQuire:HISTory:TSRelative?` | History commands |
| Waveform header | `CHANnel<N>:DATA:HEADer?` | Waveform export |
| Waveform data | `CHANnel<N>:DATA?` | Waveform export |
| Measurement | `MEASurement<N>:RESult:ACTual?` | Measurement commands |
