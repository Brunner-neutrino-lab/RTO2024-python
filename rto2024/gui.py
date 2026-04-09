"""
rto2024/gui.py

Standalone PyQt5 GUI for the R&S RTO2024 oscilloscope.

Launch directly:
    python -m rto2024.gui

Key features:
  - Switch between "measure" mode (scope-side amplitude) and
    "waveform" mode (PC-side pulse finding)
  - Optional raw waveform save in measure mode
  - Live waveform and spectrum plots after acquisition
"""

import sys
import time
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox,
    QDoubleSpinBox, QCheckBox, QTextEdit, QTabWidget, QGridLayout,
    QHeaderView, QTableWidget, QTableWidgetItem,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont

try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .controller import RTO2024Controller, AcquisitionResult
from .driver import MEASUREMENT_FUNCTIONS


# ---------------------------------------------------------------------------
# Worker signals
# ---------------------------------------------------------------------------

class _Signals(QObject):
    status           = pyqtSignal(str)
    connected        = pyqtSignal(bool, str)
    progress         = pyqtSignal(int, int)
    acquisition_done = pyqtSignal(object)


class _ConnectWorker(QThread):
    def __init__(self, ctrl: RTO2024Controller, signals: _Signals):
        super().__init__()
        self._ctrl    = ctrl
        self._signals = signals

    def run(self):
        try:
            self._ctrl.connect()
            idn = self._ctrl.identify()
            self._signals.connected.emit(True, idn)
        except Exception as e:
            self._signals.connected.emit(False, str(e))


class _AcquireWorker(QThread):
    def __init__(self, ctrl: RTO2024Controller, n: int, signals: _Signals):
        super().__init__()
        self._ctrl    = ctrl
        self._n       = n
        self._signals = signals

    def run(self):
        try:
            self._ctrl.on_progress = lambda a, t: self._signals.progress.emit(a, t)
            result = self._ctrl.run(self._n)
            self._signals.acquisition_done.emit(result)
        except Exception as e:
            self._signals.status.emit(f"Acquisition error: {e}")
        finally:
            self._ctrl.on_progress = None


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class RTO2024Window(QMainWindow):
    """
    Standalone GUI window for the R&S RTO2024.

    Tabs:
        Connection   — IP, mode, connect/disconnect
        Channels     — channel enable, scale, offset, trigger
        Acquisition  — record window, waveform count, acq mode, measurement function
        Waveforms    — raw waveform plot (last acquisition)
        Spectrum     — pulse amplitude histogram (last acquisition)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("R&S RTO2024 Oscilloscope Control")
        self.resize(950, 720)

        self._ctrl:    RTO2024Controller | None = None
        self._signals: _Signals                = _Signals()
        self._worker:  QThread | None          = None
        self._last_result: AcquisitionResult | None = None

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)

        tabs = QTabWidget()
        tabs.addTab(self._build_connection_tab(), "Connection")
        tabs.addTab(self._build_channels_tab(),   "Channels")
        tabs.addTab(self._build_acquisition_tab(),"Acquisition")
        if HAS_MPL:
            tabs.addTab(self._build_waveform_tab(),  "Waveforms")
            tabs.addTab(self._build_spectrum_tab(),  "Spectrum")

        lay.addWidget(tabs)
        lay.addWidget(self._build_status_log())

    # --- Connection tab ---
    def _build_connection_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        box = QGroupBox("Instrument Connection")
        g   = QGridLayout(box)

        g.addWidget(QLabel("IP Address:"), 0, 0)
        self._ip_edit = QLineEdit("192.168.0.2")
        g.addWidget(self._ip_edit, 0, 1)

        g.addWidget(QLabel("Mode:"), 1, 0)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["simulation", "hardware"])
        g.addWidget(self._mode_combo, 1, 1)

        btn_row = QHBoxLayout()
        self._connect_btn    = QPushButton("Connect")
        self._disconnect_btn = QPushButton("Disconnect")
        self._test_btn       = QPushButton("Test Connection")
        self._disconnect_btn.setEnabled(False)
        btn_row.addWidget(self._connect_btn)
        btn_row.addWidget(self._disconnect_btn)
        btn_row.addWidget(self._test_btn)
        g.addLayout(btn_row, 2, 0, 1, 2)

        self._status_label = QLabel("Not connected")
        self._status_label.setStyleSheet("color: red; font-weight: bold;")
        g.addWidget(self._status_label, 3, 0, 1, 2)

        lay.addWidget(box)
        lay.addStretch()
        return w

    # --- Channels tab ---
    def _build_channels_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        ch_box = QGroupBox("Channel Configuration")
        g      = QGridLayout(ch_box)
        g.addWidget(QLabel("Channel"), 0, 0)
        g.addWidget(QLabel("Enable"), 0, 1)
        g.addWidget(QLabel("Scale (V/div)"), 0, 2)
        g.addWidget(QLabel("Offset (V)"), 0, 3)

        self._ch_enable = []
        self._ch_scale  = []
        self._ch_offset = []

        for row, ch in enumerate([1, 2, 3, 4], start=1):
            g.addWidget(QLabel(f"CH{ch}"), row, 0)

            chk = QCheckBox()
            chk.setChecked(ch == 1)
            g.addWidget(chk, row, 1)
            self._ch_enable.append(chk)

            scale = QDoubleSpinBox()
            scale.setRange(0.001, 10.0)
            scale.setValue(0.05)
            scale.setSingleStep(0.01)
            scale.setDecimals(3)
            g.addWidget(scale, row, 2)
            self._ch_scale.append(scale)

            offset = QDoubleSpinBox()
            offset.setRange(-10.0, 10.0)
            offset.setValue(0.0)
            offset.setSingleStep(0.01)
            g.addWidget(offset, row, 3)
            self._ch_offset.append(offset)

        lay.addWidget(ch_box)

        # Trigger
        trig_box = QGroupBox("Trigger")
        t_lay    = QGridLayout(trig_box)
        t_lay.addWidget(QLabel("Source:"), 0, 0)
        self._trig_src_combo = QComboBox()
        self._trig_src_combo.addItems(["CH1", "CH2", "CH3", "CH4", "External"])
        t_lay.addWidget(self._trig_src_combo, 0, 1)

        t_lay.addWidget(QLabel("Level (mV):"), 1, 0)
        self._trig_level_spin = QDoubleSpinBox()
        self._trig_level_spin.setRange(-1000.0, 1000.0)
        self._trig_level_spin.setValue(10.0)
        self._trig_level_spin.setSingleStep(1.0)
        t_lay.addWidget(self._trig_level_spin, 1, 1)

        t_lay.addWidget(QLabel("Slope:"), 2, 0)
        self._trig_slope_combo = QComboBox()
        self._trig_slope_combo.addItems(["POS", "NEG"])
        t_lay.addWidget(self._trig_slope_combo, 2, 1)

        lay.addWidget(trig_box)
        lay.addStretch()
        return w

    # --- Acquisition tab ---
    def _build_acquisition_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        # Record window
        rw_box = QGroupBox("Record Window")
        g      = QGridLayout(rw_box)
        g.addWidget(QLabel("Pre-trigger (µs):"), 0, 0)
        self._pre_spin = QDoubleSpinBox()
        self._pre_spin.setRange(0.1, 1000.0)
        self._pre_spin.setValue(2.0)
        self._pre_spin.setSingleStep(0.5)
        g.addWidget(self._pre_spin, 0, 1)

        g.addWidget(QLabel("Post-trigger (µs):"), 1, 0)
        self._post_spin = QDoubleSpinBox()
        self._post_spin.setRange(0.1, 1000.0)
        self._post_spin.setValue(10.0)
        self._post_spin.setSingleStep(1.0)
        g.addWidget(self._post_spin, 1, 1)
        lay.addWidget(rw_box)

        # Acquisition mode
        mode_box = QGroupBox("Acquisition Mode")
        m_lay    = QGridLayout(mode_box)

        m_lay.addWidget(QLabel("Mode:"), 0, 0)
        self._acq_mode_combo = QComboBox()
        self._acq_mode_combo.addItems(["measure", "waveform"])
        self._acq_mode_combo.currentTextChanged.connect(self._on_acq_mode_changed)
        m_lay.addWidget(self._acq_mode_combo, 0, 1)

        m_lay.addWidget(QLabel("Measure function:"), 1, 0)
        self._meas_func_combo = QComboBox()
        self._meas_func_combo.addItems(list(MEASUREMENT_FUNCTIONS.keys()))
        m_lay.addWidget(self._meas_func_combo, 1, 1)

        self._save_wfm_check = QCheckBox("Also save raw waveforms (measure mode)")
        self._save_wfm_check.setChecked(False)
        m_lay.addWidget(self._save_wfm_check, 2, 0, 1, 2)

        m_lay.addWidget(QLabel("Pulse find threshold (mV, waveform mode):"), 3, 0)
        self._pf_thresh_spin = QDoubleSpinBox()
        self._pf_thresh_spin.setRange(0.1, 1000.0)
        self._pf_thresh_spin.setValue(10.0)
        self._pf_thresh_spin.setSingleStep(1.0)
        m_lay.addWidget(self._pf_thresh_spin, 3, 1)

        lay.addWidget(mode_box)

        # Waveform count + run
        run_box = QGroupBox("Run")
        r_lay   = QGridLayout(run_box)
        r_lay.addWidget(QLabel("Waveforms to acquire:"), 0, 0)
        self._n_wfm_spin = QSpinBox()
        self._n_wfm_spin.setRange(1, 50_000)
        self._n_wfm_spin.setValue(1000)
        r_lay.addWidget(self._n_wfm_spin, 0, 1)

        self._progress_label = QLabel("Ready")
        r_lay.addWidget(self._progress_label, 1, 0, 1, 2)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Acquisition")
        self._stop_btn  = QPushButton("Stop")
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        r_lay.addLayout(btn_row, 2, 0, 1, 2)
        lay.addWidget(run_box)
        lay.addStretch()
        return w

    # --- Waveform plot tab ---
    def _build_waveform_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Channel:"))
        self._wf_ch_combo = QComboBox()
        for ch in [1, 2, 3, 4]:
            self._wf_ch_combo.addItem(f"CH{ch}")
        self._wf_ch_combo.currentIndexChanged.connect(self._refresh_waveform_plot)
        ctrl_row.addWidget(self._wf_ch_combo)

        ctrl_row.addWidget(QLabel("Index:"))
        self._wf_idx_spin = QSpinBox()
        self._wf_idx_spin.setRange(0, 0)
        self._wf_idx_spin.valueChanged.connect(self._refresh_waveform_plot)
        ctrl_row.addWidget(self._wf_idx_spin)
        ctrl_row.addStretch()
        lay.addLayout(ctrl_row)

        self._wf_fig    = Figure(figsize=(8, 3))
        self._wf_canvas = FigureCanvas(self._wf_fig)
        self._wf_ax     = self._wf_fig.add_subplot(111)
        self._wf_ax.set_xlabel("Time (µs)")
        self._wf_ax.set_ylabel("Amplitude (V)")
        lay.addWidget(self._wf_canvas)
        return w

    # --- Spectrum tab ---
    def _build_spectrum_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Channel:"))
        self._spec_ch_combo = QComboBox()
        for ch in [1, 2, 3, 4]:
            self._spec_ch_combo.addItem(f"CH{ch}")
        self._spec_ch_combo.currentIndexChanged.connect(self._refresh_spectrum_plot)
        ctrl_row.addWidget(self._spec_ch_combo)

        ctrl_row.addWidget(QLabel("Bins:"))
        self._spec_bins = QSpinBox()
        self._spec_bins.setRange(10, 1000)
        self._spec_bins.setValue(100)
        self._spec_bins.valueChanged.connect(self._refresh_spectrum_plot)
        ctrl_row.addWidget(self._spec_bins)
        ctrl_row.addStretch()
        lay.addLayout(ctrl_row)

        self._spec_fig    = Figure(figsize=(8, 3))
        self._spec_canvas = FigureCanvas(self._spec_fig)
        self._spec_ax     = self._spec_fig.add_subplot(111)
        self._spec_ax.set_xlabel("Amplitude (V)")
        self._spec_ax.set_ylabel("Counts")
        lay.addWidget(self._spec_canvas)
        return w

    # --- Status log ---
    def _build_status_log(self) -> QWidget:
        box = QGroupBox("Status Log")
        lay = QVBoxLayout(box)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        self._log.setFont(QFont("Courier", 9))
        lay.addWidget(self._log)
        return box

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._connect_btn.clicked.connect(self._on_connect)
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._test_btn.clicked.connect(self._on_test)
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)

        self._signals.status.connect(self._log_message)
        self._signals.connected.connect(self._on_connect_result)
        self._signals.progress.connect(self._on_progress)
        self._signals.acquisition_done.connect(self._on_done)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_connect(self):
        ip   = self._ip_edit.text().strip()
        mode = self._mode_combo.currentText()
        self._ctrl = RTO2024Controller(address=ip, mode=mode)
        self._log_message(f"Connecting to {ip} ({mode} mode)...")
        self._connect_btn.setEnabled(False)
        w = _ConnectWorker(self._ctrl, self._signals)
        w.start(); self._worker = w

    def _on_connect_result(self, success: bool, msg: str):
        self._connect_btn.setEnabled(True)
        if success:
            self._status_label.setText(f"Connected: {msg}")
            self._status_label.setStyleSheet("color: green; font-weight: bold;")
            self._disconnect_btn.setEnabled(True)
            self._start_btn.setEnabled(True)
        else:
            self._status_label.setText("Connection failed")
            self._status_label.setStyleSheet("color: red; font-weight: bold;")
            self._ctrl = None
        self._log_message(("Connected: " if success else "Failed: ") + msg)

    def _on_disconnect(self):
        if self._ctrl:
            try:    self._ctrl.disconnect()
            except Exception as e: self._log_message(f"Disconnect error: {e}")
            self._ctrl = None
        self._status_label.setText("Not connected")
        self._status_label.setStyleSheet("color: red; font-weight: bold;")
        self._disconnect_btn.setEnabled(False)
        self._start_btn.setEnabled(False)
        self._log_message("Disconnected.")

    def _on_test(self):
        config = {"address": self._ip_edit.text().strip(),
                  "mode":    self._mode_combo.currentText()}
        self._log_message("Testing connection...")

        class _T(QThread):
            done = pyqtSignal(bool, str)
            def run(self_):
                ok, msg = RTO2024Controller.test(config)
                self_.done.emit(ok, msg)

        t = _T(self)
        t.done.connect(lambda ok, m: self._log_message(
            f"Test {'OK' if ok else 'FAILED'}: {m}"))
        t.start(); self._worker = t

    def _on_start(self):
        if not self._ctrl:
            return
        # Push all UI config to controller
        self._ctrl.configure_record_window(
            pre_us=self._pre_spin.value(), post_us=self._post_spin.value()
        )

        trig_text = self._trig_src_combo.currentText()
        if trig_text == "External":
            self._ctrl.configure_trigger(mode="external",
                                         level_v=self._trig_level_spin.value() / 1000.0,
                                         slope=self._trig_slope_combo.currentText())
        else:
            ch = int(trig_text.replace("CH", ""))
            self._ctrl.configure_trigger(source=ch,
                                         level_v=self._trig_level_spin.value() / 1000.0,
                                         slope=self._trig_slope_combo.currentText())

        channels = [i+1 for i, chk in enumerate(self._ch_enable) if chk.isChecked()]
        for i, ch in enumerate([1,2,3,4]):
            if self._ch_enable[i].isChecked():
                self._ctrl.configure_channel(
                    ch,
                    scale_v=self._ch_scale[i].value(),
                    offset_v=self._ch_offset[i].value()
                )

        acq_mode = self._acq_mode_combo.currentText()
        self._ctrl.configure_acquisition_mode(
            acq_mode, save_waveforms=self._save_wfm_check.isChecked()
        )
        self._ctrl.configure_measurement(function=self._meas_func_combo.currentText())
        self._ctrl.configure_pulse_finding(
            threshold_v=self._pf_thresh_spin.value() / 1000.0
        )

        n = self._n_wfm_spin.value()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._log_message(f"Starting: {n} waveforms, mode={acq_mode}")

        w = _AcquireWorker(self._ctrl, n, self._signals)
        w.start(); self._worker = w

    def _on_stop(self):
        if self._ctrl:
            try: self._ctrl._driver.write("STOP")
            except Exception: pass
        self._stop_btn.setEnabled(False)
        self._log_message("Stop requested.")

    def _on_progress(self, acquired: int, total: int):
        pct = 100 * acquired // total
        self._progress_label.setText(f"{acquired}/{total} ({pct}%)")

    def _on_done(self, result: AcquisitionResult):
        self._last_result = result
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        total_pulses = sum(len(result.amplitudes.get(ch, [])) for ch in result.channel_ids)
        self._progress_label.setText(
            f"Done — {result.n_waveforms} waveforms, {total_pulses} pulses"
        )
        self._log_message(
            f"Complete: {result.n_waveforms} waveforms, {total_pulses} pulses, "
            f"source={result.source}"
        )
        if HAS_MPL:
            self._refresh_waveform_plot()
            self._refresh_spectrum_plot()

    def _on_acq_mode_changed(self, mode: str):
        is_measure = (mode == "measure")
        self._meas_func_combo.setEnabled(is_measure)
        self._save_wfm_check.setEnabled(is_measure)
        self._pf_thresh_spin.setEnabled(not is_measure)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _refresh_waveform_plot(self):
        if not HAS_MPL or self._last_result is None:
            return
        ch   = self._wf_ch_combo.currentIndex() + 1
        wfms = self._last_result.waveforms.get(ch)
        if wfms is None or len(wfms) == 0:
            return
        self._wf_idx_spin.setMaximum(len(wfms) - 1)
        idx  = min(self._wf_idx_spin.value(), len(wfms) - 1)
        t_us = self._last_result.time_axis * 1e6 if self._last_result.time_axis is not None \
               else np.arange(len(wfms[idx]))
        self._wf_ax.clear()
        self._wf_ax.plot(t_us, wfms[idx], lw=0.8)
        self._wf_ax.axvline(0, color="red", ls="--", lw=0.8, label="trigger")
        self._wf_ax.set_xlabel("Time (µs)")
        self._wf_ax.set_ylabel("Amplitude (V)")
        self._wf_ax.set_title(f"Waveform — CH{ch}, index {idx}")
        self._wf_ax.legend(fontsize=8)
        self._wf_fig.tight_layout()
        self._wf_canvas.draw()

    def _refresh_spectrum_plot(self):
        if not HAS_MPL or self._last_result is None:
            return
        ch   = self._spec_ch_combo.currentIndex() + 1
        amps = self._last_result.amplitudes.get(ch)
        if amps is None or len(amps) == 0:
            return
        self._spec_ax.clear()
        self._spec_ax.hist(amps * 1000, bins=self._spec_bins.value(),
                           color="steelblue", edgecolor="none")
        self._spec_ax.set_xlabel("Amplitude (mV)")
        self._spec_ax.set_ylabel("Counts")
        self._spec_ax.set_title(
            f"Pulse spectrum — CH{ch}  (N={len(amps)}, "
            f"src={self._last_result.source})"
        )
        self._spec_fig.tight_layout()
        self._spec_canvas.draw()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _log_message(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {msg}")

    def closeEvent(self, event):
        if self._ctrl:
            try: self._ctrl.disconnect()
            except Exception: pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    win = RTO2024Window()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
