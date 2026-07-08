#!/usr/bin/env python3
"""
raw_edf_explorer.py

An interactive PyQt5 helper application for exploring raw, uncalibrated EDF files
from the BM28 XMaS beamline at ESRF.
Features:
- Premium Charcoal / Light theme toggle using an animated sliding switch.
- Auto-loads the first .edf file found in the example directory.
- Displays EDF ASCII header metadata in a clean panel.
- 2D Heatmap of raw MCA counts (with log/linear scale and standard interactive tools).
- 1D raw MCA spectrum slice at selected scan points (with log/linear scale).
- Full compatibility with Matplotlib standard interactive tools (zoom, pan, etc.).
- Mouse wheel scroll zoom centered at cursor on both plots.
- Auto Y-scaling on Home button click via callbacks.
"""

import os
import sys
import glob
import re
import json

RAW_EDF_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_edf_defaults.json")
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QSpinBox,
    QFrame, QSplitter, QCheckBox, QMessageBox, QTextEdit, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize, QPropertyAnimation, QRectF, pyqtProperty
from PyQt5.QtGui import QPainter, QColor, QBrush

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from data_loader import read_edf

class QToggleSwitch(QPushButton):
    """
    A custom premium sliding switch button widget.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # Color definitions
        self._track_color_on = QColor("#6366f1")
        self._track_color_off = QColor("#d4d4d8")
        self._thumb_color = QColor("#ffffff")
        
        # Thumb animation property
        self._thumb_position = 1.0  # Default to ON (Dark mode)
        self.animation = QPropertyAnimation(self, b"thumb_position", self)
        self.animation.setDuration(120)
        
    @pyqtProperty(float)
    def thumb_position(self):
        return self._thumb_position
        
    @thumb_position.setter
    def thumb_position(self, pos):
        self._thumb_position = pos
        self.update()
        
    def sizeHint(self):
        return QSize(50, 24)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw track
        track_rect = QRectF(0, 2, self.width(), self.height() - 4)
        color = self._track_color_on if self.isChecked() else self._track_color_off
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(track_rect, 10, 10)
        
        # Draw thumb
        thumb_size = self.height() - 6
        x_min = 3
        x_max = self.width() - thumb_size - 3
        x = x_min + self._thumb_position * (x_max - x_min)
        
        thumb_rect = QRectF(x, 3, thumb_size, thumb_size)
        painter.setBrush(QBrush(self._thumb_color))
        painter.drawEllipse(thumb_rect)
        
    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # Trigger animation
        target = 1.0 if self.isChecked() else 0.0
        self.animation.stop()
        self.animation.setStartValue(self._thumb_position)
        self.animation.setEndValue(target)
        self.animation.start()


class RawEDFExplorerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BM28 XMaS - Raw EDF File Explorer")
        self.setGeometry(100, 100, 1300, 850)
        
        # Data storage variables
        self.data_2d = None
        self.header = None
        self.file_path = None
        
        # State variables
        self.current_idx = 0
        self.use_log_scale_1d = True
        self.use_log_scale_2d = True
        
        # Load theme from config
        self.theme = "charcoal"
        if os.path.exists(RAW_EDF_CONFIG):
            try:
                with open(RAW_EDF_CONFIG, 'r') as f:
                    config = json.load(f)
                self.theme = config.get("theme", "charcoal")
            except Exception:
                pass
        
        # Stylesheet definitions
        self.styles = {
            "charcoal": {
                "qss": """
                    QMainWindow { background-color: #18181b; }
                    QWidget { background-color: #18181b; color: #f4f4f5; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }
                    QFrame { background-color: #27272a; border: 1px solid #3f3f46; border-radius: 6px; }
                    QLabel { border: none; background-color: transparent; }
                    QTextEdit { background-color: #18181b; color: #a1a1aa; border: 1px solid #3f3f46; border-radius: 4px; font-family: 'Consolas', monospace; font-size: 11px; }
                    QPushButton { background-color: #3f3f46; color: #f4f4f5; border: 1px solid #52525b; border-radius: 4px; padding: 6px 12px; font-weight: bold; }
                    QPushButton:hover { background-color: #52525b; }
                    QPushButton:pressed { background-color: #6366f1; }
                    QPushButton#btn_accent { background-color: #6366f1; border: 1px solid #4f46e5; }
                    QPushButton#btn_accent:hover { background-color: #4f46e5; }
                    QSlider::groove:horizontal { border: 1px solid #3f3f46; height: 8px; background: #27272a; border-radius: 4px; }
                    QSlider::handle:horizontal { background: #6366f1; border: 1px solid #4f46e5; width: 18px; margin: -5px 0; border-radius: 9px; }
                    QSlider::handle:horizontal:hover { background: #4f46e5; }
                    QSpinBox { background-color: #18181b; border: 1px solid #3f3f46; border-radius: 4px; padding: 4px; color: #f4f4f5; }
                    QCheckBox { background-color: transparent; }
                    QToolBar { background-color: #27272a; border: 1px solid #3f3f46; border-radius: 6px; padding: 2px; spacing: 4px; }
                    QToolButton { background-color: transparent; border: none; border-radius: 4px; padding: 4px; color: #f4f4f5; }
                    QToolButton:hover { background-color: #3f3f46; }
                    QToolButton:checked { background-color: #6366f1; }
                """,
                "fig_face": "#27272a",
                "ax_face": "#18181b",
                "text": "#f4f4f5",
                "grid": "#3f3f46",
                "spine": "#3f3f46",
                "line_color": "white"
            },
            "light": {
                "qss": """
                    QMainWindow { background-color: #f4f4f5; }
                    QWidget { background-color: #f4f4f5; color: #18181b; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }
                    QFrame { background-color: #ffffff; border: 1px solid #e4e4e7; border-radius: 6px; }
                    QLabel { border: none; background-color: transparent; }
                    QTextEdit { background-color: #f4f4f5; color: #71717a; border: 1px solid #d4d4d8; border-radius: 4px; font-family: 'Consolas', monospace; font-size: 11px; }
                    QPushButton { background-color: #e4e4e7; color: #18181b; border: 1px solid #d4d4d8; border-radius: 4px; padding: 6px 12px; font-weight: bold; }
                    QPushButton:hover { background-color: #d4d4d8; }
                    QPushButton:pressed { background-color: #3b82f6; color: white; }
                    QPushButton#btn_accent { background-color: #3b82f6; border: 1px solid #2563eb; color: white; }
                    QPushButton#btn_accent:hover { background-color: #2563eb; }
                    QSlider::groove:horizontal { border: 1px solid #d4d4d8; height: 8px; background: #e4e4e7; border-radius: 4px; }
                    QSlider::handle:horizontal { background: #3b82f6; border: 1px solid #2563eb; width: 18px; margin: -5px 0; border-radius: 9px; }
                    QSlider::handle:horizontal:hover { background: #2563eb; }
                    QSpinBox { background-color: #ffffff; border: 1px solid #d4d4d8; border-radius: 4px; padding: 4px; color: #18181b; }
                    QCheckBox { background-color: transparent; }
                    QToolBar { background-color: #ffffff; border: 1px solid #e4e4e7; border-radius: 6px; padding: 2px; spacing: 4px; }
                    QToolButton { background-color: transparent; border: none; border-radius: 4px; padding: 4px; color: #18181b; }
                    QToolButton:hover { background-color: #e4e4e7; }
                    QToolButton:checked { background-color: #3b82f6; color: white; }
                """,
                "fig_face": "#ffffff",
                "ax_face": "#f4f4f5",
                "text": "#18181b",
                "grid": "#d4d4d8",
                "spine": "#d4d4d8",
                "line_color": "black"
            }
        }
        
        self.init_ui()
        self.auto_load_files()

    def init_ui(self):
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter layout to separate left panel and right plots
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # ==========================================
        # LEFT PANEL: Loading, Selection, Metadata
        # ==========================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Top-left layout for theme toggle button
        top_bar = QHBoxLayout()
        self.btn_toggle_theme = QPushButton("☀️ Light Mode")
        self.btn_toggle_theme.clicked.connect(self.toggle_theme)
        top_bar.addWidget(self.btn_toggle_theme)
        top_bar.addStretch()
        left_layout.addLayout(top_bar)
        
        # Card 1: File Loading
        load_card = QFrame()
        load_card_layout = QVBoxLayout(load_card)
        
        self.lbl_file_path = QLabel("Raw File: No EDF loaded")
        self.lbl_file_path.setStyleSheet("color: #71717a; font-style: italic;")
        self.lbl_file_path.setWordWrap(True)
        
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("Load Raw EDF")
        btn_load.setObjectName("btn_accent")
        btn_load.clicked.connect(self.on_load_clicked)
        btn_layout.addWidget(btn_load)
        
        load_card_layout.addWidget(self.lbl_file_path)
        load_card_layout.addLayout(btn_layout)
        left_layout.addWidget(load_card)
        
        # Card 2: Scan Point Slider & SpinBox
        controls_card = QFrame()
        controls_layout = QVBoxLayout(controls_card)
        
        lbl_control_title = QLabel("Scan Point Slicer")
        lbl_control_title.setStyleSheet("font-weight: bold; color: #6366f1;")
        
        control_row = QHBoxLayout()
        self.spin_index = QSpinBox()
        self.spin_index.setRange(0, 4700)
        self.spin_index.setValue(0)
        self.spin_index.valueChanged.connect(self.on_spin_changed)
        
        lbl_info = QLabel("Select raw scan row index:")
        lbl_info.setStyleSheet("color: #71717a;")
        
        control_row.addWidget(lbl_info)
        control_row.addWidget(self.spin_index)
        
        self.slider_index = QSlider(Qt.Horizontal)
        self.slider_index.setRange(0, 4700)
        self.slider_index.setValue(0)
        self.slider_index.valueChanged.connect(self.on_slider_changed)
        
        controls_layout.addWidget(lbl_control_title)
        controls_layout.addLayout(control_row)
        controls_layout.addWidget(self.slider_index)
        left_layout.addWidget(controls_card)
        
        # Card 3: EDF Header Metadata Display
        meta_card = QFrame()
        meta_layout = QVBoxLayout(meta_card)
        lbl_meta_title = QLabel("EDF ASCII Header Metadata")
        lbl_meta_title.setStyleSheet("font-weight: bold; color: #71717a;")
        self.txt_metadata = QTextEdit()
        self.txt_metadata.setReadOnly(True)
        self.txt_metadata.setPlaceholderText("Header metadata will be printed here after loading.")
        
        meta_layout.addWidget(lbl_meta_title)
        meta_layout.addWidget(self.txt_metadata)
        left_layout.addWidget(meta_card)
        
        # ==========================================
        # RIGHT PANEL: Raw 2D Heatmap & 1D Spectrum
        # ==========================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Upper Splitter: Separates Heatmap and Spectrum
        right_splitter = QSplitter(Qt.Vertical)
        right_layout.addWidget(right_splitter)
        
        # --- 2D HEATMAP PANEL ---
        panel_heatmap = QWidget()
        heatmap_lay = QVBoxLayout(panel_heatmap)
        heatmap_lay.setContentsMargins(0, 0, 0, 0)
        
        heatmap_options = QHBoxLayout()
        lbl_heatmap_title = QLabel("Raw 2D MCA Data Map (Click to slice)")
        lbl_heatmap_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #6366f1;")
        
        self.chk_log_2d = QCheckBox("Log Scale (2D)")
        self.chk_log_2d.setChecked(True)
        self.chk_log_2d.stateChanged.connect(self.on_log_2d_toggled)
        
        heatmap_options.addWidget(lbl_heatmap_title)
        heatmap_options.addStretch()
        heatmap_options.addWidget(self.chk_log_2d)
        
        self.heatmap_fig = Figure(facecolor='#27272a', edgecolor='none')
        self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
        self.heatmap_canvas.mpl_connect('button_press_event', self.on_heatmap_click)
        self.heatmap_toolbar = NavigationToolbar(self.heatmap_canvas, self)
        
        heatmap_lay.addLayout(heatmap_options)
        heatmap_lay.addWidget(self.heatmap_toolbar)
        heatmap_lay.addWidget(self.heatmap_canvas)
        right_splitter.addWidget(panel_heatmap)
        
        # --- 1D SPECTRUM PANEL ---
        panel_spectrum = QWidget()
        spectrum_lay = QVBoxLayout(panel_spectrum)
        spectrum_lay.setContentsMargins(0, 0, 0, 0)
        
        spectrum_options = QHBoxLayout()
        lbl_spectrum_title = QLabel("Raw 1D MCA Spectrum Slice")
        lbl_spectrum_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #6366f1;")
        
        self.chk_log_1d = QCheckBox("Log Scale (1D)")
        self.chk_log_1d.setChecked(True)
        self.chk_log_1d.stateChanged.connect(self.on_log_1d_toggled)
        
        self.btn_save_spectrum = QPushButton("Save Spectrum")
        self.btn_save_spectrum.clicked.connect(self.on_save_spectrum_clicked)
        
        spectrum_options.addWidget(lbl_spectrum_title)
        spectrum_options.addStretch()
        spectrum_options.addWidget(self.chk_log_1d)
        spectrum_options.addWidget(self.btn_save_spectrum)
        
        self.mca_fig = Figure(facecolor='#27272a', edgecolor='none')
        self.mca_canvas = FigureCanvas(self.mca_fig)
        self.mca_ax = self.mca_fig.add_subplot(111, facecolor='#18181b')
        self.mca_toolbar = NavigationToolbar(self.mca_canvas, self)
        
        self.spectrum_line, = self.mca_ax.plot([], [], color='#6366f1', linewidth=1.5, label='Raw Counts')
        
        spectrum_lay.addLayout(spectrum_options)
        spectrum_lay.addWidget(self.mca_toolbar)
        spectrum_lay.addWidget(self.mca_canvas)
        right_splitter.addWidget(panel_spectrum)
        
        # Add panels to layout
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([380, 920])
        right_splitter.setSizes([450, 400])
        
        # Connect mouse scroll wheel zoom
        self.mca_canvas.mpl_connect('scroll_event', self.on_scroll)
        self.heatmap_canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Connect limit callbacks to auto-adjust Y limits on Home button click
        self.mca_ax.callbacks.connect('xlim_changed', self.on_mca_xlim_changed)
        
        # Apply initial theme
        self.apply_theme()

    def toggle_theme(self):
        self.theme = "light" if self.theme == "charcoal" else "charcoal"
        self.apply_theme()
        # Save theme to config file
        try:
            with open(RAW_EDF_CONFIG, 'w') as f:
                json.dump({"theme": self.theme}, f, indent=4)
        except Exception:
            pass

    def apply_theme(self):
        theme_cfg = self.styles[self.theme]
        self.setStyleSheet(theme_cfg["qss"])
        self.btn_toggle_theme.setText("☀️ Light Mode" if self.theme == "charcoal" else "🌙 Dark Mode")
        
        # Re-style Matplotlib figures
        self.mca_fig.patch.set_facecolor(theme_cfg["fig_face"])
        self.mca_ax.set_facecolor(theme_cfg["ax_face"])
        self.mca_ax.tick_params(colors=theme_cfg["text"], labelsize=13)
        self.mca_ax.xaxis.label.set_color(theme_cfg["text"])
        self.mca_ax.yaxis.label.set_color(theme_cfg["text"])
        self.mca_ax.title.set_color(theme_cfg["text"])
        self.mca_ax.grid(True, color=theme_cfg["grid"], linestyle=':', alpha=0.5)
        for spine in self.mca_ax.spines.values():
            spine.set_color(theme_cfg["spine"])
            
        self.heatmap_fig.patch.set_facecolor(theme_cfg["fig_face"])
        
        if self.data_2d is not None:
            self.plot_heatmap()
            self.plot_spectrum()
        else:
            self.mca_canvas.draw()
            self.heatmap_canvas.draw()

    def auto_load_files(self):
        target_dir = os.path.join(os.getcwd(), "example_data", "ZAP", "0223_P3")
        if os.path.exists(target_dir):
            files = glob.glob(os.path.join(target_dir, "*.edf"))
            if files:
                self.load_dataset(files[0])

    def on_load_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Raw EDF Scan", "", "ESRF Data Format Files (*.edf)"
        )
        if file_path:
            self.load_dataset(file_path)

    def load_dataset(self, file_path):
        try:
            self.header, self.data_2d = read_edf(file_path)
            self.file_path = file_path
            self.lbl_file_path.setText(f"Raw File: {os.path.basename(file_path)}")
            
            # Print metadata in TextEdit
            meta_str = ""
            for k, v in sorted(self.header.items()):
                meta_str += f"{k}: {v}\n"
            self.txt_metadata.setText(meta_str)
            
            # Configure controls
            num_points = self.data_2d.shape[0]
            num_channels = self.data_2d.shape[1]
            
            self.spin_index.setRange(0, num_points - 1)
            self.slider_index.setRange(0, num_points - 1)
            
            # Reset values
            self.current_idx = 0
            self.spin_index.setValue(0)
            self.slider_index.setValue(0)
            
            # Reset limits for MCA plot
            self.mca_ax.set_xlim(0, num_channels - 1)
            
            # Render plots
            self.plot_heatmap()
            self.plot_spectrum()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load raw EDF file:\n{str(e)}")

    def plot_heatmap(self):
        if self.data_2d is None:
            return
            
        cfg = self.styles[self.theme]
        self.heatmap_fig.clear()
        ax = self.heatmap_fig.add_subplot(111, facecolor=cfg["ax_face"])
        ax.tick_params(colors=cfg["text"], labelsize=9)
        
        num_points = self.data_2d.shape[0]
        num_channels = self.data_2d.shape[1]
        
        if self.use_log_scale_2d:
            display_data = np.log10(np.clip(self.data_2d, 1e-1, None))
        else:
            display_data = self.data_2d
            
        extent = [0, num_channels - 1, 0, num_points - 1]
        im = ax.imshow(display_data, cmap='jet', aspect='auto', interpolation='none', extent=extent, origin='upper')
        
        # Horizontal line showing current raw slice index
        self.heat_slice_line = ax.axhline(num_points - 1 - self.current_idx, color=cfg["line_color"], linestyle='-', linewidth=1.5)
        
        ax.set_xlabel('MCA Channel', color=cfg["text"], fontsize=13, fontweight='bold')
        ax.set_ylabel('Scan Point Row Index', color=cfg["text"], fontsize=13, fontweight='bold')
        ax.set_title("2D Raw Counts Map (log scale)" if self.use_log_scale_2d else "2D Raw Counts Map", color=cfg["text"], fontsize=14)
        
        for spine in ax.spines.values():
            spine.set_color(cfg["spine"])
            
        self.heatmap_fig.patch.set_facecolor(cfg["fig_face"])
        self.heatmap_fig.tight_layout()
        self.heatmap_canvas.draw()

    def plot_spectrum(self):
        if self.data_2d is None:
            return
            
        cfg = self.styles[self.theme]
        spectrum = self.data_2d[self.current_idx]
        channels = np.arange(len(spectrum))
        
        # 1. Update line data
        self.spectrum_line.set_data(channels, spectrum)
        
        # 2. Check if user is zoomed
        xlim = self.mca_ax.get_xlim()
        is_zoomed = not (abs(xlim[0] - 0.0) < 1.0 and abs(xlim[1] - (len(spectrum) - 1)) < 1.0)
        
        # Apply scale settings (only if not zoomed or scale changed)
        current_scale = self.mca_ax.get_yscale()
        target_scale = 'log' if self.use_log_scale_1d else 'linear'
        scale_changed = (current_scale != target_scale)
        
        if scale_changed or not is_zoomed:
            if self.use_log_scale_1d:
                self.mca_ax.set_yscale('log')
                self.mca_ax.set_ylim(max(1e-1, spectrum.min() * 0.5), spectrum.max() * 2.0)
            else:
                self.mca_ax.set_yscale('linear')
                self.mca_ax.set_ylim(0, spectrum.max() * 1.1)
                
        self.mca_ax.set_xlabel("MCA Channel", color=cfg["text"], fontsize=15, fontweight='bold')
        self.mca_ax.set_ylabel("Raw Counts", color=cfg["text"], fontsize=15, fontweight='bold')
        self.mca_ax.set_title(f"Raw MCA Spectrum Slice (Row {self.current_idx})", color=cfg["text"], fontsize=16)
        
        self.mca_canvas.draw_idle()

    def update_slice(self, index):
        if self.data_2d is None:
            return
            
        self.current_idx = int(np.clip(index, 0, self.data_2d.shape[0] - 1))
        
        self.spin_index.blockSignals(True)
        self.slider_index.blockSignals(True)
        
        self.spin_index.setValue(self.current_idx)
        self.slider_index.setValue(self.current_idx)
        
        self.spin_index.blockSignals(False)
        self.slider_index.blockSignals(False)
        
        # Update slice line on heatmap
        if hasattr(self, 'heat_slice_line'):
            num_points = self.data_2d.shape[0]
            self.heat_slice_line.set_ydata([num_points - 1 - self.current_idx])
            self.heatmap_canvas.draw_idle()
            
        self.plot_spectrum()

    def on_slider_changed(self, value):
        self.update_slice(value)

    def on_spin_changed(self, value):
        self.update_slice(value)

    def on_heatmap_click(self, event):
        tb = getattr(self.heatmap_canvas, 'toolbar', None)
        if tb is not None and tb.mode != "":
            return
        if event.inaxes is None:
            return
        click_y = event.ydata
        if click_y is not None and self.data_2d is not None:
            num_points = self.data_2d.shape[0]
            closest_idx = int(np.round(num_points - 1 - click_y))
            self.update_slice(closest_idx)

    def on_log_1d_toggled(self, state):
        self.use_log_scale_1d = (state == Qt.Checked)
        self.plot_spectrum()

    def on_log_2d_toggled(self, state):
        self.use_log_scale_2d = (state == Qt.Checked)
        self.plot_heatmap()

    def on_scroll(self, event):
        """Enable mouse wheel scrolling to zoom in/out centered at mouse cursor."""
        if event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        ax = event.inaxes
        base_scale = 1.15
        if event.step > 0:
            scale_factor = 1.0 / base_scale
        else:
            scale_factor = base_scale
            
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        
        # Calculate new X limits centered at mouse pointer
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        new_xlim = [xdata - new_width * (1 - rel_x), xdata + new_width * rel_x]
        
        # Calculate new Y limits centered at mouse pointer (supporting log-space correctly)
        if ax.get_yscale() == 'log':
            log_ymin = np.log10(max(1e-20, cur_ylim[0]))
            log_ymax = np.log10(max(1e-20, cur_ylim[1]))
            log_ydata = np.log10(max(1e-20, ydata))
            new_log_height = (log_ymax - log_ymin) * scale_factor
            rel_y = (log_ymax - log_ydata) / (log_ymax - log_ymin)
            new_ylim = [10**(log_ydata - new_log_height * (1 - rel_y)), 10**(log_ydata + new_log_height * rel_y)]
        else:
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            rel_y = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
            new_ylim = [ydata - new_height * (1 - rel_y), ydata + new_height * rel_y]
            
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        ax.figure.canvas.draw_idle()

    def on_mca_xlim_changed(self, ax):
        """Callback when MCA plot X limits change (e.g. click Home to autoscale Y)."""
        if self.data_2d is None:
            return
        xlim = ax.get_xlim()
        num_channels = self.data_2d.shape[1]
        # If we returned to default full limits (within small float margin), autoscale Y
        if abs(xlim[0] - 0.0) < 1.0 and abs(xlim[1] - (num_channels - 1)) < 1.0:
            spectrum = self.data_2d[self.current_idx]
            if self.use_log_scale_1d:
                ax.set_ylim(max(1e-1, spectrum.min() * 0.5), spectrum.max() * 2.0)
            else:
                ax.set_ylim(0, spectrum.max() * 1.1)
            ax.figure.canvas.draw_idle()

    def get_incident_energy_for_raw_row(self):
        """Tries to find the matching CSV file and extract incident energy for the current row."""
        if not self.file_path:
            return None
            
        base = os.path.basename(self.file_path)
        m = re.search(r"xia\d+_(\d+)_0000_0000\.edf", base)
        if not m:
            m = re.search(r"_(\d+)_", base)
            
        if not m:
            return None
            
        scan_num = int(m.group(1))
        target_dir = os.path.join(os.getcwd(), "example_data", "BL-align-XAS_batch1_output_csvs_01")
        if os.path.exists(target_dir):
            csv_files = glob.glob(os.path.join(target_dir, f"*scan_{scan_num}_*.csv"))
            if not csv_files:
                csv_files = glob.glob(os.path.join(target_dir, f"*_{scan_num}_*.csv"))
            if csv_files:
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_files[0])
                    if 'zap_energy' in df.columns:
                        energies = df['zap_energy'].values
                        if self.current_idx < len(energies):
                            return float(energies[self.current_idx])
                except:
                    pass
        return None

    def on_save_spectrum_clicked(self):
        """Saves the current raw 1D spectrum as a .txt file, multiplying raw counts by 100000 and presenting energies in keV."""
        if self.data_2d is None:
            QMessageBox.warning(self, "Save Error", "No raw dataset loaded.")
            return
            
        # Get current spectrum data
        spectrum = self.data_2d[self.current_idx]
        channels = np.arange(len(spectrum))
        
        # Try to get matching incident energy
        e_inc = self.get_incident_energy_for_raw_row()
        e_inc_kev = (e_inc / 1000.0) if e_inc is not None else None
        
        # Open save file dialog
        default_name = f"raw_spectrum_row_{self.current_idx}"
        if e_inc_kev is not None:
            default_name += f"_inc_{e_inc_kev:.4f}keV"
        default_name += ".txt"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Current Raw 1D Spectrum", default_name, "Text Files (*.txt)"
        )
        
        if not file_path:
            return
            
        try:
            # Build metadata header
            header_lines = [
                "# BM28 XMaS - Raw Uncalibrated 1D MCA Spectrum Slice (keV)",
                f"# Raw EDF File: {self.file_path}",
                f"# Row Index (Scan Point): {self.current_idx}"
            ]
            
            if e_inc_kev is not None:
                header_lines.append(f"# Incident Photon Energy: {e_inc_kev:.8f} keV (found in matched CSV)")
            else:
                header_lines.append("# Incident Photon Energy: Not Found (No matched CSV)")
                
            header_lines.extend([
                "# ",
                "# Columns:",
                "# 1: MCA Channel (Index)",
                "# 2: Raw Counts (Unnormalized)"
            ])
            
            data_to_save = np.column_stack((
                channels,
                spectrum * 10000000
            ))
            
            header_text = "\n".join(header_lines)
            
            np.savetxt(file_path, data_to_save, fmt=["%d", "%.8f"], header=header_text, comments="")
            QMessageBox.information(self, "Success", f"Raw spectrum saved successfully (scaled by 10000000 and in keV) to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save raw spectrum:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    window = RawEDFExplorerGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
