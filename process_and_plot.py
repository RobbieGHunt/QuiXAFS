#!/usr/bin/env python3
"""
process_and_plot.py

An interactive PyQt5 application for processing raw ZAP (MCA) data from BM28 XMaS.
Provides a GUI interface to:
1. Select the ZAP directory, CSV directory, output save directory, and output base filename.
2. Dynamically scan and display matching scans in a checklist.
3. Select/deselect scan numbers for averaging and calculations.
4. Normalize and average EDF scans, aligning them to matching CSV metadata and handling mismatched sizes.
5. Compute standard deviation of selected scans.
6. Calibrate emission energies using IUPAC reference lines.
7. Save the results as NumPy arrays (.npy) and plot/save the log-calibrated heatmap.
"""

import os
import sys
import re
import glob
import json

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zap_defaults.json")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QListWidget, QListWidgetItem,
    QProgressBar, QTextEdit, QSplitter, QFrame, QMessageBox, QGridLayout, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

CHARCOAL_QSS = """
    QMainWindow { background-color: #18181b; }
    QWidget { background-color: #18181b; color: #f4f4f5; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }
    QFrame { background-color: #27272a; border: 1px solid #3f3f46; border-radius: 6px; }
    QLabel { border: none; background-color: transparent; }
    QLineEdit { background-color: #18181b; border: 1px solid #3f3f46; border-radius: 4px; padding: 6px; color: #f4f4f5; }
    QListWidget { background-color: #18181b; border: 1px solid #3f3f46; border-radius: 4px; color: #f4f4f5; padding: 4px; }
    QListWidget::item { padding: 4px; }
    QListWidget::item:hover { background-color: #3f3f46; border-radius: 4px; }
    QListWidget::item:selected { background-color: #6366f1; color: #ffffff; border-radius: 4px; }
    QPushButton { background-color: #3f3f46; color: #f4f4f5; border: 1px solid #52525b; border-radius: 4px; padding: 6px 12px; font-weight: bold; }
    QPushButton:hover { background-color: #52525b; }
    QPushButton:pressed { background-color: #6366f1; }
    QPushButton#btn_accent { background-color: #6366f1; border: 1px solid #4f46e5; }
    QPushButton#btn_accent:hover { background-color: #4f46e5; }
    QPushButton#btn_accent:pressed { background-color: #4338ca; }
    QProgressBar { border: 1px solid #3f3f46; border-radius: 4px; text-align: center; background-color: #18181b; color: #ffffff; font-weight: bold; }
    QProgressBar::chunk { background-color: #6366f1; border-radius: 3px; }
    QSpinBox { background-color: #18181b; border: 1px solid #3f3f46; border-radius: 4px; padding: 4px; color: #f4f4f5; }
    QTextEdit { background-color: #18181b; border: 1px solid #3f3f46; border-radius: 4px; color: #f4f4f5; font-family: 'Consolas', 'Monaco', monospace; font-size: 12px; }
"""

class ProcessingWorker(QThread):
    progress = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(dict)
    
    def __init__(self, scan_numbers, matching_scans, save_dir, basename):
        super().__init__()
        self.scan_numbers = scan_numbers
        self.matching_scans = matching_scans
        self.save_dir = save_dir
        self.basename = basename
        self.is_running = True
        
    def run(self):
        try:
            normalized_scans = []
            energy_arrays = []
            processed_scans = []
            
            total = len(self.scan_numbers)
            for i, s_num in enumerate(self.scan_numbers):
                if not self.is_running:
                    self.log_signal.emit("Processing cancelled by user.")
                    self.finished.emit({"success": False, "error": "Cancelled"})
                    return
                    
                csv_path = self.matching_scans[s_num]['csv']
                edf_path = self.matching_scans[s_num]['edf']
                
                self.log_signal.emit(f"Processing scan {s_num}...")
                
                try:
                    # Load CSV
                    if not os.path.exists(csv_path):
                        self.log_signal.emit(f"  Warning: CSV file for scan {s_num} not found. Skipping.")
                        continue
                    df = pd.read_csv(csv_path)
                    if 'zap_Iref_p' not in df.columns or 'zap_energy' not in df.columns:
                        self.log_signal.emit(f"  Warning: CSV for scan {s_num} lacks required columns. Skipping.")
                        continue
                    iref = df['zap_Iref_p'].values
                    energy = df['zap_energy'].values
                    num_points = len(iref)
                    
                    # Load EDF
                    if not os.path.exists(edf_path):
                        self.log_signal.emit(f"  Warning: EDF file for scan {s_num} not found. Skipping.")
                        continue
                        
                    with open(edf_path, 'rb') as f:
                        f.seek(1024)  # Skip 1024-byte ASCII header
                        edf_raw = np.fromfile(f, dtype='<i4')
                        
                    # Reshape. Assume 4096 channels.
                    dim_1 = 4096
                    dim_2 = len(edf_raw) // dim_1
                    edf_data = edf_raw[:dim_2 * dim_1].reshape((dim_2, dim_1))
                    
                    # Align shapes to the minimum common length to handle missing/corrupted rows
                    min_points = min(edf_data.shape[0], num_points)
                    if min_points == 0:
                        self.log_signal.emit(f"  Warning: Scan {s_num} has empty datasets. Skipping.")
                        continue
                        
                    if edf_data.shape[0] != num_points:
                        self.log_signal.emit(f"  Note: Row count mismatch in scan {s_num} (EDF={edf_data.shape[0]}, CSV={num_points}). Aligning to {min_points} points.")
                        
                    edf_data = edf_data[:min_points].astype(np.float64)
                    iref_fixed = iref[:min_points]
                    energy_fixed = energy[:min_points]
                    
                    # Avoid division by zero
                    if np.any(iref_fixed == 0):
                        iref_fixed = np.where(iref_fixed == 0, 1.0, iref_fixed)
                        
                    # Normalize
                    normalized_data = edf_data / iref_fixed[:, np.newaxis]
                    normalized_scans.append(normalized_data)
                    energy_arrays.append(energy_fixed)
                    processed_scans.append(s_num)
                    
                    self.log_signal.emit(f"  Successfully processed scan {s_num}")
                    
                except Exception as e:
                    self.log_signal.emit(f"  Error processing scan {s_num}: {str(e)}")
                    continue
                    
                prog_val = int((i + 1) * 100 / total)
                self.progress.emit(prog_val)
                
            if not normalized_scans:
                self.log_signal.emit("Error: No scans were successfully processed.")
                self.finished.emit({"success": False, "error": "No scans processed"})
                return
                
            self.log_signal.emit("Aligning selected scans to a common minimum length...")
            min_length = min(scan.shape[0] for scan in normalized_scans)
            self.log_signal.emit(f"Common length determined: {min_length} points")
            
            normalized_scans_aligned = [scan[:min_length] for scan in normalized_scans]
            energy_arrays_aligned = [arr[:min_length] for arr in energy_arrays]
            
            self.log_signal.emit("Stacking datasets and computing statistics...")
            normalized_array_3d = np.stack(normalized_scans_aligned, axis=0)
            energy_array_2d = np.stack(energy_arrays_aligned, axis=0)
            
            # Averages and standard deviations
            average_data = np.mean(normalized_array_3d, axis=0)
            if len(normalized_scans_aligned) > 1:
                std_dev = np.std(normalized_array_3d, axis=0, ddof=1)
            else:
                self.log_signal.emit("Note: Only 1 scan processed; standard deviation set to zero.")
                std_dev = np.zeros_like(average_data)
                
            average_energy = np.mean(energy_array_2d, axis=0)
            
            # Calibration parameters
            G = 3.866022
            O = -146.330606
            mca_energies = G * np.arange(4096) + O
            
            # Save files
            avg_name = f"{self.basename}_avg.npy"
            std_name = f"{self.basename}_std.npy"
            energy_in_name = f"{self.basename}_energy_in.npy"
            energy_out_name = f"{self.basename}_energy_out.npy"
            combined_name = f"{self.basename}.npz"
            
            avg_path = os.path.join(self.save_dir, avg_name)
            std_path = os.path.join(self.save_dir, std_name)
            energy_in_path = os.path.join(self.save_dir, energy_in_name)
            energy_out_path = os.path.join(self.save_dir, energy_out_name)
            combined_path = os.path.join(self.save_dir, combined_name)
            
            np.save(avg_path, average_data)
            np.save(std_path, std_dev)
            np.save(energy_in_path, average_energy)
            np.save(energy_out_path, mca_energies)
            np.savez_compressed(
                combined_path,
                average_data=average_data,
                std_dev=std_dev,
                energy_in=average_energy,
                energy_out=mca_energies
            )
            
            self.log_signal.emit(f"Saved average to: {avg_path}")
            self.log_signal.emit(f"Saved standard deviation to: {std_path}")
            self.log_signal.emit(f"Saved incident energy axis to: {energy_in_path}")
            self.log_signal.emit(f"Saved emission energy axis to: {energy_out_path}")
            self.log_signal.emit(f"Saved combined dataset to: {combined_path}")
            
            self.finished.emit({
                "success": True,
                "average_data": average_data,
                "average_energy": average_energy,
                "mca_energies": mca_energies,
                "processed_scans": processed_scans
            })
            
        except Exception as e:
            self.log_signal.emit(f"General processing error: {str(e)}")
            self.finished.emit({"success": False, "error": str(e)})
            
    def stop(self):
        self.is_running = False

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#27272a')
        self.axes = fig.add_subplot(111, facecolor='#18181b')
        super().__init__(fig)
        self.setParent(parent)
        
        self.axes.tick_params(colors='#f4f4f5', labelsize=9)
        self.axes.xaxis.label.set_color('#f4f4f5')
        self.axes.yaxis.label.set_color('#f4f4f5')
        for spine in self.axes.spines.values():
            spine.set_color('#3f3f46')
        
        self.axes.text(0.5, 0.5, "No data processed yet.\nSelect directories and click 'Process & Save' to plot.",
                       color='#a1a1aa', fontsize=12, ha='center', va='center', transform=self.axes.transAxes)

class ZAPProcessingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BM28 XMaS - ZAP Data Processing")
        self.setGeometry(100, 100, 1200, 750)
        self.setStyleSheet(CHARCOAL_QSS)
        
        self.matching_scans = {}
        self.colorbar = None
        self.worker = None
        
        self.init_ui()
        self.load_defaults()
        
    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        
        # Splitter to allow resizing of left settings panel and right plot panel
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left Panel (Controls & Inputs)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        # Header
        lbl_title = QLabel("ZAP Data Processing Interface")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #6366f1; padding-bottom: 5px;")
        left_layout.addWidget(lbl_title)
        
        # 1. Directory inputs Frame
        dir_frame = QFrame()
        dir_layout = QGridLayout(dir_frame)
        dir_layout.setContentsMargins(10, 10, 10, 10)
        dir_layout.setSpacing(8)
        
        dir_layout.addWidget(QLabel("ZAP (EDF) Directory:"), 0, 0)
        self.txt_zap_dir = QLineEdit()
        self.txt_zap_dir.editingFinished.connect(self.scan_directories)
        btn_zap = QPushButton("Browse...")
        btn_zap.clicked.connect(self.browse_zap_dir)
        dir_layout.addWidget(self.txt_zap_dir, 0, 1)
        dir_layout.addWidget(btn_zap, 0, 2)
        
        dir_layout.addWidget(QLabel("CSV Directory:"), 1, 0)
        self.txt_csv_dir = QLineEdit()
        self.txt_csv_dir.editingFinished.connect(self.scan_directories)
        btn_csv = QPushButton("Browse...")
        btn_csv.clicked.connect(self.browse_csv_dir)
        dir_layout.addWidget(self.txt_csv_dir, 1, 1)
        dir_layout.addWidget(btn_csv, 1, 2)
        
        dir_layout.addWidget(QLabel("Save Directory:"), 2, 0)
        self.txt_save_dir = QLineEdit()
        btn_save = QPushButton("Browse...")
        btn_save.clicked.connect(self.browse_save_dir)
        dir_layout.addWidget(self.txt_save_dir, 2, 1)
        dir_layout.addWidget(btn_save, 2, 2)
        
        dir_layout.addWidget(QLabel("Output Base Name:"), 3, 0)
        self.txt_basename = QLineEdit("averaged_zap")
        dir_layout.addWidget(self.txt_basename, 3, 1, 1, 2)
        
        # Row 4: Default configuration actions
        defaults_layout = QHBoxLayout()
        btn_save_defaults = QPushButton("Save as Defaults")
        btn_save_defaults.clicked.connect(self.save_defaults_action)
        btn_clear_defaults = QPushButton("Clear Defaults")
        btn_clear_defaults.clicked.connect(self.clear_defaults_action)
        defaults_layout.addWidget(btn_save_defaults)
        defaults_layout.addWidget(btn_clear_defaults)
        dir_layout.addLayout(defaults_layout, 4, 0, 1, 3)
        
        left_layout.addWidget(dir_frame)
        
        # 2. Scan Selection checklist Frame
        scan_frame = QFrame()
        scan_layout = QVBoxLayout(scan_frame)
        scan_layout.setContentsMargins(10, 10, 10, 10)
        scan_layout.setSpacing(6)
        
        scan_header_layout = QHBoxLayout()
        self.lbl_scan_status = QLabel("Matched Scans: 0")
        self.lbl_scan_status.setStyleSheet("font-weight: bold; color: #a1a1aa;")
        scan_header_layout.addWidget(self.lbl_scan_status)
        scan_header_layout.addStretch()
        
        btn_sel_all = QPushButton("Select All")
        btn_sel_all.clicked.connect(self.select_all_scans)
        btn_clear_all = QPushButton("Clear All")
        btn_clear_all.clicked.connect(self.clear_all_scans)
        scan_header_layout.addWidget(btn_sel_all)
        scan_header_layout.addWidget(btn_clear_all)
        scan_layout.addLayout(scan_header_layout)
        
        # Range selection UI
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Range - Start:"))
        self.spin_start = QSpinBox()
        self.spin_start.setMinimum(0)
        self.spin_start.setMaximum(999999)
        range_layout.addWidget(self.spin_start)
        
        range_layout.addWidget(QLabel("Finish:"))
        self.spin_finish = QSpinBox()
        self.spin_finish.setMinimum(0)
        self.spin_finish.setMaximum(999999)
        range_layout.addWidget(self.spin_finish)
        
        btn_apply_range = QPushButton("Select Range")
        btn_apply_range.clicked.connect(self.select_range_scans)
        range_layout.addWidget(btn_apply_range)
        scan_layout.addLayout(range_layout)
        
        self.list_scans = QListWidget()
        scan_layout.addWidget(self.list_scans)
        
        left_layout.addWidget(scan_frame)
        
        # 3. Progress and Log Frame
        progress_frame = QFrame()
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(10, 10, 10, 10)
        progress_layout.setSpacing(8)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setPlaceholderText("Processing output log will appear here...")
        progress_layout.addWidget(self.txt_log)
        
        left_layout.addWidget(progress_frame)
        
        # Action Run Button
        self.btn_run = QPushButton("Process & Save")
        self.btn_run.setObjectName("btn_accent")
        self.btn_run.setStyleSheet("font-size: 15px; padding: 10px;")
        self.btn_run.clicked.connect(self.run_processing)
        left_layout.addWidget(self.btn_run)
        
        # Add left panel to splitter
        splitter.addWidget(left_widget)
        
        # Right Panel (Matplotlib plot view)
        right_widget = QFrame()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(4)
        
        # Embed canvas
        self.canvas = MplCanvas(self, width=6, height=6, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
        splitter.addWidget(right_widget)
        
        # Initial proportions: 40% left panel, 60% right panel
        splitter.setSizes([450, 750])
        
    def load_defaults(self):
        # Start with empty directories
        self.txt_zap_dir.setText("")
        self.txt_csv_dir.setText("")
        self.txt_save_dir.setText("")
        
        # Load from config file if exists
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                self.txt_zap_dir.setText(config.get("zap_dir", ""))
                self.txt_csv_dir.setText(config.get("csv_dir", ""))
                self.txt_save_dir.setText(config.get("save_dir", ""))
                self.txt_basename.setText(config.get("basename", "averaged_zap"))
            except Exception:
                pass
                
        self.scan_directories()
        
    def save_defaults_action(self):
        config = {
            "zap_dir": self.txt_zap_dir.text().strip(),
            "csv_dir": self.txt_csv_dir.text().strip(),
            "save_dir": self.txt_save_dir.text().strip(),
            "basename": self.txt_basename.text().strip()
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            QMessageBox.information(self, "Defaults Saved", "Current paths and base name have been saved as your defaults!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save defaults:\n{str(e)}")

    def clear_defaults_action(self):
        if os.path.exists(CONFIG_FILE):
            try:
                os.remove(CONFIG_FILE)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear defaults file:\n{str(e)}")
                return
        self.txt_zap_dir.setText("")
        self.txt_csv_dir.setText("")
        self.txt_save_dir.setText("")
        self.txt_basename.setText("averaged_zap")
        self.scan_directories()
        QMessageBox.information(self, "Defaults Cleared", "Defaults cleared and inputs reset.")
        
    def browse_zap_dir(self):
        start_dir = self.txt_zap_dir.text().strip()
        if not start_dir:
            start_dir = os.getcwd()
        path = QFileDialog.getExistingDirectory(self, "Select ZAP (EDF) Directory", start_dir)
        if path:
            self.txt_zap_dir.setText(os.path.normpath(path))
            self.scan_directories()
            
    def browse_csv_dir(self):
        start_dir = self.txt_csv_dir.text().strip()
        if not start_dir:
            start_dir = os.getcwd()
        path = QFileDialog.getExistingDirectory(self, "Select CSV Directory", start_dir)
        if path:
            self.txt_csv_dir.setText(os.path.normpath(path))
            self.scan_directories()
            
    def browse_save_dir(self):
        start_dir = self.txt_save_dir.text().strip()
        if not start_dir:
            start_dir = os.getcwd()
        path = QFileDialog.getExistingDirectory(self, "Select Save Directory", start_dir)
        if path:
            self.txt_save_dir.setText(os.path.normpath(path))
            
    def scan_directories(self):
        zap_dir = self.txt_zap_dir.text().strip()
        csv_dir = self.txt_csv_dir.text().strip()
        
        self.list_scans.clear()
        self.lbl_scan_status.setText("Matched Scans: 0")
        self.matching_scans = {}
        
        if not zap_dir or not csv_dir or not os.path.isdir(zap_dir) or not os.path.isdir(csv_dir):
            return
            
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        edf_files = glob.glob(os.path.join(zap_dir, "*.edf"))
        
        scan_map = {}
        for f in csv_files:
            m = re.search(r"scan_(\d+)_", os.path.basename(f))
            if m:
                scan_num = int(m.group(1))
                scan_map[scan_num] = {'csv': f, 'edf': None}
                
        for f in edf_files:
            m = re.search(r"xia\d+_(\d+)_0000_0000\.edf", os.path.basename(f))
            if m:
                scan_num = int(m.group(1))
                if scan_num in scan_map:
                    scan_map[scan_num]['edf'] = f
                    
        self.matching_scans = {k: v for k, v in scan_map.items() if v['edf'] is not None}
        scan_numbers = sorted(list(self.matching_scans.keys()))
        
        if not scan_numbers:
            self.lbl_scan_status.setText("Matched Scans: 0 (No matching scans found)")
            self.spin_start.setValue(0)
            self.spin_finish.setValue(0)
            return
            
        self.lbl_scan_status.setText(f"Matched Scans: {len(scan_numbers)}")
        
        min_scan = scan_numbers[0]
        max_scan = scan_numbers[-1]
        self.spin_start.blockSignals(True)
        self.spin_finish.blockSignals(True)
        self.spin_start.setRange(min_scan, max_scan)
        self.spin_finish.setRange(min_scan, max_scan)
        self.spin_start.setValue(min_scan)
        self.spin_finish.setValue(max_scan)
        self.spin_start.blockSignals(False)
        self.spin_finish.blockSignals(False)
        
        for s_num in scan_numbers:
            item = QListWidgetItem(f"Scan {s_num}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, s_num)
            self.list_scans.addItem(item)
            
    def select_all_scans(self):
        for i in range(self.list_scans.count()):
            self.list_scans.item(i).setCheckState(Qt.Checked)
            
    def clear_all_scans(self):
        for i in range(self.list_scans.count()):
            self.list_scans.item(i).setCheckState(Qt.Unchecked)
            
    def select_range_scans(self):
        start_val = self.spin_start.value()
        finish_val = self.spin_finish.value()
        lower = min(start_val, finish_val)
        upper = max(start_val, finish_val)
        for i in range(self.list_scans.count()):
            item = self.list_scans.item(i)
            s_num = item.data(Qt.UserRole)
            if lower <= s_num <= upper:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            
    def log_message(self, message):
        self.txt_log.append(message)
        self.txt_log.ensureCursorVisible()
        
    def run_processing(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.btn_run.setText("Process & Save")
            return
            
        save_dir = self.txt_save_dir.text()
        basename = self.txt_basename.text().strip()
        
        if not os.path.isdir(save_dir):
            QMessageBox.warning(self, "Invalid Directory", "The specified save directory does not exist.")
            return
            
        if not basename:
            QMessageBox.warning(self, "Missing Name", "Please enter a base output name.")
            return
            
        # Get selected scans
        selected_scan_numbers = []
        for i in range(self.list_scans.count()):
            item = self.list_scans.item(i)
            if item.checkState() == Qt.Checked:
                selected_scan_numbers.append(item.data(Qt.UserRole))
                
        if not selected_scan_numbers:
            QMessageBox.warning(self, "No Scans Selected", "Please select at least one scan to process.")
            return
            
        # Start worker thread
        self.progress_bar.setValue(0)
        self.txt_log.clear()
        self.btn_run.setText("Cancel Processing")
        self.btn_run.setStyleSheet("background-color: #ef4444; border: 1px solid #dc2626;")
        
        self.worker = ProcessingWorker(selected_scan_numbers, self.matching_scans, save_dir, basename)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log_signal.connect(self.log_message)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()
        
    def on_processing_finished(self, result):
        # Reset button style
        self.btn_run.setText("Process & Save")
        self.btn_run.setStyleSheet("")
        
        if not result.get("success", False):
            if result.get("error") != "Cancelled":
                QMessageBox.critical(self, "Error", f"Processing failed:\n{result.get('error')}")
            return
            
        self.progress_bar.setValue(100)
        
        average_data = result["average_data"]
        average_energy = result["average_energy"]
        mca_energies = result["mca_energies"]
        
        save_dir = self.txt_save_dir.text()
        basename = self.txt_basename.text().strip()
        
        self.plot_heatmap(average_data, average_energy, mca_energies, save_dir, basename)
        
        QMessageBox.information(
            self, "Processing Finished", 
            f"Successfully processed {len(result['processed_scans'])} scans!\n\n"
            f"Single combined file saved under save directory:\n"
            f" - {basename}.npz (All datasets packed together)\n\n"
            f"Individual array files saved (for backwards compatibility):\n"
            f" - {basename}_avg.npy\n"
            f" - {basename}_std.npy\n"
            f" - {basename}_energy_in.npy\n"
            f" - {basename}_energy_out.npy\n\n"
            f"Calibrated 2D heatmap plot saved as:\n"
            f" - {basename}_heatmap.png"
        )
        
    def plot_heatmap(self, average_data, average_energy, mca_energies, save_dir, basename):
        self.canvas.axes.clear()
        
        # Set dark colors
        self.canvas.axes.tick_params(colors='#f4f4f5', labelsize=9)
        self.canvas.axes.xaxis.label.set_color('#f4f4f5')
        self.canvas.axes.yaxis.label.set_color('#f4f4f5')
        for spine in self.canvas.axes.spines.values():
            spine.set_color('#3f3f46')
            
        # Crop the emission energy to the region of interest (2000 eV to 8500 eV)
        idx_crop = np.where((mca_energies >= 2000) & (mca_energies <= 8500))[0]
        c_start, c_end = idx_crop[0], idx_crop[-1] + 1
        
        cropped_data = average_data[:, c_start:c_end]
        cropped_mca_energies = mca_energies[c_start:c_end]
        
        energy_min = average_energy.min()
        energy_max = average_energy.max()
        
        log_data = np.log10(np.clip(cropped_data, 1e-6, None))
        
        im = self.canvas.axes.imshow(
            log_data, cmap='jet', aspect='auto', interpolation='none',
            extent=[cropped_mca_energies[0], cropped_mca_energies[-1], energy_min, energy_max]
        )
        
        # Clear colorbar if it exists
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None
            
        # Draw reference emission lines
        ref_lines = {
            r'Ar K-L3 (2958 eV)': 2958.0,
            r'Tb L3-M5 (6275 eV)': 6275.0,
            r'Co K-L3 (6930 eV)': 6930.0,
            r'Co K-M3 (7649 eV)': 7649.0
        }
        
        for label, val in ref_lines.items():
            self.canvas.axes.axvline(val, color='white', linestyle='--', alpha=0.5, linewidth=1.2)
            self.canvas.axes.text(val + 50, energy_max - 50, label, color='white', rotation=90, 
                                 verticalalignment='top', fontsize=8, fontweight='bold',
                                 bbox=dict(facecolor='#18181b', alpha=0.7, edgecolor='none', pad=2))
            
        # Draw elastic diagonal line
        self.canvas.axes.plot([energy_min, energy_max], [energy_min, energy_max], 'w--', alpha=0.5, linewidth=1.2, label='Elastic Scatter (Diagonal)')
        
        # Add colorbar
        self.colorbar = self.canvas.figure.colorbar(im, ax=self.canvas.axes, pad=0.02)
        self.colorbar.ax.yaxis.set_tick_params(colors='#f4f4f5', labelsize=9)
        self.colorbar.set_label('Log10 Normalized Intensity', color='#f4f4f5', fontsize=10)
        
        self.canvas.axes.set_xlabel('Emission Energy (eV)', fontsize=10)
        self.canvas.axes.set_ylabel('Incident Photon Energy (eV)', fontsize=10)
        self.canvas.axes.set_title('2D Average ZAP Dataset - Energy-Calibrated Heatmap', color='#f4f4f5', fontsize=12, fontweight='bold')
        self.canvas.axes.legend(loc='lower left', facecolor='#27272a', edgecolor='#3f3f46', labelcolor='#f4f4f5', fontsize=9)
        
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        
        # Save plot to PNG
        plot_path = os.path.join(save_dir, f"{basename}_heatmap.png")
        self.canvas.figure.savefig(plot_path, dpi=300)
        self.log_message(f"Saved calibrated 2D average heatmap plot to: {plot_path}")

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = ZAPProcessingGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
