#!/usr/bin/env python3
"""
xas_explorer_gui.py

An interactive PyQt5 application for exploring calibrated 2D ZAP datasets
from the BM28 XMaS beamline of the ESRF.
Features:
- Charcoal / Light theme toggle using an animated sliding switch.
- Region of Interest (ROI) integration tool with DRAGGABLE vertical lines on the 1D spectrum plot.
- Element picker using a periodic table dialog to dynamically select elements to fit.
- Local emission lines database integration.
- Custom background subtraction & EXAFS Fourier Transform (R-space) analysis.
"""

import os
import sys
import json
import warnings
import multiprocessing
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QDoubleSpinBox, QSpinBox, QComboBox,
    QFrame, QSplitter, QCheckBox, QMessageBox, QTabWidget, QSizePolicy, QProgressBar,
    QGridLayout, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QSize, QPropertyAnimation, QRectF, pyqtProperty, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QBrush, QFont

from scipy.optimize import least_squares, minimize
from scipy.special import erfc, erfcx, i0
from scipy.interpolate import LSQUnivariateSpline

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

INV_SQRT2 = 0.7071067811865475

# ---------------------------------------------------------
# Periodic Table Dialog for Element Picker
# ---------------------------------------------------------
class PeriodicTableDialog(QDialog):
    def __init__(self, database, selected_elements, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Periodic Table - Select Elements for Peak Fitting")
        self.database = database
        self.selected_elements = list(selected_elements)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        info_label = QLabel("Click elements to select/deselect them for peak fitting. Selected elements will be modeled.")
        info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px; color: #6366f1;")
        layout.addWidget(info_label)
        
        grid = QGridLayout()
        grid.setSpacing(6)
        
        # Periodic Table grid positions for key elements in the database
        # elements: symbol, row, col
        table_elements = [
            ("H", 1, 1), ("He", 1, 18),
            ("Li", 2, 1), ("Be", 2, 2), ("B", 2, 13), ("C", 2, 14), ("N", 2, 15), ("O", 2, 16), ("F", 2, 17), ("Ne", 2, 18),
            ("Na", 3, 1), ("Mg", 3, 2), ("Al", 3, 13), ("Si", 3, 14), ("P", 3, 15), ("S", 3, 16), ("Cl", 3, 17), ("Ar", 3, 18),
            ("K", 4, 1), ("Ca", 4, 2), ("Sc", 4, 3), ("Ti", 4, 4), ("V", 4, 5), ("Cr", 4, 6), ("Mn", 4, 7), ("Fe", 4, 8), ("Co", 4, 9), ("Ni", 4, 10), ("Cu", 4, 11), ("Zn", 4, 12), ("Ga", 4, 13), ("Ge", 4, 14), ("As", 4, 15), ("Se", 4, 16), ("Br", 4, 17), ("Kr", 4, 18),
            ("Rb", 5, 1), ("Sr", 5, 2), ("Y", 5, 3), ("Zr", 5, 4), ("Nb", 5, 5), ("Mo", 5, 6), ("Tc", 5, 7), ("Ru", 5, 8), ("Rh", 5, 9), ("Pd", 5, 10), ("Ag", 5, 11), ("Cd", 5, 12), ("In", 5, 13), ("Sn", 5, 14), ("Sb", 5, 15), ("Te", 5, 16), ("I", 5, 17), ("Xe", 5, 18),
            ("Cs", 6, 1), ("Ba", 6, 2), ("La", 6, 3),
            ("Hf", 6, 4), ("Ta", 6, 5), ("W", 6, 6), ("Re", 6, 7), ("Os", 6, 8), ("Ir", 6, 9), ("Pt", 6, 10), ("Au", 6, 11), ("Hg", 6, 12), ("Tl", 6, 13), ("Pb", 6, 14), ("Bi", 6, 15), ("Po", 6, 16), ("At", 6, 17), ("Rn", 6, 18),
            
            # Lanthanides (La-Lu Row 8)
            ("Ce", 8, 4), ("Pr", 8, 5), ("Nd", 8, 6), ("Pm", 8, 7), ("Sm", 8, 8), ("Eu", 8, 9), ("Gd", 8, 10), ("Tb", 8, 11), ("Dy", 8, 12), ("Ho", 8, 13), ("Er", 8, 14), ("Tm", 8, 15), ("Yb", 8, 16), ("Lu", 8, 17)
        ]
        
        self.buttons = {}
        for symbol, r, c in table_elements:
            btn = QPushButton(symbol)
            btn.setFixedSize(45, 45)
            
            # Highlight if the element is available in our database
            in_db = symbol in self.database
            if in_db:
                if symbol in self.selected_elements:
                    btn.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; border-radius: 4px; border: 2px solid #047857;")
                else:
                    btn.setStyleSheet("background-color: #3b82f6; color: white; font-weight: bold; border-radius: 4px;")
            else:
                btn.setStyleSheet("background-color: #27272a; color: #52525b; border-radius: 4px; border: 1px solid #3f3f46;")
                btn.setEnabled(False)
                
            btn.clicked.connect(lambda checked, s=symbol: self.toggle_element(s))
            grid.addWidget(btn, r, c)
            self.buttons[symbol] = btn
            
        layout.addLayout(grid)
        
        # OK and Cancel buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def toggle_element(self, symbol):
        if symbol in self.selected_elements:
            self.selected_elements.remove(symbol)
            self.buttons[symbol].setStyleSheet("background-color: #3b82f6; color: white; font-weight: bold; border-radius: 4px;")
        else:
            self.selected_elements.append(symbol)
            self.buttons[symbol].setStyleSheet("background-color: #10b981; color: white; font-weight: bold; border-radius: 4px; border: 2px solid #047857;")

    def get_selected(self):
        return self.selected_elements

# ---------------------------------------------------------
# Dynamic Multiprocessing fit functions
# ---------------------------------------------------------
def worker_calc_peaks_dynamic(x, intensity, peaks, ints, w, e_inc=None, line_edges=None):
    term = np.zeros_like(x)
    
    # Physics-based thresholding: Zero out lines whose excitation edge is above the current incident energy.
    # Re-normalize/re-scale the remaining active lines so the relative intensities are physically correct.
    active_sum = 0.0
    db_sum = 0.0
    
    # If e_inc is not provided, assume fully excited (i.e. all lines visible)
    for p, val, name in zip(peaks, ints, ["Ka1", "Ka2", "Kb1", "Kb3", "La1", "La2", "Lb1", "Lb2", "Lg1", "Ll"][:len(peaks)]):
        db_sum += val
        line_edge = None
        if line_edges is not None and name in line_edges:
            line_edge = line_edges[name]
        else:
            # Fallback physics calculation: edge is generally slightly higher than transition energy
            line_edge = p * 1000.0 * 1.05  # convert keV back to eV and apply 5% buffer
            
        if e_inc is None or e_inc >= line_edge:
            active_sum += val
            term += val * np.exp(-((x - p) ** 2) / (2 * (w ** 2)))
            
    if active_sum == 0.0:
        return np.zeros_like(x)
        
    # Scale active peaks dynamically to preserve relative scaling factors
    scale_factor = db_sum / active_sum
    return intensity * term * scale_factor

def worker_calc_scatter_opt(x, mu, amp, sigma, sf, tf, b, b0, b1, ls, cut, rd):
    bg = b0 + b1 * x
    num = np.maximum(mu - x, 0)
    
    inv_rd = 1.0 / rd
    exp_arg = np.clip((x - cut) * inv_rd, -50.0, 50.0)
    denom = 1.0 + np.exp(exp_arg)
    left_term = -ls * num / denom
    
    inv_sigma = 1.0 / sigma
    arg = (x - mu) * inv_sigma
    g_peak = np.exp(-0.5 * arg * arg)
    
    arg_sqrt2 = arg * INV_SQRT2
    step_comp = sf * erfc(arg_sqrt2)
    
    inv_b = 1.0 / b
    tail_arg1 = arg * inv_b
    tail_arg2 = arg_sqrt2 + inv_b
    
    tail_comp = np.zeros_like(x)
    pos_mask = tail_arg2 >= 0
    neg_mask = ~pos_mask
    if np.any(pos_mask):
        tail_comp[pos_mask] = tf * np.exp(tail_arg1[pos_mask] - tail_arg2[pos_mask]*tail_arg2[pos_mask]) * erfcx(tail_arg2[pos_mask])
    if np.any(neg_mask):
        tail_comp[neg_mask] = tf * np.exp(tail_arg1[neg_mask]) * erfc(tail_arg2[neg_mask])
        
    return bg + left_term + amp * (g_peak + step_comp + tail_comp)

def worker_model_func_dynamic(params, x, mu_inc, element_list, shapes_dict, init_bg_vals):
    n_elems = len(element_list)
    elem_ints = params[0:n_elems]
    
    amp, mu, b0, b1 = params[n_elems:n_elems+4]
    sigma, sf, tf, ls = init_bg_vals
    b = 1.04
    cut = 5.8
    rd = 1.0
    
    total = np.zeros_like(x)
    for idx, name in enumerate(element_list):
        total += elem_ints[idx] * shapes_dict[name]
        
    scatt_c = worker_calc_scatter_opt(x, mu, amp, sigma, sf, tf, b, b0, b1, ls, cut, rd)
    return total + scatt_c

def worker_residuals_dynamic(params, x, y, y_err, mu_inc, element_list, shapes_dict, init_bg_vals, edge_energies):
    fit = worker_model_func_dynamic(params, x, mu_inc, element_list, shapes_dict, init_bg_vals)
    res = (fit - y) / y_err
    
    # Penalties for elements near/above absorption edge
    penalties = []
    for idx, name in enumerate(element_list):
        edge = edge_energies.get(name, 0.0) / 1000.0
        if edge > 0:
            if edge - 0.020 <= mu_inc < edge:
                fraction = (edge - mu_inc) / 0.020
                penalties.append(0.08 * fraction * params[idx])
                
    if len(penalties) > 0:
        res = np.concatenate([res, np.array(penalties)])
    return res

def fit_slices_chunk_dynamic(s_indices, data_2d_chunk, energy_axis_chunk, mca_energy_axis, fit_min_kev, fit_max_kev, init_bg_params, w_peaks, tol, element_list, element_peaks_ints, edge_energies):
    import numpy as np
    from scipy.optimize import least_squares
    
    sigma_scatt, step_frac, tail_frac, beta_val, b0, b1, leftslope, cutoff, rounding = init_bg_params
    
    n_elems = len(element_list)
    mu_start = energy_axis_chunk[0] / 1000.0
    
    # Initial guess for params
    p_prev = [50.0] * n_elems + [120.0, mu_start, b0, b1]
    
    x_mca = mca_energy_axis / 1000.0
    fit_mask = (x_mca >= fit_min_kev) & (x_mca <= fit_max_kev)
    x_fit = x_mca[fit_mask]
    
    chunk_fitted_profiles = {name: [] for name in element_list}
    chunk_fitted_params = []
    
    bounds_min = [0.0] * n_elems + [0.0, mu_start - 0.1, -1.0, -0.1]
    bounds_max = [5000.0] * n_elems + [5000.0, mu_start + 0.1, 5.0, 0.1]
    
    for idx, s_idx in enumerate(s_indices):
        e_inc = energy_axis_chunk[idx]
        mu_inc = e_inc / 1000.0
        
        y_data = data_2d_chunk[idx] * 1e7
        y_fit = y_data[fit_mask]
        y_fit_err = np.sqrt(np.maximum(y_fit, 1.0))
        
        p_prev[n_elems + 1] = mu_inc
        bounds_min[n_elems + 1] = mu_inc - 0.1
        bounds_max[n_elems + 1] = mu_inc + 0.1
        
        for idx_p in range(len(p_prev)):
            p_prev[idx_p] = np.clip(p_prev[idx_p], bounds_min[idx_p], bounds_max[idx_p])
            
        # Re-compute peak shape vectors for each element dynamically based on current slice incident energy
        shapes_dict = {}
        for name in element_list:
            p_arr = np.array(element_peaks_ints[name]["peaks"]) / 1000.0
            i_arr = np.array(element_peaks_ints[name]["intensities"])
            
            # Retrieve line edges or compute via fallback
            line_edges = element_peaks_ints[name].get("line_edges", {})
            shapes_dict[name] = worker_calc_peaks_dynamic(x_fit, 1.0, p_arr, i_arr, w_peaks, e_inc=e_inc, line_edges=line_edges)
            
        try:
            init_bg_vals = (sigma_scatt, step_frac, tail_frac, leftslope)
            res = least_squares(
                worker_residuals_dynamic, p_prev, bounds=(bounds_min, bounds_max),
                ftol=tol, xtol=tol, gtol=tol,
                args=(x_fit, y_fit, y_fit_err, mu_inc, element_list, shapes_dict, init_bg_vals, edge_energies)
            )
            if res.success:
                # Clamp element intensities below their excitation edges
                for elem_idx, name in enumerate(element_list):
                    edge = edge_energies.get(name, 0.0)
                    if e_inc < edge - 20.0:
                        res.x[elem_idx] = 0.0
                p_prev = list(res.x)
                chunk_fitted_params.append(res.x)
                for elem_idx, name in enumerate(element_list):
                    chunk_fitted_profiles[name].append(res.x[elem_idx] / 1e7)
            else:
                chunk_fitted_params.append(p_prev)
                for name in element_list:
                    chunk_fitted_profiles[name].append(0.0)
        except Exception:
            chunk_fitted_params.append(p_prev)
            for name in element_list:
                chunk_fitted_profiles[name].append(0.0)
                
    return {
        "s_indices": s_indices,
        "fitted_profiles": {name: np.array(vals) for name, vals in chunk_fitted_profiles.items()},
        "fitted_params": np.array(chunk_fitted_params)
    }

class DynamicFitWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    
    def __init__(self, data_2d, energy_axis, mca_energy_axis, element_list, element_peaks_ints, edge_energies, fit_min_kev=2.0, fit_max_kev=10.0, init_bg_params=None, w_peaks=0.08, tol=1e-6):
        super().__init__()
        self.data_2d = data_2d
        self.energy_axis = energy_axis
        self.mca_energy_axis = mca_energy_axis
        self.element_list = element_list
        self.element_peaks_ints = element_peaks_ints
        self.edge_energies = edge_energies
        self.fit_min_kev = fit_min_kev
        self.fit_max_kev = fit_max_kev
        self.init_bg_params = init_bg_params
        self.w_peaks = w_peaks
        self.tol = tol
        self.is_running = True
        
    def run(self):
        n_slices = len(self.energy_axis)
        chunk_size = 100
        chunks = []
        for i in range(0, n_slices, chunk_size):
            chunks.append(list(range(i, min(i + chunk_size, n_slices))))
            
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        pool = multiprocessing.Pool(processes=num_processes)
        
        results_objs = []
        for chunk in chunks:
            data_chunk = self.data_2d[chunk]
            energy_chunk = self.energy_axis[chunk]
            res_obj = pool.apply_async(
                fit_slices_chunk_dynamic,
                args=(chunk, data_chunk, energy_chunk, self.mca_energy_axis,
                      self.fit_min_kev, self.fit_max_kev, self.init_bg_params,
                      self.w_peaks, self.tol, self.element_list, self.element_peaks_ints, self.edge_energies)
            )
            results_objs.append((chunk, res_obj))
            
        completed_slices = 0
        finished_chunks = [False] * len(results_objs)
        while self.is_running and completed_slices < n_slices:
            for idx, (chunk, res_obj) in enumerate(results_objs):
                if not finished_chunks[idx] and res_obj.ready():
                    finished_chunks[idx] = True
                    completed_slices += len(chunk)
                    prog = int(completed_slices * 100 / n_slices)
                    self.progress.emit(prog)
            self.msleep(100)
            
        if not self.is_running:
            pool.terminate()
            pool.join()
            self.finished.emit({"stopped": True})
            return
            
        pool.close()
        pool.join()
        
        fitted_profiles = {name: np.zeros(n_slices) for name in self.element_list}
        fitted_params = np.zeros((n_slices, len(self.element_list) + 4))
        
        for chunk, res_obj in results_objs:
            res_dict = res_obj.get()
            s_indices = res_dict["s_indices"]
            fitted_params[s_indices] = res_dict["fitted_params"]
            for name in self.element_list:
                fitted_profiles[name][s_indices] = res_dict["fitted_profiles"][name]
                
        results = {
            "fitted_profiles": fitted_profiles,
            "fitted_params": fitted_params
        }
        self.finished.emit(results)
        
    def stop(self):
        self.is_running = False

# ---------------------------------------------------------
# DraggableVLine
# ---------------------------------------------------------
class DraggableVLine:
    def __init__(self, line, handle, callback):
        self.line = line
        self.handle = handle
        self.callback = callback
        self.press = None
        self.canvas = line.figure.canvas
        self.cidpress = self.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
    def on_press(self, event):
        if event.inaxes != self.line.axes: 
            return
        tb = getattr(self.canvas, 'toolbar', None)
        if tb is not None and tb.mode != "":
            return
        contains_line, _ = self.line.contains(event)
        contains_handle, _ = self.handle.contains(event)
        if not (contains_line or contains_handle): 
            return
        self.press = self.line.get_xdata()[0], event.xdata
        
    def on_motion(self, event):
        if self.press is None: 
            return
        if event.inaxes != self.line.axes: 
            return
        x0, xpress = self.press
        dx = event.xdata - xpress
        new_x = x0 + dx
        
        new_x = max(2000.0, min(10000.0, new_x))
        self.line.set_xdata([new_x, new_x])
        self.handle.set_xdata([new_x])
        self.canvas.draw_idle()
        self.callback(new_x)
        
    def on_release(self, event):
        self.press = None
        
    def disconnect(self):
        self.canvas.mpl_disconnect(self.cidpress)
        self.canvas.mpl_disconnect(self.cidrelease)
        self.canvas.mpl_disconnect(self.cidmotion)

# ---------------------------------------------------------
# QToggleSwitch
# ---------------------------------------------------------
class QToggleSwitch(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._track_color_on = QColor("#6366f1")
        self._track_color_off = QColor("#d4d4d8")
        self._thumb_color = QColor("#ffffff")
        self._thumb_position = 1.0  # ON by default
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
        track_rect = QRectF(0, 2, self.width(), self.height() - 4)
        color = self._track_color_on if self.isChecked() else self._track_color_off
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(track_rect, 10, 10)
        
        thumb_size = self.height() - 6
        x_min = 3
        x_max = self.width() - thumb_size - 3
        x = x_min + self._thumb_position * (x_max - x_min)
        thumb_rect = QRectF(x, 3, thumb_size, thumb_size)
        painter.setBrush(QBrush(self._thumb_color))
        painter.drawEllipse(thumb_rect)
        
    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        target = 1.0 if self.isChecked() else 0.0
        self.animation.stop()
        self.animation.setStartValue(self._thumb_position)
        self.animation.setEndValue(target)
        self.animation.start()

# ---------------------------------------------------------
# XASExplorerGUI main class
# ---------------------------------------------------------
class XASExplorerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BM28 XMaS - XAS/EXAFS Quick Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Load local database
        self.db = {}
        db_path = os.path.join(os.path.dirname(__file__), "emission_lines.json")
        if os.path.exists(db_path):
            try:
                with open(db_path, "r") as f:
                    self.db = json.load(f)
            except Exception as e:
                print(f"Error loading database: {e}")
                
        # Active selected elements
        self.active_elements = []
        self.element_peaks_ints = {}
        self.element_edges = {}
        
        # Data variables
        self.data_2d = None
        self.energy_axis = None
        self.mca_energy_axis = None
        self.standard_error = None
        self.integrated_xas = None
        self.integrated_err = None
        
        # Fitting variables
        self.fitted_profiles = {}
        self.fitted_params = None
        self.fit_worker = None
        
        # Background parameters
        self.sigma_scatt = 0.0786
        self.step_frac = 0.0145
        self.tail_frac = 1.470
        self.beta = 1.04
        self.b0 = 0.176
        self.b1 = 0.00739
        self.leftslope = 1.351
        self.cutoff = 5.8
        self.rounding = 1.0
        self.w_peaks = 0.08
        
        # Calibration defaults
        self.gain = 3.866022
        self.offset = -146.330606
        
        # State variables
        self.current_idx = 0
        self.use_log_scale_1d = True
        self.theme = "charcoal"
        
        # ROI defaults
        self.roi_start = 6800.0
        self.roi_end = 7200.0
        
        # Element Indicators for 1D MCA Plot
        self.element_indicator_lines = []
        self.element_indicator_texts = []
        
        # Stylesheets
        self.styles = {
            "charcoal": {
                "qss": """
                    QMainWindow { background-color: #18181b; }
                    QWidget { background-color: #18181b; color: #f4f4f5; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }
                    QFrame { background-color: #27272a; border: 1px solid #3f3f46; border-radius: 6px; }
                    QLabel { border: none; background-color: transparent; }
                    QPushButton { background-color: #3f3f46; color: #f4f4f5; border: 1px solid #52525b; border-radius: 4px; padding: 6px 12px; font-weight: bold; }
                    QPushButton:hover { background-color: #52525b; }
                    QPushButton:pressed { background-color: #6366f1; }
                    QPushButton#btn_accent { background-color: #6366f1; border: 1px solid #4f46e5; }
                    QPushButton#btn_accent:hover { background-color: #4f46e5; }
                    QSlider::groove:horizontal { border: 1px solid #3f3f46; height: 8px; background: #27272a; border-radius: 4px; }
                    QSlider::handle:horizontal { background: #6366f1; border: 1px solid #4f46e5; width: 18px; margin: -5px 0; border-radius: 9px; }
                    QSlider::handle:horizontal:hover { background: #4f46e5; }
                    QDoubleSpinBox { background-color: #18181b; border: 1px solid #3f3f46; border-radius: 4px; padding: 4px; color: #f4f4f5; }
                    QCheckBox { background-color: transparent; }
                    QTabWidget::pane { border: 1px solid #3f3f46; background: #27272a; border-radius: 6px; }
                    QTabBar::tab { background: #18181b; color: #a1a1aa; border: 1px solid #3f3f46; border-bottom: none; padding: 8px 16px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
                    QTabBar::tab:selected { background: #27272a; color: #f4f4f5; border-bottom: 1px solid #27272a; font-weight: bold; }
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
                    QPushButton { background-color: #e4e4e7; color: #18181b; border: 1px solid #d4d4d8; border-radius: 4px; padding: 6px 12px; font-weight: bold; }
                    QPushButton:hover { background-color: #d4d4d8; }
                    QPushButton:pressed { background-color: #3b82f6; color: white; }
                    QPushButton#btn_accent { background-color: #3b82f6; border: 1px solid #2563eb; color: white; }
                    QPushButton#btn_accent:hover { background-color: #2563eb; }
                    QSlider::groove:horizontal { border: 1px solid #d4d4d8; height: 8px; background: #e4e4e7; border-radius: 4px; }
                    QSlider::handle:horizontal { background: #3b82f6; border: 1px solid #2563eb; width: 18px; margin: -5px 0; border-radius: 9px; }
                    QSlider::handle:horizontal:hover { background: #2563eb; }
                    QDoubleSpinBox { background-color: #ffffff; border: 1px solid #d4d4d8; border-radius: 4px; padding: 4px; color: #18181b; }
                    QCheckBox { background-color: transparent; }
                    QTabWidget::pane { border: 1px solid #d4d4d8; background: #ffffff; border-radius: 6px; }
                    QTabBar::tab { background: #f4f4f5; color: #71717a; border: 1px solid #d4d4d8; border-bottom: none; padding: 8px 16px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
                    QTabBar::tab:selected { background: #ffffff; color: #18181b; border-bottom: 1px solid #ffffff; font-weight: bold; }
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
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # ==========================================
        # LEFT PANEL
        # ==========================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Load & Theme Card
        load_card = QFrame()
        load_card_layout = QVBoxLayout(load_card)
        self.lbl_file_path = QLabel("Dataset: No file loaded")
        self.lbl_file_path.setStyleSheet("color: #71717a; font-style: italic;")
        self.lbl_file_path.setWordWrap(True)
        
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("Load 2D NumPy")
        btn_load.setObjectName("btn_accent")
        btn_load.clicked.connect(self.on_load_clicked)
        btn_layout.addWidget(btn_load)
        
        toggle_layout = QHBoxLayout()
        toggle_layout.addWidget(QLabel("Theme Switch (Light / Dark):"))
        self.switch_theme = QToggleSwitch()
        self.switch_theme.setChecked(True)
        self.switch_theme.toggled.connect(self.on_theme_switch_toggled)
        toggle_layout.addWidget(self.switch_theme)
        
        load_card_layout.addWidget(self.lbl_file_path)
        load_card_layout.addLayout(btn_layout)
        load_card_layout.addLayout(toggle_layout)
        left_layout.addWidget(load_card)
        
        # Energy Sweeper Card
        controls_card = QFrame()
        controls_layout = QVBoxLayout(controls_card)
        lbl_control_title = QLabel("Sweeper Controls")
        lbl_control_title.setStyleSheet("font-weight: bold;")
        
        sweeper_row = QHBoxLayout()
        self.spin_energy = QDoubleSpinBox()
        self.spin_energy.setRange(0, 15000)
        self.spin_energy.setValue(0.0)
        self.spin_energy.setDecimals(2)
        self.spin_energy.setSuffix(" eV")
        self.spin_energy.setFixedWidth(110)
        self.spin_energy.valueChanged.connect(self.on_spin_changed)
        
        self.lbl_row_idx = QLabel("Row Index: 0")
        self.lbl_row_idx.setStyleSheet("font-family: monospace; font-size: 13px; font-weight: bold;")
        
        sweeper_row.addWidget(QLabel("Energy:"))
        sweeper_row.addWidget(self.spin_energy)
        sweeper_row.addWidget(self.lbl_row_idx)
        
        self.slider_energy = QSlider(Qt.Horizontal)
        self.slider_energy.setRange(0, 100)
        self.slider_energy.setValue(0)
        self.slider_energy.valueChanged.connect(self.on_slider_changed)
        
        controls_layout.addWidget(lbl_control_title)
        controls_layout.addLayout(sweeper_row)
        controls_layout.addWidget(self.slider_energy)
        left_layout.addWidget(controls_card)
        
        # ROI Card
        roi_card = QFrame()
        roi_layout = QVBoxLayout(roi_card)
        lbl_roi_title = QLabel("EXAFS ROI Integration Tool")
        lbl_roi_title.setStyleSheet("font-weight: bold; color: #3b82f6;")
        
        roi_inputs = QHBoxLayout()
        self.spin_roi_start = QDoubleSpinBox()
        self.spin_roi_start.setRange(2000.0, 10000.0)
        self.spin_roi_start.setValue(self.roi_start)
        self.spin_roi_start.setSuffix(" eV")
        self.spin_roi_start.setDecimals(1)
        self.spin_roi_start.valueChanged.connect(self.on_roi_bounds_changed)
        
        self.spin_roi_end = QDoubleSpinBox()
        self.spin_roi_end.setRange(2000.0, 10000.0)
        self.spin_roi_end.setValue(self.roi_end)
        self.spin_roi_end.setSuffix(" eV")
        self.spin_roi_end.setDecimals(1)
        self.spin_roi_end.valueChanged.connect(self.on_roi_bounds_changed)
        
        roi_inputs.addWidget(QLabel("Start:"))
        roi_inputs.addWidget(self.spin_roi_start)
        roi_inputs.addWidget(QLabel("End:"))
        roi_inputs.addWidget(self.spin_roi_end)
        
        roi_layout.addWidget(lbl_roi_title)
        roi_layout.addLayout(roi_inputs)
        left_layout.addWidget(roi_card)
        
        # Heatmap Card
        heatmap_card = QFrame()
        heatmap_layout = QVBoxLayout(heatmap_card)
        lbl_heatmap_title = QLabel("2D Heatmap (Click to slice)")
        lbl_heatmap_title.setStyleSheet("font-weight: bold; color: #71717a;")
        
        self.heatmap_fig = Figure(facecolor='#27272a', edgecolor='none')
        self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
        self.heatmap_canvas.mpl_connect('button_press_event', self.on_heatmap_click)
        self.heatmap_toolbar = NavigationToolbar(self.heatmap_canvas, self)
        
        heatmap_layout.addWidget(lbl_heatmap_title)
        heatmap_layout.addWidget(self.heatmap_toolbar)
        heatmap_layout.addWidget(self.heatmap_canvas)
        left_layout.addWidget(heatmap_card)
        
        # ==========================================
        # RIGHT TABS PANEL
        # ==========================================
        self.tabs = QTabWidget()
        
        # Tab 1: 1D MCA Spectrum
        tab_mca = QWidget()
        tab_mca_layout = QVBoxLayout(tab_mca)
        mca_options = QHBoxLayout()
        lbl_mca_title = QLabel("1D Spectral Emission Profile")
        lbl_mca_title.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.chk_log_1d = QCheckBox("Log Scale (1D)")
        self.chk_log_1d.setChecked(True)
        self.chk_log_1d.stateChanged.connect(self.on_log_1d_toggled)
        
        self.btn_element_picker = QPushButton("Select Elements")
        self.btn_element_picker.clicked.connect(self.open_element_picker)
        self.btn_element_picker.setObjectName("btn_accent")
        
        self.btn_save_spectrum = QPushButton("Save Spectrum")
        self.btn_save_spectrum.clicked.connect(self.on_save_spectrum_clicked)
        
        mca_options.addWidget(lbl_mca_title)
        mca_options.addStretch()
        mca_options.addWidget(self.btn_element_picker)
        mca_options.addWidget(self.chk_log_1d)
        mca_options.addWidget(self.btn_save_spectrum)
        
        self.mca_fig = Figure(facecolor='#27272a', edgecolor='none')
        self.mca_canvas = FigureCanvas(self.mca_fig)
        self.mca_ax = self.mca_fig.add_subplot(111, facecolor='#18181b')
        self.mca_toolbar = NavigationToolbar(self.mca_canvas, self)
        
        self.spectrum_line, = self.mca_ax.plot([], [], color='#6366f1', linewidth=1.5, label='Normalized Spectrum')
        self.incident_line = self.mca_ax.axvline(0, color='red', linestyle=':', linewidth=1.5, label='Incident Energy')
        self.roi_start_line = self.mca_ax.axvline(self.roi_start, color='#10b981', linestyle='-.', alpha=0.8, linewidth=2, picker=20)
        self.roi_end_line = self.mca_ax.axvline(self.roi_end, color='#10b981', linestyle='-.', alpha=0.8, linewidth=2, picker=20)
        self.roi_start_handle, = self.mca_ax.plot([self.roi_start], [1.0], color='#10b981', marker='v', markersize=12,
                                                  transform=self.mca_ax.get_xaxis_transform(), clip_on=False, picker=20, zorder=5)
        self.roi_end_handle, = self.mca_ax.plot([self.roi_end], [1.0], color='#10b981', marker='v', markersize=12,
                                                transform=self.mca_ax.get_xaxis_transform(), clip_on=False, picker=20, zorder=5)
        
        self.roi_start_text = self.mca_ax.text(self.roi_start, 0.70, f"ROI Start\n({self.roi_start:.0f} eV)",
                                               color='#10b981', rotation=90, transform=self.mca_ax.get_xaxis_transform(),
                                               verticalalignment='center', horizontalalignment='right', fontsize=11, fontweight='bold',
                                               bbox=dict(facecolor='#27272a', alpha=0.8, edgecolor='none', pad=2))
        self.roi_end_text = self.mca_ax.text(self.roi_end, 0.70, f"ROI End\n({self.roi_end:.0f} eV)",
                                             color='#10b981', rotation=90, transform=self.mca_ax.get_xaxis_transform(),
                                             verticalalignment='center', horizontalalignment='left', fontsize=11, fontweight='bold',
                                             bbox=dict(facecolor='#27272a', alpha=0.8, edgecolor='none', pad=2))
        self.roi_span = self.mca_ax.axvspan(self.roi_start, self.roi_end, color='#10b981', alpha=0.1, label='Integration ROI')
        self.drag_start = DraggableVLine(self.roi_start_line, self.roi_start_handle, self.on_drag_roi_start)
        self.drag_end = DraggableVLine(self.roi_end_line, self.roi_end_handle, self.on_drag_roi_end)
        
        tab_mca_layout.addLayout(mca_options)
        tab_mca_layout.addWidget(self.mca_toolbar)
        tab_mca_layout.addWidget(self.mca_canvas)
        self.tabs.addTab(tab_mca, "1D Spectrum (MCA)")
        
        # Tab 2: Integrated XAS/EXAFS
        tab_xas = QWidget()
        tab_xas_layout = QVBoxLayout(tab_xas)
        k_controls_card = QFrame()
        k_controls_layout = QHBoxLayout(k_controls_card)
        k_controls_layout.setContentsMargins(6, 6, 6, 6)
        
        k_controls_layout.addWidget(QLabel("E0 (eV):"))
        self.spin_e0 = QDoubleSpinBox()
        self.spin_e0.setRange(2000.0, 15000.0)
        self.spin_e0.setValue(7514.0)
        self.spin_e0.setDecimals(1)
        self.spin_e0.setFixedWidth(85)
        self.spin_e0.valueChanged.connect(self.on_k_params_changed)
        k_controls_layout.addWidget(self.spin_e0)
        
        self.btn_auto_e0 = QPushButton("Auto E0")
        self.btn_auto_e0.clicked.connect(self.auto_detect_e0)
        k_controls_layout.addWidget(self.btn_auto_e0)
        
        k_controls_layout.addWidget(QLabel("  E Min (eV):"))
        self.spin_k_min = QDoubleSpinBox()
        self.spin_k_min.setRange(2000.0, 15000.0)
        self.spin_k_min.setValue(7564.0)
        self.spin_k_min.setDecimals(1)
        self.spin_k_min.setFixedWidth(85)
        self.spin_k_min.valueChanged.connect(self.on_k_params_changed)
        k_controls_layout.addWidget(self.spin_k_min)
        
        k_controls_layout.addWidget(QLabel("  E Max (eV):"))
        self.spin_k_max = QDoubleSpinBox()
        self.spin_k_max.setRange(2000.0, 15000.0)
        self.spin_k_max.setValue(8240.0)
        self.spin_k_max.setDecimals(1)
        self.spin_k_max.setFixedWidth(85)
        self.spin_k_max.valueChanged.connect(self.on_k_params_changed)
        k_controls_layout.addWidget(self.spin_k_max)
        
        k_controls_layout.addWidget(QLabel("  Weight:"))
        self.combo_weight = QComboBox()
        self.combo_weight.addItems(["k", "k²", "k³"])
        self.combo_weight.currentIndexChanged.connect(self.on_k_params_changed)
        k_controls_layout.addWidget(self.combo_weight)
        
        self.chk_rebin = QCheckBox("Rebin")
        self.chk_rebin.stateChanged.connect(self.on_k_params_changed)
        k_controls_layout.addWidget(self.chk_rebin)
        
        self.chk_subtract_bg = QCheckBox("Subtract BG")
        self.chk_subtract_bg.setChecked(True)
        self.chk_subtract_bg.stateChanged.connect(self.on_k_params_changed)
        k_controls_layout.addWidget(self.chk_subtract_bg)
        
        k_controls_layout.addWidget(QLabel("Group Size:"))
        self.spin_rebin_size = QSpinBox()
        self.spin_rebin_size.setRange(2, 50)
        self.spin_rebin_size.setValue(5)
        self.spin_rebin_size.valueChanged.connect(self.on_k_params_changed)
        k_controls_layout.addWidget(self.spin_rebin_size)
        
        tab_xas_splitter = QSplitter(Qt.Vertical)
        k_container = QWidget()
        k_container_layout = QVBoxLayout(k_container)
        k_container_layout.setContentsMargins(0, 0, 0, 0)
        self.k_fig = Figure(facecolor='#27272a', edgecolor='none')
        self.k_canvas = FigureCanvas(self.k_fig)
        self.k_ax = self.k_fig.add_subplot(111, facecolor='#18181b')
        self.k_toolbar = NavigationToolbar(self.k_canvas, self)
        k_container_layout.addWidget(self.k_toolbar)
        k_container_layout.addWidget(self.k_canvas)
        
        xas_container = QWidget()
        xas_container_layout = QVBoxLayout(xas_container)
        xas_container_layout.setContentsMargins(0, 0, 0, 0)
        self.xas_fig = Figure(facecolor='#27272a', edgecolor='none')
        self.xas_canvas = FigureCanvas(self.xas_fig)
        self.xas_ax = self.xas_fig.add_subplot(111, facecolor='#18181b')
        self.xas_toolbar = NavigationToolbar(self.xas_canvas, self)
        xas_container_layout.addWidget(self.xas_toolbar)
        xas_container_layout.addWidget(self.xas_canvas)
        
        self.k_line, = self.k_ax.plot([], [], color='#6366f1', linewidth=1.5, label='k-Weighted Data')
        self.xas_line, = self.xas_ax.plot([], [], color='#10b981', linewidth=2, label='Raw XAS Curve')
        self.xas_slice_line = self.xas_ax.axvline(0, color='white', linestyle='-', linewidth=1.5, label='Selected Energy')
        self.xas_edge_lines = []
        self.xas_edge_texts = []
        self.xas_error_fill = None
        self.k_error_fill = None
        
        tab_xas_splitter.addWidget(k_container)
        tab_xas_splitter.addWidget(xas_container)
        tab_xas_splitter.setSizes([350, 350])
        tab_xas_layout.addWidget(k_controls_card)
        tab_xas_layout.addWidget(tab_xas_splitter)
        self.tabs.addTab(tab_xas, "Integrated XAS/EXAFS")
        
        # Tab 3: Peak Fitting EXAFS
        tab_fit = QWidget()
        tab_fit_layout = QVBoxLayout(tab_fit)
        fit_controls_card = QFrame()
        fit_controls_vlayout = QVBoxLayout(fit_controls_card)
        fit_controls_vlayout.setContentsMargins(6, 6, 6, 6)
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Element:"))
        self.fit_combo_element = QComboBox()
        self.fit_combo_element.currentIndexChanged.connect(self.on_fit_element_changed)
        row1.addWidget(self.fit_combo_element)
        
        row1.addWidget(QLabel("  Fit Min (keV):"))
        self.fit_spin_range_min = QDoubleSpinBox()
        self.fit_spin_range_min.setRange(2.0, 12.0)
        self.fit_spin_range_min.setValue(5.0)
        self.fit_spin_range_min.setDecimals(2)
        self.fit_spin_range_min.valueChanged.connect(self.update_fit_tab_slice)
        row1.addWidget(self.fit_spin_range_min)
        
        row1.addWidget(QLabel("  Fit Max (keV):"))
        self.fit_spin_range_max = QDoubleSpinBox()
        self.fit_spin_range_max.setRange(2.0, 12.0)
        self.fit_spin_range_max.setValue(8.5)
        self.fit_spin_range_max.setDecimals(2)
        self.fit_spin_range_max.valueChanged.connect(self.update_fit_tab_slice)
        row1.addWidget(self.fit_spin_range_max)
        
        self.btn_run_fit = QPushButton("Run Batch Fit")
        self.btn_run_fit.setObjectName("btn_accent")
        self.btn_run_fit.clicked.connect(self.run_batch_fit)
        row1.addWidget(self.btn_run_fit)
        
        self.btn_fit_export_exafs = QPushButton("Export EXAFS")
        self.btn_fit_export_exafs.clicked.connect(self.export_fit_exafs_data)
        row1.addWidget(self.btn_fit_export_exafs)
        
        self.btn_fit_export_xas = QPushButton("Export XAS")
        self.btn_fit_export_xas.clicked.connect(self.export_fit_xas_data)
        row1.addWidget(self.btn_fit_export_xas)
        
        self.fit_progress = QProgressBar()
        self.fit_progress.setRange(0, 100)
        self.fit_progress.setValue(0)
        self.fit_progress.setFixedHeight(18)
        row1.addWidget(self.fit_progress)
        
        self.fit_lbl_status = QLabel("Status: Not fitted")
        row1.addWidget(self.fit_lbl_status)
        fit_controls_vlayout.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("E0 (eV):"))
        self.fit_spin_e0 = QDoubleSpinBox()
        self.fit_spin_e0.setRange(2000.0, 15000.0)
        self.fit_spin_e0.setValue(7514.0)
        self.fit_spin_e0.setDecimals(1)
        self.fit_spin_e0.valueChanged.connect(self.on_fit_k_params_changed)
        row2.addWidget(self.fit_spin_e0)
        
        self.fit_btn_auto_e0 = QPushButton("Auto E0")
        self.fit_btn_auto_e0.clicked.connect(self.auto_detect_fit_e0)
        row2.addWidget(self.fit_btn_auto_e0)
        
        row2.addWidget(QLabel("  E Min (eV):"))
        self.fit_spin_k_min = QDoubleSpinBox()
        self.fit_spin_k_min.setRange(2000.0, 15000.0)
        self.fit_spin_k_min.setValue(7564.0)
        self.fit_spin_k_min.setDecimals(1)
        self.fit_spin_k_min.valueChanged.connect(self.on_fit_k_params_changed)
        row2.addWidget(self.fit_spin_k_min)
        
        row2.addWidget(QLabel("  E Max (eV):"))
        self.fit_spin_k_max = QDoubleSpinBox()
        self.fit_spin_k_max.setRange(2000.0, 15000.0)
        self.fit_spin_k_max.setValue(8240.0)
        self.fit_spin_k_max.setDecimals(1)
        self.fit_spin_k_max.valueChanged.connect(self.on_fit_k_params_changed)
        row2.addWidget(self.fit_spin_k_max)
        
        row2.addWidget(QLabel("  Weight:"))
        self.fit_combo_weight = QComboBox()
        self.fit_combo_weight.addItems(["k", "k²", "k³"])
        self.fit_combo_weight.currentIndexChanged.connect(self.on_fit_k_params_changed)
        row2.addWidget(self.fit_combo_weight)
        
        self.fit_chk_rebin = QCheckBox("Rebin")
        self.fit_chk_rebin.stateChanged.connect(self.on_fit_k_params_changed)
        row2.addWidget(self.fit_chk_rebin)
        
        self.fit_chk_subtract_bg = QCheckBox("Subtract BG")
        self.fit_chk_subtract_bg.setChecked(True)
        self.fit_chk_subtract_bg.stateChanged.connect(self.on_fit_k_params_changed)
        row2.addWidget(self.fit_chk_subtract_bg)
        
        row2.addWidget(QLabel("Group:"))
        self.fit_spin_rebin_size = QSpinBox()
        self.fit_spin_rebin_size.setRange(2, 50)
        self.fit_spin_rebin_size.setValue(5)
        self.fit_spin_rebin_size.valueChanged.connect(self.on_fit_k_params_changed)
        row2.addWidget(self.fit_spin_rebin_size)
        fit_controls_vlayout.addLayout(row2)
        tab_fit_layout.addWidget(fit_controls_card)
        
        tab_fit_splitter = QSplitter(Qt.Vertical)
        fit_k_container = QWidget()
        fit_k_container_layout = QVBoxLayout(fit_k_container)
        fit_k_container_layout.setContentsMargins(0, 0, 0, 0)
        self.fit_k_fig = Figure(facecolor='#27272a', edgecolor='none')
        self.fit_k_canvas = FigureCanvas(self.fit_k_fig)
        self.fit_k_ax = self.fit_k_fig.add_subplot(111, facecolor='#18181b')
        self.fit_k_toolbar = NavigationToolbar(self.fit_k_canvas, self)
        fit_k_container_layout.addWidget(self.fit_k_toolbar)
        fit_k_container_layout.addWidget(self.fit_k_canvas)
        
        fit_xas_container = QWidget()
        fit_xas_container_layout = QVBoxLayout(fit_xas_container)
        fit_xas_container_layout.setContentsMargins(0, 0, 0, 0)
        self.fit_xas_fig = Figure(facecolor='#27272a', edgecolor='none')
        self.fit_xas_canvas = FigureCanvas(self.fit_xas_fig)
        self.fit_xas_ax = self.fit_xas_fig.add_subplot(111, facecolor='#18181b')
        self.fit_xas_toolbar = NavigationToolbar(self.fit_xas_canvas, self)
        fit_xas_container_layout.addWidget(self.fit_xas_toolbar)
        fit_xas_container_layout.addWidget(self.fit_xas_canvas)
        
        fit_mca_container = QWidget()
        fit_mca_container_layout = QVBoxLayout(fit_mca_container)
        fit_mca_container_layout.setContentsMargins(0, 0, 0, 0)
        self.fit_mca_fig = Figure(facecolor='#27272a', edgecolor='none')
        self.fit_mca_canvas = FigureCanvas(self.fit_mca_fig)
        self.fit_mca_ax = self.fit_mca_fig.add_subplot(111, facecolor='#18181b')
        self.fit_mca_toolbar = NavigationToolbar(self.fit_mca_canvas, self)
        fit_mca_container_layout.addWidget(self.fit_mca_toolbar)
        fit_mca_container_layout.addWidget(self.fit_mca_canvas)
        
        self.fit_k_line, = self.fit_k_ax.plot([], [], color='#6366f1', linewidth=1.5, label='k-Weighted Data')
        self.fit_xas_line, = self.fit_xas_ax.plot([], [], color='#10b981', linewidth=2, label='Fitted Elemental XAS')
        self.fit_xas_slice_line = self.fit_xas_ax.axvline(0, color='white', linestyle='-', linewidth=1.5, label='Selected Energy')
        self.fit_mca_raw_line, = self.fit_mca_ax.plot([], [], '.', color='#a1a1aa', markersize=3, label='Measured Data', zorder=1)
        self.fit_mca_total_line, = self.fit_mca_ax.plot([], [], color='#6366f1', linewidth=2, label='Total Fit', zorder=5)
        self.fit_mca_scatt_line, = self.fit_mca_ax.plot([], [], ':', color='#ec4899', linewidth=1.2, label='Scatter Background', zorder=2)
        
        self.fit_mca_elem_lines = {}
        
        self.fit_xas_edge_lines = []
        self.fit_xas_edge_texts = []
        
        tab_fit_splitter.addWidget(fit_k_container)
        tab_fit_splitter.addWidget(fit_xas_container)
        tab_fit_splitter.addWidget(fit_mca_container)
        tab_fit_splitter.setSizes([230, 230, 230])
        tab_fit_layout.addWidget(tab_fit_splitter)
        self.tabs.addTab(tab_fit, "Peak Fitting EXAFS")
        
        # Tab 4: EXAFS Fourier Transform (R-space)
        tab_ft = QWidget()
        tab_ft_layout = QVBoxLayout(tab_ft)
        ft_controls_card = QFrame()
        ft_controls_layout = QHBoxLayout(ft_controls_card)
        ft_controls_layout.setContentsMargins(6, 6, 6, 6)
        
        ft_controls_layout.addWidget(QLabel("Source:"))
        self.ft_combo_source = QComboBox()
        self.ft_combo_source.addItems(["ROI Dataset (Raw XAS)", "Peak-Fitted Dataset"])
        self.ft_combo_source.currentTextChanged.connect(self.on_ft_source_changed)
        ft_controls_layout.addWidget(self.ft_combo_source)
        
        ft_controls_layout.addWidget(QLabel("Element:"))
        self.ft_combo_element = QComboBox()
        self.ft_combo_element.setEnabled(False)
        ft_controls_layout.addWidget(self.ft_combo_element)
        
        def sync_ft_element_to_fit(text):
            idx = self.ft_combo_element.findText(text)
            if idx >= 0:
                self.fit_combo_element.blockSignals(True)
                self.fit_combo_element.setCurrentIndex(idx)
                self.fit_combo_element.blockSignals(False)
                self.auto_detect_fit_e0()
                self.plot_fit_xas_and_exafs()
            self.calculate_fourier_transform()
        self.ft_combo_element.currentTextChanged.connect(sync_ft_element_to_fit)
        
        ft_controls_layout.addWidget(QLabel("k-weight:"))
        self.ft_spin_kweight = QSpinBox()
        self.ft_spin_kweight.setRange(1, 3)
        self.ft_spin_kweight.setValue(2)
        self.ft_spin_kweight.valueChanged.connect(self.calculate_fourier_transform)
        ft_controls_layout.addWidget(self.ft_spin_kweight)
        
        ft_controls_layout.addWidget(QLabel("Window:"))
        self.ft_combo_window = QComboBox()
        self.ft_combo_window.addItems(["Hanning", "Hamming", "Kaiser-Bessel", "None"])
        self.ft_combo_window.currentTextChanged.connect(self.calculate_fourier_transform)
        ft_controls_layout.addWidget(self.ft_combo_window)
        
        ft_controls_layout.addWidget(QLabel("k Min (Å⁻¹):"))
        self.ft_spin_kmin = QDoubleSpinBox()
        self.ft_spin_kmin.setRange(0.0, 15.0)
        self.ft_spin_kmin.setValue(2.0)
        self.ft_spin_kmin.valueChanged.connect(self.calculate_fourier_transform)
        ft_controls_layout.addWidget(self.ft_spin_kmin)
        
        ft_controls_layout.addWidget(QLabel("k Max (Å⁻¹):"))
        self.ft_spin_kmax = QDoubleSpinBox()
        self.ft_spin_kmax.setRange(0.0, 20.0)
        self.ft_spin_kmax.setValue(10.0)
        self.ft_spin_kmax.valueChanged.connect(self.calculate_fourier_transform)
        ft_controls_layout.addWidget(self.ft_spin_kmax)
        
        ft_controls_layout.addWidget(QLabel("dk (Å⁻¹):"))
        self.ft_spin_dk = QDoubleSpinBox()
        self.ft_spin_dk.setRange(0.0, 5.0)
        self.ft_spin_dk.setValue(1.0)
        self.ft_spin_dk.setSingleStep(0.1)
        self.ft_spin_dk.valueChanged.connect(self.calculate_fourier_transform)
        ft_controls_layout.addWidget(self.ft_spin_dk)
        
        ft_controls_layout.addWidget(QLabel("k Step (Å⁻¹):"))
        self.ft_spin_kstep = QDoubleSpinBox()
        self.ft_spin_kstep.setRange(0.01, 0.20)
        self.ft_spin_kstep.setValue(0.05)
        self.ft_spin_kstep.setSingleStep(0.01)
        self.ft_spin_kstep.valueChanged.connect(self.calculate_fourier_transform)
        ft_controls_layout.addWidget(self.ft_spin_kstep)
        
        ft_controls_layout.addWidget(QLabel("R Max (Å):"))
        self.ft_spin_rmax = QDoubleSpinBox()
        self.ft_spin_rmax.setRange(1.0, 10.0)
        self.ft_spin_rmax.setValue(6.0)
        self.ft_spin_rmax.valueChanged.connect(self.calculate_fourier_transform)
        ft_controls_layout.addWidget(self.ft_spin_rmax)
        
        self.btn_run_ft = QPushButton("Calculate FT")
        self.btn_run_ft.clicked.connect(self.calculate_fourier_transform)
        ft_controls_layout.addWidget(self.btn_run_ft)
        
        self.btn_export_exafs = QPushButton("Export EXAFS")
        self.btn_export_exafs.clicked.connect(self.export_exafs_data)
        ft_controls_layout.addWidget(self.btn_export_exafs)
        
        tab_ft_layout.addWidget(ft_controls_card)
        
        self.ft_fig = Figure(facecolor='#27272a', edgecolor='none')
        self.ft_canvas = FigureCanvas(self.ft_fig)
        self.ft_k_ax = self.ft_fig.add_subplot(211, facecolor='#18181b')
        self.ft_r_ax = self.ft_fig.add_subplot(212, facecolor='#18181b')
        self.ft_toolbar = NavigationToolbar(self.ft_canvas, self)
        
        tab_ft_layout.addWidget(self.ft_toolbar)
        tab_ft_layout.addWidget(self.ft_canvas, 1)
        self.tabs.addTab(tab_ft, "EXAFS Fourier Transform (R-space)")
        
        # Tab 5: Calibration & Parameters
        tab_params = QWidget()
        tab_params_layout = QVBoxLayout(tab_params)
        params_card = QFrame()
        params_card_layout = QVBoxLayout(params_card)
        lbl_params_title = QLabel("Calibration & Model Parameters")
        lbl_params_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #6366f1; margin-bottom: 12px;")
        params_card_layout.addWidget(lbl_params_title)
        
        grid = QGridLayout()
        grid.setSpacing(15)
        
        lbl_gain = QLabel("Emission Energy Gain (eV/channel):")
        lbl_gain.setStyleSheet("font-weight: bold;")
        self.lbl_val_gain = QLabel(f"{self.gain:.6f} eV/ch")
        self.lbl_val_gain.setStyleSheet("font-family: monospace; font-size: 14px;")
        grid.addWidget(lbl_gain, 0, 0)
        grid.addWidget(self.lbl_val_gain, 0, 1)
        
        lbl_offset = QLabel("Emission Energy Offset (eV):")
        lbl_offset.setStyleSheet("font-weight: bold;")
        self.lbl_val_offset = QLabel(f"{self.offset:.3f} eV")
        self.lbl_val_offset.setStyleSheet("font-family: monospace; font-size: 14px;")
        grid.addWidget(lbl_offset, 1, 0)
        grid.addWidget(self.lbl_val_offset, 1, 1)
        
        lbl_w_peaks_ev = QLabel("Manual Peak Width (w_peaks, eV):")
        lbl_w_peaks_ev.setStyleSheet("font-weight: bold;")
        self.spin_w_peaks_ev = QDoubleSpinBox()
        self.spin_w_peaks_ev.setRange(10.0, 300.0)
        self.spin_w_peaks_ev.setValue(self.w_peaks * 1000.0)
        self.spin_w_peaks_ev.setDecimals(3)
        grid.addWidget(lbl_w_peaks_ev, 2, 0)
        grid.addWidget(self.spin_w_peaks_ev, 2, 1)
        
        self.btn_apply_w_peaks = QPushButton("Apply Manual Width")
        self.btn_apply_w_peaks.clicked.connect(self.on_manual_w_peaks_changed)
        grid.addWidget(self.btn_apply_w_peaks, 2, 2)
        
        params_card_layout.addLayout(grid)
        tab_params_layout.addWidget(params_card)
        tab_params_layout.addStretch()
        self.tabs.addTab(tab_params, "Calibration & Parameters")
        
        splitter.addWidget(left_widget)
        splitter.addWidget(self.tabs)
        splitter.setSizes([450, 950])
        
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.mca_canvas.mpl_connect('scroll_event', self.on_scroll)
        self.k_canvas.mpl_connect('scroll_event', self.on_scroll)
        self.xas_canvas.mpl_connect('scroll_event', self.on_scroll)
        self.heatmap_canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fit_k_canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fit_xas_canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fit_mca_canvas.mpl_connect('scroll_event', self.on_scroll)
        self.ft_canvas.mpl_connect('scroll_event', self.on_scroll)
        
        self.apply_theme()

    def open_element_picker(self):
        dialog = PeriodicTableDialog(self.db, self.active_elements, self)
        if dialog.exec_() == QDialog.Accepted:
            self.active_elements = dialog.get_selected()
            
            # Re-read peaks and intensities for all selected elements
            self.element_peaks_ints = {}
            self.element_edges = {}
            
            # Clear old dropdown values
            self.fit_combo_element.blockSignals(True)
            self.ft_combo_element.blockSignals(True)
            self.fit_combo_element.clear()
            self.ft_combo_element.clear()
            
            # Clear Matplotlib lines for element contributions on Tab 3
            for line in self.fit_mca_elem_lines.values():
                try: line.remove()
                except: pass
            self.fit_mca_elem_lines = {}
            
            colors = ['#fbbf24', '#ef4444', '#10b981', '#06b6d4', '#ec4899', '#8b5cf6', '#f59e0b']
            
            first_peak = None
            for idx, name in enumerate(self.active_elements):
                elem_data = self.db[name]
                peaks = list(elem_data["lines"].values())
                ints = list(elem_data["intensities"].values())
                
                self.element_peaks_ints[name] = {
                    "peaks": peaks,
                    "intensities": ints,
                    "line_edges": elem_data.get("line_edges", {})
                }
                self.element_edges[name] = elem_data["edge"]
                
                # Check for Ka1 or La1 to use as the primary emission peak to center the ROI around
                p_keys = list(elem_data["lines"].keys())
                for pk in ["Ka1", "La1", "Ka2", "La2"]:
                    if pk in p_keys:
                        if first_peak is None:
                            first_peak = elem_data["lines"][pk]
                        break
                        
                self.fit_combo_element.addItem(name)
                self.ft_combo_element.addItem(name)
                
                # Create a new contribution line for Tab 3 MCA
                color = colors[idx % len(colors)]
                line, = self.fit_mca_ax.plot([], [], '--', color=color, linewidth=1.2, label=f'{name} contribution', zorder=3)
                self.fit_mca_elem_lines[name] = line
                
            self.fit_combo_element.blockSignals(False)
            self.ft_combo_element.blockSignals(False)
            
            # If we found a valid peak energy for the selected elements, coarse-fit the ROI around it
            if first_peak is not None:
                self.roi_start = max(2000.0, first_peak - 150.0)
                self.roi_end = min(10000.0, first_peak + 150.0)
                
                self.spin_roi_start.blockSignals(True)
                self.spin_roi_start.setValue(self.roi_start)
                self.spin_roi_start.blockSignals(False)
                
                self.spin_roi_end.blockSignals(True)
                self.spin_roi_end.setValue(self.roi_end)
                self.spin_roi_end.blockSignals(False)
                
                self.calculate_roi_integration()
                
            # Add legends/re-render
            self.fit_mca_ax.legend(loc='upper right', fontsize=11, facecolor=self.styles[self.theme]["fig_face"], edgecolor=self.styles[self.theme]["spine"], labelcolor=self.styles[self.theme]["text"])
            self.plot_spectrum()
            self.plot_integrated_xas()
            self.plot_fit_xas_and_exafs()
            self.update_fit_tab_slice()
            self.calculate_fourier_transform()

    def on_fit_element_changed(self, idx):
        if idx >= 0:
            self.auto_detect_fit_e0()
            self.plot_fit_xas_and_exafs()
            self.update_fit_tab_slice()

    def on_fit_k_params_changed(self):
        self.plot_fit_xas_and_exafs()

    def on_k_params_changed(self):
        self.plot_integrated_xas()

    def on_manual_w_peaks_changed(self):
        self.w_peaks = self.spin_w_peaks_ev.value() / 1000.0
        self.lbl_val_gain.setText(f"{self.gain:.6f} eV/ch")
        self.lbl_val_offset.setText(f"{self.offset:.3f} eV")
        self.plot_spectrum()
        if self.data_2d is not None:
            self.update_fit_tab_slice()

    def update_params_tab(self):
        self.lbl_val_gain.setText(f"{self.gain:.6f} eV/ch")
        self.lbl_val_offset.setText(f"{self.offset:.3f} eV")

    def auto_detect_e0(self):
        if self.energy_axis is None or self.integrated_xas is None:
            return
        sort_idx = np.argsort(self.energy_axis)
        e_arr = self.energy_axis[sort_idx]
        i_arr = self.integrated_xas[sort_idx]
        dI_dE = np.gradient(i_arr, e_arr)
        inflection_idx = np.argmax(dI_dE)
        detected_e0 = float(e_arr[inflection_idx])
        
        self.spin_e0.blockSignals(True)
        self.spin_e0.setValue(detected_e0)
        self.spin_e0.blockSignals(False)
        
        self.spin_k_min.blockSignals(True)
        self.spin_k_min.setValue(detected_e0 + 50.0)
        self.spin_k_min.blockSignals(False)
        self.plot_integrated_xas()

    def auto_detect_fit_e0(self):
        element = self.fit_combo_element.currentText()
        if not element or self.energy_axis is None or element not in self.fitted_profiles:
            return
        xas_profile = self.fitted_profiles[element]
        sort_idx = np.argsort(self.energy_axis)
        e_arr = self.energy_axis[sort_idx]
        i_arr = xas_profile[sort_idx]
        
        dI_dE = np.gradient(i_arr, e_arr)
        inflection_idx = np.argmax(dI_dE)
        detected_e0 = float(e_arr[inflection_idx])
        
        self.fit_spin_e0.blockSignals(True)
        self.fit_spin_e0.setValue(detected_e0)
        self.fit_spin_e0.blockSignals(False)
        
        self.fit_spin_k_min.blockSignals(True)
        self.fit_spin_k_min.setValue(detected_e0 + 50.0)
        self.fit_spin_k_min.blockSignals(False)
        self.plot_fit_xas_and_exafs()

    def on_load_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Averaged ZAP Dataset", "", "NumPy Files (*.npy *.npz)")
        if file_path:
            self.load_dataset(file_path)

    def auto_load_files(self):
        target_dir = os.getcwd()
        npz_file = os.path.join(target_dir, "averaged_zap.npz")
        if os.path.exists(npz_file):
            self.load_dataset(npz_file)
            return
        data_file = os.path.join(target_dir, "averaged_normalized_zap.npy")
        if os.path.exists(data_file):
            self.load_dataset(data_file)

    def load_dataset(self, file_path):
        try:
            if file_path.endswith('.npz'):
                with np.load(file_path) as dataset:
                    self.data_2d = dataset['average_data']
                    self.energy_axis = dataset['energy_in']
                    self.mca_energy_axis = dataset['energy_out']
                    if 'std_dev' in dataset:
                        self.standard_error = dataset['std_dev']
                    else:
                        self.standard_error = None
                self.lbl_file_path.setText(f"Dataset: {os.path.basename(file_path)}")
            else:
                self.data_2d = np.load(file_path)
                self.lbl_file_path.setText(f"Dataset: {os.path.basename(file_path)}")
                dir_name = os.path.dirname(file_path)
                
                # Check if it has our new suffix pattern
                base, ext = os.path.splitext(file_path)
                if base.endswith('_avg'):
                    base_prefix = base[:-4]
                    energy_path = f"{base_prefix}_energy_in.npy"
                    mca_energy_path = f"{base_prefix}_energy_out.npy"
                    err_path = f"{base_prefix}_std.npy"
                else:
                    energy_path = os.path.join(dir_name, "zap_energy_axis.npy")
                    mca_energy_path = os.path.join(dir_name, "mca_energy_axis.npy")
                    err_path = os.path.join(dir_name, "standard_error_zap.npy")
                    
                if os.path.exists(energy_path):
                    self.energy_axis = np.load(energy_path)
                else:
                    self.energy_axis = np.linspace(8239.62, 7301.02, self.data_2d.shape[0])
                    
                if os.path.exists(mca_energy_path):
                    self.mca_energy_axis = np.load(mca_energy_path)
                else:
                    self.mca_energy_axis = self.gain * np.arange(self.data_2d.shape[1]) + self.offset
                    
                if os.path.exists(err_path):
                    self.standard_error = np.load(err_path)
                else:
                    self.standard_error = None
                
            num_points = len(self.energy_axis)
            self.slider_energy.setRange(0, num_points - 1)
            self.spin_energy.setRange(float(self.energy_axis.min()), float(self.energy_axis.max()))
            
            e_min_axis = float(self.energy_axis.min())
            e_max_axis = float(self.energy_axis.max())
            self.spin_e0.setRange(e_min_axis, e_max_axis)
            self.spin_k_min.setRange(e_min_axis, e_max_axis)
            self.spin_k_max.setRange(e_min_axis, e_max_axis)
            
            self.current_idx = 0
            self.slider_energy.setValue(self.get_slider_val(0))
            self.spin_energy.setValue(float(self.energy_axis[0]))
            
            self.calculate_roi_integration()
            self.auto_detect_e0()
            
            self.mca_ax.set_xlim(2000, 10000)
            self.xas_ax.set_xlim(e_min_axis, e_max_axis)
            
            self.fitted_profiles = {}
            self.fitted_params = None
            self.fit_lbl_status.setText("Status: Not fitted")
            self.fit_progress.setValue(0)
            self.btn_run_fit.setEnabled(True)
            self.btn_run_fit.setText("Run Batch Fit")
            
            self.fit_spin_e0.setRange(e_min_axis, e_max_axis)
            self.fit_spin_k_min.setRange(e_min_axis, e_max_axis)
            self.fit_spin_k_max.setRange(e_min_axis, e_max_axis)
            self.fit_mca_ax.set_xlim(2.0, 10.0)
            self.fit_xas_ax.set_xlim(e_min_axis, e_max_axis)
            
            self.update_params_tab()
            self.plot_heatmap()
            self.plot_spectrum()
            self.plot_fit_xas_and_exafs()
            self.update_fit_tab_slice()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load dataset:\n{str(e)}")

    def get_raw_idx(self, slider_val):
        if self.energy_axis is None: return 0
        if self.energy_axis[0] > self.energy_axis[-1]:
            return len(self.energy_axis) - 1 - slider_val
        return slider_val

    def get_slider_val(self, raw_idx):
        if self.energy_axis is None: return 0
        if self.energy_axis[0] > self.energy_axis[-1]:
            return len(self.energy_axis) - 1 - raw_idx
        return raw_idx

    def calculate_roi_integration(self):
        if self.data_2d is None: return
        idx_crop = np.where((self.mca_energy_axis >= self.roi_start) & (self.mca_energy_axis <= self.roi_end))[0]
        if len(idx_crop) == 0: return
        start_idx, end_idx = idx_crop[0], idx_crop[-1] + 1
        self.integrated_xas = np.sum(self.data_2d[:, start_idx:end_idx], axis=1)
        if self.standard_error is not None:
            self.integrated_err = np.sqrt(np.sum(self.standard_error[:, start_idx:end_idx]**2, axis=1))
        else:
            self.integrated_err = None

    def plot_heatmap(self):
        if self.data_2d is None:
            return
            
        cfg = self.styles[self.theme]
        self.heatmap_fig.clear()
        ax = self.heatmap_fig.add_subplot(111, facecolor=cfg["ax_face"])
        ax.tick_params(colors=cfg["text"], labelsize=10)
        
        idx_crop = np.where((self.mca_energy_axis >= 2000) & (self.mca_energy_axis <= 10000))[0]
        if len(idx_crop) == 0:
            idx_crop = np.arange(len(self.mca_energy_axis))
        c_start, c_end = idx_crop[0], idx_crop[-1] + 1
        
        cropped_data = self.data_2d[:, c_start:c_end]
        cropped_mca_energies = self.mca_energy_axis[c_start:c_end]
        
        log_data = np.log10(np.clip(cropped_data, 1e-6, None))
        
        energy_min = self.energy_axis.min()
        energy_max = self.energy_axis.max()
        extent = [cropped_mca_energies[0], cropped_mca_energies[-1], energy_min, energy_max]
        ax.imshow(log_data, cmap='jet', aspect='auto', interpolation='none', extent=extent)
        
        # Horizontal line showing current incident energy slice
        self.heat_slice_line = ax.axhline(self.energy_axis[self.current_idx], color=cfg["line_color"], linestyle='-', linewidth=1.5)
        
        # Diagonal elastic scatter line
        ax.plot([energy_min, energy_max], [energy_min, energy_max], 
                cfg["line_color"], linestyle='--', alpha=0.5, linewidth=1)
                
        ax.set_xlabel('Emission Energy (eV)', color=cfg["text"], fontsize=13, fontweight='bold')
        ax.set_ylabel('Incident Energy (eV)', color=cfg["text"], fontsize=13, fontweight='bold')
        ax.set_title('2D MCA Emission Heatmap (Log10)', color=cfg["text"], fontsize=14, fontweight='bold')
        
        # Preserve viewport limits
        ax.set_xlim(cropped_mca_energies[0], cropped_mca_energies[-1])
        ax.set_ylim(energy_min, energy_max)
        
        self.heatmap_fig.tight_layout()
        self.heatmap_canvas.draw_idle()

    def plot_spectrum(self):
        if self.data_2d is None: return
        cfg = self.styles[self.theme]
        spectrum = self.data_2d[self.current_idx]
        e_inc = self.energy_axis[self.current_idx]
        
        self.spectrum_line.set_data(self.mca_energy_axis, spectrum)
        self.incident_line.set_xdata([e_inc, e_inc])
        self.incident_line.set_label(f'Incident Energy ({e_inc:.1f} eV)')
        
        self.roi_start_line.set_xdata([self.roi_start, self.roi_start])
        self.roi_end_line.set_xdata([self.roi_end, self.roi_end])
        self.roi_start_handle.set_xdata([self.roi_start])
        self.roi_end_handle.set_xdata([self.roi_end])
        self.roi_start_text.set_x(self.roi_start)
        self.roi_start_text.set_text(f"ROI Start\n({self.roi_start:.0f} eV)")
        self.roi_end_text.set_x(self.roi_end)
        self.roi_end_text.set_text(f"ROI End\n({self.roi_end:.0f} eV)")
        
        # Clear old element indicators
        for line in self.element_indicator_lines:
            try: line.remove()
            except: pass
        self.element_indicator_lines.clear()
        
        for txt in self.element_indicator_texts:
            try: txt.remove()
            except: pass
        self.element_indicator_texts.clear()
        
        # Draw indicators for all user-selected active elements
        colors = ['#f59e0b', '#ef4444', '#10b981', '#06b6d4', '#ec4899', '#8b5cf6', '#3b82f6']
        for elem_idx, name in enumerate(self.active_elements):
            color = colors[elem_idx % len(colors)]
            elem_data = self.db[name]
            
            # Map of line-specific excitation edges (defaulting to overall edge if missing)
            line_edges = elem_data.get("line_edges", {})
            
            # Draw individual emission lines & labels ONLY when their corresponding subshell is excited
            any_excited = False
            for line_name, line_energy in elem_data["lines"].items():
                line_edge_energy = line_edges.get(line_name, elem_data["edge"])
                
                if e_inc >= line_edge_energy:
                    any_excited = True
                    em_line = self.mca_ax.axvline(line_energy, color=color, linestyle='-', alpha=0.8, linewidth=1.5)
                    self.element_indicator_lines.append(em_line)
                    
                    em_txt = self.mca_ax.text(line_energy, 0.15 + (elem_idx * 0.15), f"{name} {line_name}\n({line_energy:.0f} eV)",
                                              color=color, rotation=90, transform=self.mca_ax.get_xaxis_transform(),
                                              verticalalignment='bottom', horizontalalignment='left', fontsize=9,
                                              bbox=dict(facecolor=cfg["fig_face"], alpha=0.7, edgecolor='none', pad=1))
                    self.element_indicator_texts.append(em_txt)
            
            # Draw absorption edge line & label if the incident energy has surpassed it
            # We show the edge corresponding to the active excited subshells (e.g. show L3 edge if L3 lines are excited)
            unique_excited_edges = set(line_edges[ln] for ln in elem_data["lines"].keys() if ln in line_edges and e_inc >= line_edges[ln])
            if not unique_excited_edges and e_inc >= elem_data["edge"]:
                unique_excited_edges.add(elem_data["edge"])
                
            for edge_energy in sorted(unique_excited_edges):
                edge_line = self.mca_ax.axvline(edge_energy, color=color, linestyle='--', alpha=0.6, linewidth=1.2)
                self.element_indicator_lines.append(edge_line)
                
                # Identify which shell edge it corresponds to (e.g. K, L3, L2)
                edge_label = "Edge"
                edges_db = {val: key for key, val in line_edges.items()}
                # Find matching subshell name by looking at database entries
                for ln, val in line_edges.items():
                    if val == edge_energy:
                        # E.g. Ka1/Ka2 -> K-shell; La1/La2 -> L3-shell
                        edge_label = "L3 Edge" if "La" in ln or "Ll" in ln else ("L2 Edge" if "Lb1" in ln or "Lg1" in ln else "K Edge")
                        break
                        
                edge_txt = self.mca_ax.text(edge_energy, 0.90, f"{name} {edge_label}\n({edge_energy:.0f} eV)",
                                            color=color, rotation=90, transform=self.mca_ax.get_xaxis_transform(),
                                            verticalalignment='top', horizontalalignment='right', fontsize=9, fontweight='bold',
                                            bbox=dict(facecolor=cfg["fig_face"], alpha=0.7, edgecolor='none', pad=1))
                self.element_indicator_texts.append(edge_txt)
                
        if hasattr(self, 'roi_span') and self.roi_span is not None:
            try: self.roi_span.remove()
            except: pass
        self.roi_span = self.mca_ax.axvspan(self.roi_start, self.roi_end, color='#10b981', alpha=0.1)
        
        current_scale = self.mca_ax.get_yscale()
        target_scale = 'log' if self.use_log_scale_1d else 'linear'
        if current_scale != target_scale:
            self.mca_ax.set_yscale(target_scale)
            
        self.mca_ax.relim()
        self.mca_ax.autoscale_view(scalex=False, scaley=True)
        self.mca_ax.set_title(f"Spectral Profile at Incident Energy: {e_inc:.2f} eV", color=cfg["text"], fontsize=16)
        self.mca_canvas.draw_idle()

    def plot_integrated_xas(self):
        if self.integrated_xas is None: return
        cfg = self.styles[self.theme]
        self.xas_line.set_data(self.energy_axis, self.integrated_xas)
        e_inc = self.energy_axis[self.current_idx]
        self.xas_slice_line.set_xdata([e_inc, e_inc])
        
        self.xas_ax.relim()
        self.xas_ax.autoscale_view(scalex=False, scaley=True)
        self.xas_ax.set_title(f"Raw Integrated XAS Profile (ROI: {self.roi_start:.0f} - {self.roi_end:.0f} eV)", color=cfg["text"], fontsize=16)
        
        # Plot k-weighted EXAFS
        e0 = self.spin_e0.value()
        k_min = self.spin_k_min.value()
        k_max = self.spin_k_max.value()
        w = self.combo_weight.currentIndex() + 1
        
        if e0 < self.energy_axis.max() and k_min > e0:
            # Filter range and ensure above edge E0
            k_min_real = min(k_min, k_max)
            k_max_real = max(k_min, k_max)
            mask = (self.energy_axis >= k_min_real) & (self.energy_axis <= k_max_real) & (self.energy_axis > e0)
            
            e_sel = self.energy_axis[mask]
            i_sel = self.integrated_xas[mask]
            
            if len(e_sel) > 3:
                # Sort by energy
                sort_idx = np.argsort(e_sel)
                e_sel = e_sel[sort_idx]
                i_sel = i_sel[sort_idx]
                
                k_sel = np.sqrt(0.26246718 * (e_sel - e0))
                
                if hasattr(self, 'chk_subtract_bg') and self.chk_subtract_bg.isChecked() and len(self.energy_axis) > 10:
                    # 1. Fit pre-edge: Fit linear function below E0 - 30 eV
                    pre_mask = self.energy_axis < (e0 - 30.0)
                    if np.sum(pre_mask) < 3:
                        sorted_energies = np.sort(self.energy_axis)
                        threshold = sorted_energies[int(len(sorted_energies) * 0.15)]
                        pre_mask = self.energy_axis <= threshold
                    
                    e_pre = self.energy_axis[pre_mask]
                    i_pre = self.integrated_xas[pre_mask]
                    
                    if len(e_pre) > 1:
                        pre_poly = np.polyfit(e_pre, i_pre, 1)
                    else:
                        pre_poly = [0.0, np.mean(self.integrated_xas[:3])]
                        
                    # Baseline-subtracted data
                    xas_baseline_sub = self.integrated_xas - np.polyval(pre_poly, self.energy_axis)
                    i_sel_sub = xas_baseline_sub[mask][sort_idx]
                    
                    # Edge jump step calculation at E0
                    edge_jump = np.polyval(pre_poly, e0)
                    if edge_jump <= 0.0:
                        edge_jump = 1.0
                    
                    # 2. Fit post-edge spline/polynomial: above E0 + 50 eV
                    post_mask = self.energy_axis > (e0 + 50.0)
                    if np.sum(post_mask) < 3:
                        sorted_energies = np.sort(self.energy_axis)
                        threshold = sorted_energies[int(len(sorted_energies) * 0.80)]
                        post_mask = self.energy_axis >= threshold
                        
                    e_post = self.energy_axis[post_mask]
                    i_post = xas_baseline_sub[post_mask]
                    
                    if len(e_post) > 3:
                        post_poly = np.polyfit(e_post, i_post, 3)
                    else:
                        post_poly = np.polyfit(e_post, i_post, 1) if len(e_post) > 1 else [0.0, np.mean(i_post)]
                        
                    # Evaluate post-edge background on active range
                    i_bg_sub = np.polyval(post_poly, e_sel)
                    
                    # 3. EXAFS chi(k) calculation
                    chi = (i_sel_sub - i_bg_sub) / edge_jump
                    
                    idx_e0 = np.argmin(np.abs(self.energy_axis - e0))
                    i0 = self.integrated_xas[idx_e0]
                    y_sel = chi * (k_sel ** w) + i0
                else:
                    idx_e0 = np.argmin(np.abs(self.energy_axis - e0))
                    i0 = self.integrated_xas[idx_e0]
                    y_sel = (i_sel - i0) * (k_sel ** w) + i0
                
                # Apply rebinning if requested
                if self.chk_rebin.isChecked():
                    rebin_size = self.spin_rebin_size.value()
                    n_pts = len(k_sel)
                    num_bins = n_pts // rebin_size
                    if num_bins > 0:
                        k_binned = []
                        y_binned = []
                        for b_idx in range(num_bins):
                            s_idx = b_idx * rebin_size
                            e_idx = (b_idx + 1) * rebin_size
                            k_binned.append(np.mean(k_sel[s_idx:e_idx]))
                            y_binned.append(np.mean(y_sel[s_idx:e_idx]))
                        k_sel = np.array(k_binned)
                        y_sel = np.array(y_binned)
                        
                self.k_line.set_data(k_sel, y_sel)
                self.k_ax.set_xlim(k_sel.min() * 0.95, k_sel.max() * 1.05)
                self.k_ax.set_ylim(y_sel.min() * 0.9, y_sel.max() * 1.1)
            else:
                self.k_line.set_data([], [])
        else:
            self.k_line.set_data([], [])
                
        self.k_canvas.draw_idle()
        self.xas_canvas.draw_idle()

    def update_slice_from_raw_idx(self, raw_idx):
        if self.data_2d is None: return
        self.current_idx = int(np.clip(raw_idx, 0, len(self.energy_axis) - 1))
        
        self.slider_energy.blockSignals(True)
        self.spin_energy.blockSignals(True)
        self.slider_energy.setValue(self.get_slider_val(self.current_idx))
        self.spin_energy.setValue(float(self.energy_axis[self.current_idx]))
        self.lbl_row_idx.setText(f"Row Index: {self.current_idx}")
        self.slider_energy.blockSignals(False)
        self.spin_energy.blockSignals(False)
        
        if hasattr(self, 'heat_slice_line'):
            self.heat_slice_line.set_ydata([self.energy_axis[self.current_idx]])
            self.heatmap_canvas.draw_idle()
            
        self.xas_slice_line.set_xdata([self.energy_axis[self.current_idx]])
        self.xas_canvas.draw_idle()
        
        self.plot_spectrum()
        self.update_fit_tab_slice()

    def on_slider_changed(self, value):
        self.update_slice_from_raw_idx(self.get_raw_idx(value))

    def on_spin_changed(self, value):
        closest = np.argmin(np.abs(self.energy_axis - value))
        self.update_slice_from_raw_idx(closest)

    def on_heatmap_click(self, event):
        tb = getattr(self.heatmap_canvas, 'toolbar', None)
        if tb is not None and tb.mode != "": return
        if event.inaxes is None or event.ydata is None: return
        closest = np.argmin(np.abs(self.energy_axis - event.ydata))
        self.update_slice_from_raw_idx(closest)

    def on_drag_roi_start(self, value):
        self.roi_start = value
        self.spin_roi_start.blockSignals(True)
        self.spin_roi_start.setValue(value)
        self.spin_roi_start.blockSignals(False)
        
        r_min = min(self.roi_start, self.roi_end)
        r_max = max(self.roi_start, self.roi_end)
        self.roi_start, self.roi_end = r_min, r_max
        self.calculate_roi_integration()
        
        # Sync positions
        self.roi_start_line.set_xdata([self.roi_start, self.roi_start])
        self.roi_end_line.set_xdata([self.roi_end, self.roi_end])
        self.roi_start_handle.set_xdata([self.roi_start])
        self.roi_end_handle.set_xdata([self.roi_end])
        self.roi_start_text.set_x(self.roi_start)
        self.roi_start_text.set_text(f"ROI Start\n({self.roi_start:.0f} eV)")
        self.roi_end_text.set_x(self.roi_end)
        self.roi_end_text.set_text(f"ROI End\n({self.roi_end:.0f} eV)")
        
        if hasattr(self, 'roi_span') and self.roi_span is not None:
            try: self.roi_span.remove()
            except: pass
        self.roi_span = self.mca_ax.axvspan(self.roi_start, self.roi_end, color='#10b981', alpha=0.1)
        
        self.mca_canvas.draw_idle()
        self.plot_integrated_xas()

    def on_drag_roi_end(self, value):
        self.roi_end = value
        self.spin_roi_end.blockSignals(True)
        self.spin_roi_end.setValue(value)
        self.spin_roi_end.blockSignals(False)
        
        r_min = min(self.roi_start, self.roi_end)
        r_max = max(self.roi_start, self.roi_end)
        self.roi_start, self.roi_end = r_min, r_max
        self.calculate_roi_integration()
        
        # Sync positions
        self.roi_start_line.set_xdata([self.roi_start, self.roi_start])
        self.roi_end_line.set_xdata([self.roi_end, self.roi_end])
        self.roi_start_handle.set_xdata([self.roi_start])
        self.roi_end_handle.set_xdata([self.roi_end])
        self.roi_start_text.set_x(self.roi_start)
        self.roi_start_text.set_text(f"ROI Start\n({self.roi_start:.0f} eV)")
        self.roi_end_text.set_x(self.roi_end)
        self.roi_end_text.set_text(f"ROI End\n({self.roi_end:.0f} eV)")
        
        if hasattr(self, 'roi_span') and self.roi_span is not None:
            try: self.roi_span.remove()
            except: pass
        self.roi_span = self.mca_ax.axvspan(self.roi_start, self.roi_end, color='#10b981', alpha=0.1)
        
        self.mca_canvas.draw_idle()
        self.plot_integrated_xas()

    def on_roi_bounds_changed(self):
        self.roi_start = min(self.spin_roi_start.value(), self.spin_roi_end.value())
        self.roi_end = max(self.spin_roi_start.value(), self.spin_roi_end.value())
        
        # Sync lines & text boxes
        self.roi_start_line.set_xdata([self.roi_start, self.roi_start])
        self.roi_end_line.set_xdata([self.roi_end, self.roi_end])
        self.roi_start_handle.set_xdata([self.roi_start])
        self.roi_end_handle.set_xdata([self.roi_end])
        self.roi_start_text.set_x(self.roi_start)
        self.roi_start_text.set_text(f"ROI Start\n({self.roi_start:.0f} eV)")
        self.roi_end_text.set_x(self.roi_end)
        self.roi_end_text.set_text(f"ROI End\n({self.roi_end:.0f} eV)")
        
        if hasattr(self, 'roi_span') and self.roi_span is not None:
            try: self.roi_span.remove()
            except: pass
        self.roi_span = self.mca_ax.axvspan(self.roi_start, self.roi_end, color='#10b981', alpha=0.1)
        
        self.mca_canvas.draw_idle()
        if self.data_2d is not None:
            self.calculate_roi_integration()
            self.plot_integrated_xas()

    def on_tab_changed(self, index):
        if self.tabs.tabText(index) == "EXAFS Fourier Transform (R-space)":
            self.calculate_fourier_transform()
            
        # Force re-render of canvas objects to load correctly between tabs
        if hasattr(self, 'heatmap_canvas'):
            self.heatmap_canvas.draw()
            self.mca_canvas.draw()
            self.k_canvas.draw()
            self.xas_canvas.draw()
            if hasattr(self, 'fit_k_canvas'):
                self.fit_k_canvas.draw()
                self.fit_xas_canvas.draw()
                self.fit_mca_canvas.draw()
            if hasattr(self, 'ft_canvas'):
                self.ft_canvas.draw()

    def run_batch_fit(self):
        if self.data_2d is None: return
        if len(self.active_elements) == 0:
            QMessageBox.warning(self, "Fit Error", "Please select at least one element to fit.")
            return
            
        self.btn_run_fit.setEnabled(False)
        self.fit_lbl_status.setText("Status: Fitting slices...")
        
        init_bg_params = (self.sigma_scatt, self.step_frac, self.tail_frac, self.beta, self.b0, self.b1, self.leftslope, self.cutoff, self.rounding)
        
        self.fit_worker = DynamicFitWorker(
            self.data_2d, self.energy_axis, self.mca_energy_axis,
            self.active_elements, self.element_peaks_ints, self.element_edges,
            fit_min_kev=self.fit_spin_range_min.value(), fit_max_kev=self.fit_spin_range_max.value(),
            init_bg_params=init_bg_params, w_peaks=self.w_peaks, tol=1e-6
        )
        self.fit_worker.progress.connect(self.fit_progress.setValue)
        self.fit_worker.finished.connect(self.on_fit_finished)
        self.fit_worker.start()

    def on_fit_finished(self, results):
        self.btn_run_fit.setEnabled(True)
        if "stopped" in results:
            self.fit_lbl_status.setText("Status: Stopped")
            return
            
        self.fitted_profiles = results["fitted_profiles"]
        self.fitted_params = results["fitted_params"]
        self.fit_lbl_status.setText("Status: Batch Fit Complete")
        
        # Fill element selection dropdown
        self.fit_progress.setValue(100)
        self.auto_detect_fit_e0()
        self.plot_fit_xas_and_exafs()
        self.update_fit_tab_slice()

    def plot_fit_xas_and_exafs(self):
        element = self.fit_combo_element.currentText()
        if not element or element not in self.fitted_profiles:
            return
            
        xas_profile = self.fitted_profiles[element]
        self.fit_xas_line.set_data(self.energy_axis, xas_profile * 1e7)
        self.fit_xas_slice_line.set_xdata([self.energy_axis[self.current_idx], self.energy_axis[self.current_idx]])
        
        e0 = self.fit_spin_e0.value()
        kmin = self.fit_spin_k_min.value()
        kmax = self.fit_spin_k_max.value()
        w = self.fit_combo_weight.currentIndex() + 1
        
        sort_idx = np.argsort(self.energy_axis)
        e_arr = self.energy_axis[sort_idx]
        i_arr = xas_profile[sort_idx] * 1e7
        
        mask = (e_arr >= kmin) & (e_arr <= kmax) & (e_arr > e0)
        if np.sum(mask) > 3:
            e_sel = e_arr[mask]
            i_sel = i_arr[mask]
            k_sel = np.sqrt(0.26246718 * (e_sel - e0))
            
            # Recalculate EXAFS intensity with standard pre-edge & post-edge background subtraction
            if hasattr(self, 'fit_chk_subtract_bg') and self.fit_chk_subtract_bg.isChecked() and len(self.energy_axis) > 10:
                # 1. Fit pre-edge
                pre_mask = self.energy_axis < (e0 - 30.0)
                if np.sum(pre_mask) < 3:
                    sorted_energies = np.sort(self.energy_axis)
                    threshold = sorted_energies[int(len(sorted_energies) * 0.15)]
                    pre_mask = self.energy_axis <= threshold
                
                e_pre = self.energy_axis[pre_mask]
                i_pre = (xas_profile * 1e7)[pre_mask]
                
                if len(e_pre) > 1:
                    pre_poly = np.polyfit(e_pre, i_pre, 1)
                else:
                    pre_poly = [0.0, np.mean((xas_profile * 1e7)[:3])]
                    
                xas_baseline_sub = (xas_profile * 1e7) - np.polyval(pre_poly, self.energy_axis)
                i_sel_sub = xas_baseline_sub[mask][sort_idx]
                
                edge_jump = np.polyval(pre_poly, e0)
                if edge_jump <= 0.0:
                    edge_jump = 1.0
                
                # 2. Fit post-edge
                post_mask = self.energy_axis > (e0 + 50.0)
                if np.sum(post_mask) < 3:
                    sorted_energies = np.sort(self.energy_axis)
                    threshold = sorted_energies[int(len(sorted_energies) * 0.80)]
                    post_mask = self.energy_axis >= threshold
                    
                e_post = self.energy_axis[post_mask]
                i_post = xas_baseline_sub[post_mask]
                
                if len(e_post) > 3:
                    post_poly = np.polyfit(e_post, i_post, 3)
                else:
                    post_poly = np.polyfit(e_post, i_post, 1) if len(e_post) > 1 else [0.0, np.mean(i_post)]
                    
                i_bg_sub = np.polyval(post_poly, e_sel)
                
                # 3. EXAFS chi calculation
                chi = (i_sel_sub - i_bg_sub) / edge_jump
                
                idx_e0 = np.argmin(np.abs(self.energy_axis - e0))
                i0 = xas_profile[idx_e0] * 1e7
                y_sel = chi * (k_sel ** w) + i0
            else:
                idx_e0 = np.argmin(np.abs(self.energy_axis - e0))
                i0 = xas_profile[idx_e0] * 1e7
                y_sel = (i_sel - i0) * (k_sel ** w) + i0
            
            # Apply rebinning if requested
            if self.fit_chk_rebin.isChecked():
                rebin_size = self.fit_spin_rebin_size.value()
                n_pts = len(k_sel)
                num_bins = n_pts // rebin_size
                if num_bins > 0:
                    k_binned = []
                    y_binned = []
                    for b_idx in range(num_bins):
                        s_idx = b_idx * rebin_size
                        e_idx = (b_idx + 1) * rebin_size
                        k_binned.append(np.mean(k_sel[s_idx:e_idx]))
                        y_binned.append(np.mean(y_sel[s_idx:e_idx]))
                    k_sel = np.array(k_binned)
                    y_sel = np.array(y_binned)
            
            self.fit_k_line.set_data(k_sel, y_sel)
            self.fit_k_ax.set_xlim(k_sel.min() * 0.95, k_sel.max() * 1.05)
            self.fit_k_ax.set_ylim(y_sel.min() * 0.9, y_sel.max() * 1.1)
            
        self.fit_k_canvas.draw_idle()
        self.fit_xas_canvas.draw_idle()

    def update_fit_tab_slice(self):
        if self.data_2d is None: return
        x_mca = self.mca_energy_axis / 1000.0
        y_data = self.data_2d[self.current_idx] * 1e7
        e_inc = self.energy_axis[self.current_idx]
        mu_inc = e_inc / 1000.0
        
        self.fit_mca_raw_line.set_data(self.mca_energy_axis, y_data)
        
        # Calculate fit curve model on-the-fly
        if self.fitted_params is not None:
            params = self.fitted_params[self.current_idx]
            n_elems = len(self.active_elements)
            
            shapes_dict = {}
            for name in self.active_elements:
                p_arr = np.array(self.element_peaks_ints[name]["peaks"]) / 1000.0
                i_arr = np.array(self.element_peaks_ints[name]["intensities"])
                line_edges = self.element_peaks_ints[name].get("line_edges", {})
                shapes_dict[name] = worker_calc_peaks_dynamic(x_mca, 1.0, p_arr, i_arr, self.w_peaks, e_inc=e_inc, line_edges=line_edges)
                
            init_bg_vals = (self.sigma_scatt, self.step_frac, self.tail_frac, self.leftslope)
            fit_total = worker_model_func_dynamic(params, x_mca, mu_inc, self.active_elements, shapes_dict, init_bg_vals)
            
            self.fit_mca_total_line.set_data(self.mca_energy_axis, fit_total)
            
            # Scatter line
            amp, mu, b0, b1 = params[n_elems:n_elems+4]
            scatt = worker_calc_scatter_opt(x_mca, mu, amp, self.sigma_scatt, self.step_frac, self.tail_frac, 1.04, b0, b1, self.leftslope, 5.8, 1.0)
            self.fit_mca_scatt_line.set_data(self.mca_energy_axis, scatt)
            
            # Individual element lines
            for elem_idx, name in enumerate(self.active_elements):
                amp_val = params[elem_idx]
                if e_inc < self.element_edges[name] - 20.0:
                    amp_val = 0.0
                elem_profile = amp_val * shapes_dict[name]
                self.fit_mca_elem_lines[name].set_data(self.mca_energy_axis, elem_profile)
                
        self.fit_mca_canvas.draw_idle()

    def calculate_fourier_transform(self):
        if self.energy_axis is None: return
        source = self.ft_combo_source.currentText()
        element = self.ft_combo_element.currentText()
        kweight = self.ft_spin_kweight.value()
        window_type = self.ft_combo_window.currentText()
        kmin = self.ft_spin_kmin.value()
        kmax = self.ft_spin_kmax.value()
        rmax = self.ft_spin_rmax.value()
        k_step = self.ft_spin_kstep.value()
        
        if "Peak-Fitted" in source:
            if self.fitted_params is None or not element:
                self.ft_k_ax.clear()
                self.ft_r_ax.clear()
                self.ft_canvas.draw_idle()
                return
            xas_profile = self.fitted_profiles[element]
            e0 = self.fit_spin_e0.value()
            i_xas = xas_profile * 1e7
        else:
            if self.integrated_xas is None: return
            e0 = self.spin_e0.value()
            i_xas = self.integrated_xas
            
        e_min_real = e0 + 5.0
        e_max_real = self.energy_axis.max()
        
        sort_idx = np.argsort(self.energy_axis)
        e_arr = self.energy_axis[sort_idx]
        i_arr = i_xas[sort_idx]
        
        mask = (e_arr > e0)
        if np.sum(mask) < 5: return
        
        e_exafs = e_arr[mask]
        i_exafs = i_arr[mask]
        k_arr = np.sqrt(0.26246718 * (e_exafs - e0))
        
        # Calculate EXAFS chi(k) with standard pre-edge & post-edge background subtraction
        # 1. Fit pre-edge
        pre_mask = e_arr < (e0 - 30.0)
        if np.sum(pre_mask) < 3:
            pre_mask = e_arr <= e_arr[int(len(e_arr) * 0.15)]
        
        e_pre = e_arr[pre_mask]
        i_pre = i_arr[pre_mask]
        
        if len(e_pre) > 1:
            pre_poly = np.polyfit(e_pre, i_pre, 1)
        else:
            pre_poly = [0.0, np.mean(i_arr[:3])]
            
        xas_baseline_sub = i_arr - np.polyval(pre_poly, e_arr)
        i_sel_sub = xas_baseline_sub[mask]
        
        edge_jump = np.polyval(pre_poly, e0)
        if edge_jump <= 0.0:
            edge_jump = 1.0
        
        # 2. Fit post-edge
        post_mask = e_arr > (e0 + 50.0)
        if np.sum(post_mask) < 3:
            post_mask = e_arr >= e_arr[int(len(e_arr) * 0.80)]
            
        e_post = e_arr[post_mask]
        i_post = xas_baseline_sub[post_mask]
        
        if len(e_post) > 3:
            post_poly = np.polyfit(e_post, i_post, 3)
        else:
            post_poly = np.polyfit(e_post, i_post, 1) if len(e_post) > 1 else [0.0, np.mean(i_post)]
            
        i_bg_sub = np.polyval(post_poly, e_exafs)
        
        # 3. EXAFS chi calculation
        chi = (i_sel_sub - i_bg_sub) / edge_jump
        
        kmin_clamped = max(kmin, k_arr.min())
        kmax_clamped = min(kmax, k_arr.max())
        k_grid = np.arange(kmin_clamped, kmax_clamped + 1e-9, k_step)
        
        chi_interp = np.interp(k_grid, k_arr, chi, left=chi[0], right=chi[-1])
        weighted_chi = chi_interp * (k_grid ** kweight)
        
        # FFT Window
        n_pts = len(k_grid)
        window = np.ones(n_pts)
        dk_val = self.ft_spin_dk.value()
        
        if kmax_clamped - kmin_clamped < 2 * dk_val:
            dk_val = (kmax_clamped - kmin_clamped) / 2.1
            
        if window_type == "Hanning":
            for idx_k, kv in enumerate(k_grid):
                if kv < kmin_clamped: window[idx_k] = 0.0
                elif kv < kmin_clamped + dk_val: window[idx_k] = 0.5 * (1.0 - np.cos(np.pi * (kv - kmin_clamped) / dk_val))
                elif kv > kmax_clamped: window[idx_k] = 0.0
                elif kv > kmax_clamped - dk_val: window[idx_k] = 0.5 * (1.0 + np.cos(np.pi * (kv - (kmax_clamped - dk_val)) / dk_val))
        elif window_type == "Hamming":
            for idx_k, kv in enumerate(k_grid):
                if kv < kmin_clamped: window[idx_k] = 0.0
                elif kv < kmin_clamped + dk_val: window[idx_k] = 0.54 - 0.46 * np.cos(np.pi * (kv - kmin_clamped) / dk_val)
                elif kv > kmax_clamped: window[idx_k] = 0.0
                elif kv > kmax_clamped - dk_val: window[idx_k] = 0.54 + 0.46 * np.cos(np.pi * (kv - (kmax_clamped - dk_val)) / dk_val)
        elif window_type == "Kaiser-Bessel":
            beta = 2.5
            i0_beta = i0(beta)
            for idx_k, kv in enumerate(k_grid):
                if kv < kmin_clamped or kv > kmax_clamped: window[idx_k] = 0.0
                elif kv < kmin_clamped + dk_val:
                    term = 1.0 - (1.0 - (kv - kmin_clamped)/dk_val)**2
                    window[idx_k] = i0(beta * np.sqrt(max(0.0, term))) / i0_beta
                elif kv > kmax_clamped - dk_val:
                    term = 1.0 - ((kv - (kmax_clamped - dk_val))/dk_val)**2
                    window[idx_k] = i0(beta * np.sqrt(max(0.0, term))) / i0_beta
        else:
            window[k_grid < kmin_clamped] = 0.0
            window[k_grid > kmax_clamped] = 0.0
            
        windowed_chi = weighted_chi * window
        if len(windowed_chi) > 0:
            windowed_chi = windowed_chi - np.mean(windowed_chi)
            
        n_fft = 2048
        chi_padded = np.zeros(n_fft)
        chi_padded[:n_pts] = windowed_chi
        
        r_fft = np.fft.rfft(chi_padded, n_fft)
        scale = (2.0 * k_step) / np.sqrt(2.0 * np.pi)
        r_fft = r_fft * scale
        r_fft_mag = np.abs(r_fft)
        
        r_step = np.pi / (n_fft * k_step)
        r_grid = np.arange(len(r_fft_mag)) * r_step
        
        self.ft_k_ax.clear()
        self.ft_r_ax.clear()
        
        cfg = self.styles[self.theme]
        self.ft_k_ax.plot(k_arr, chi * (k_arr ** kweight), '.', color='#a1a1aa', markersize=3.5, label='Raw Data Points')
        self.ft_k_ax.plot(k_grid, weighted_chi, color='#6366f1', linewidth=1.5, label=f'Interpolated Grid (k^{kweight})')
        self.ft_k_ax.plot(k_grid, window * np.max(np.abs(weighted_chi)), color='#ef4444', linestyle='--', alpha=0.7, label='Taper Window')
        self.ft_k_ax.set_title(f"k-Weighted EXAFS Signal & Window ({source})", color=cfg["text"])
        self.ft_k_ax.legend(loc='upper right', facecolor=cfg["fig_face"], edgecolor=cfg["spine"], labelcolor=cfg["text"])
        
        r_mask = r_grid <= rmax
        self.ft_r_ax.plot(r_grid[r_mask], r_fft_mag[r_mask], color='#10b981', linewidth=2, label='|χ(R)| (Magnitude)')
        self.ft_r_ax.set_title("Fourier Transform Magnitude |χ(R)| vs Distance R", color=cfg["text"])
        self.ft_r_ax.legend(loc='upper right', facecolor=cfg["fig_face"], edgecolor=cfg["spine"], labelcolor=cfg["text"])
        
        self.ft_k_ax.set_facecolor(cfg["ax_face"])
        self.ft_r_ax.set_facecolor(cfg["ax_face"])
        self.ft_k_ax.tick_params(colors=cfg["text"])
        self.ft_r_ax.tick_params(colors=cfg["text"])
        self.ft_fig.tight_layout()
        self.ft_canvas.draw_idle()

    def export_exafs_data(self):
        if self.energy_axis is None: return
        source = self.ft_combo_source.currentText()
        element = self.ft_combo_element.currentText() if "Peak-Fitted" in source else "ROI Integrated"
        kweight = self.ft_spin_kweight.value()
        kmin = self.ft_spin_kmin.value()
        kmax = self.ft_spin_kmax.value()
        
        default_name = f"exafs_{source.replace(' ', '_').lower()}_{element.lower()}_kw{kweight}.txt"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save EXAFS Data", default_name, "Text Files (*.txt)")
        if not file_path: return
        
        try:
            # Recalculate spline to save values
            if "Peak-Fitted" in source:
                if self.fitted_params is None or not element: return
                xas_profile = self.fitted_profiles[element]
                e0 = self.fit_spin_e0.value()
                i_xas = xas_profile * 1e7
            else:
                if self.integrated_xas is None: return
                e0 = self.spin_e0.value()
                i_xas = self.integrated_xas
                
            e_min_real = e0 + (kmin ** 2) / 0.26246718
            e_max_real = e0 + (kmax ** 2) / 0.26246718
            e_max_real = min(e_max_real, self.energy_axis.max())
            
            sort_idx = np.argsort(self.energy_axis)
            e_arr = self.energy_axis[sort_idx]
            i_arr = i_xas[sort_idx]
            
            mask = (e_arr >= e_min_real) & (e_arr <= e_max_real) & (e_arr > e0)
            if np.sum(mask) < 5: return
            
            e_exafs = e_arr[mask]
            i_exafs = i_arr[mask]
            k_arr = np.sqrt(0.26246718 * (e_exafs - e0))
            
            idx_e0 = np.argmin(np.abs(self.energy_axis - e0))
            i0_val = i_xas[idx_e0]
            
            # Calculate EXAFS chi(k) with standard pre-edge & post-edge background subtraction
            # 1. Fit pre-edge
            pre_mask = self.energy_axis < (e0 - 30.0)
            if np.sum(pre_mask) < 3:
                sorted_energies = np.sort(self.energy_axis)
                threshold = sorted_energies[int(len(sorted_energies) * 0.15)]
                pre_mask = self.energy_axis <= threshold
            
            e_pre = self.energy_axis[pre_mask]
            i_pre = i_xas[pre_mask]
            
            if len(e_pre) > 1:
                pre_poly = np.polyfit(e_pre, i_pre, 1)
            else:
                pre_poly = [0.0, np.mean(i_xas[:3])]
                
            xas_baseline_sub = i_xas - np.polyval(pre_poly, self.energy_axis)
            i_sel_sub = xas_baseline_sub[mask][sort_idx]
            
            edge_jump = np.polyval(pre_poly, e0)
            if edge_jump <= 0.0:
                edge_jump = 1.0
            
            # 2. Fit post-edge
            post_mask = self.energy_axis > (e0 + 50.0)
            if np.sum(post_mask) < 3:
                sorted_energies = np.sort(self.energy_axis)
                threshold = sorted_energies[int(len(sorted_energies) * 0.80)]
                post_mask = self.energy_axis >= threshold
                
            e_post = self.energy_axis[post_mask]
            i_post = xas_baseline_sub[post_mask]
            
            if len(e_post) > 3:
                post_poly = np.polyfit(e_post, i_post, 3)
            else:
                post_poly = np.polyfit(e_post, i_post, 1) if len(e_post) > 1 else [0.0, np.mean(i_post)]
                
            i_bg_sub = np.polyval(post_poly, e_exafs)
            
            # 3. EXAFS chi calculation
            chi = (i_sel_sub - i_bg_sub) / edge_jump
            
            k_step = self.ft_spin_kstep.value()
            kmin_clamped = max(kmin, k_arr.min())
            kmax_clamped = min(kmax, k_arr.max())
            
            k_grid = np.arange(kmin_clamped, kmax_clamped + 1e-9, k_step)
            chi_interp = np.interp(k_grid, k_arr, chi, left=chi[0], right=chi[-1])
            weighted_chi = chi_interp * (k_grid ** kweight)
            
            header_lines = [
                "# Quick XAFS EXAFS Export File",
                f"# Source Dataset: {self.lbl_file_path.text()}",
                f"# Edge Energy E0: {e0:.3f} eV",
                f"# k-weight: k^{kweight}",
                f"# Range: k = [{kmin_clamped:.3f}, {kmax_clamped:.3f}]",
                "# Columns: 1: k (Å⁻¹), 2: k^w * chi(k) (Arbitrary Units)"
            ]
            np.savetxt(file_path, np.column_stack((k_grid, weighted_chi)), fmt=["%.6f", "%.8e"], header="\n".join(header_lines), comments="")
            QMessageBox.information(self, "Success", "EXAFS data saved.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed: {e}")

    def export_fit_exafs_data(self):
        self.export_exafs_data()

    def export_fit_xas_data(self):
        element = self.fit_combo_element.currentText()
        if not element or self.fitted_profiles.get(element) is None: return
        default_name = f"xas_peakfitted_{element.lower()}.txt"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save XAS Profile Data", default_name, "Text Files (*.txt)")
        if not file_path: return
        try:
            xas_profile = self.fitted_profiles[element] * 1e7
            sort_idx = np.argsort(self.energy_axis)
            header_lines = [
                "# Quick XAFS XAS Profile Export",
                f"# Element: {element}",
                "# Columns: 1: Energy (eV), 2: Intensity (Arbitrary Units)"
            ]
            np.savetxt(file_path, np.column_stack((self.energy_axis[sort_idx], xas_profile[sort_idx])), fmt=["%.3f", "%.8e"], header="\n".join(header_lines), comments="")
            QMessageBox.information(self, "Success", "XAS profile saved.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed: {e}")

    def on_ft_source_changed(self, text):
        self.ft_combo_element.setEnabled("Peak-Fitted" in text)
        self.calculate_fourier_transform()

    def on_log_1d_toggled(self, state):
        self.use_log_scale_1d = (state == Qt.Checked)
        self.plot_spectrum()

    def on_scroll(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None: return
        ax = event.inaxes
        scale_factor = 1.0 / 1.15 if event.step > 0 else 1.15
        
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        rel_x = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        ax.set_xlim([event.xdata - new_width * (1 - rel_x), event.xdata + new_width * rel_x])
        
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        rel_y = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])
        ax.set_ylim([event.ydata - new_height * (1 - rel_y), event.ydata + new_height * rel_y])
        ax.figure.canvas.draw_idle()

    def on_save_spectrum_clicked(self):
        if self.data_2d is None: return
        default_name = f"spectrum_slice_{self.current_idx}_energy_{self.energy_axis[self.current_idx]:.0f}eV.txt"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Spectrum Slice", default_name, "Text Files (*.txt)")
        if not file_path: return
        try:
            data = self.data_2d[self.current_idx]
            header = f"# Spectrum Slice at Energy: {self.energy_axis[self.current_idx]:.3f} eV\n# Columns: 1: Emission Energy (eV), 2: Intensity (Counts)"
            np.savetxt(file_path, np.column_stack((self.mca_energy_axis, data)), fmt=["%.3f", "%.8e"], header=header, comments="")
            QMessageBox.information(self, "Success", "Spectrum saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed: {e}")

    def on_theme_switch_toggled(self, checked):
        self.theme = "charcoal" if checked else "light"
        self.apply_theme()

    def apply_theme(self):
        cfg = self.styles[self.theme]
        self.setStyleSheet(cfg["qss"])
        self.mca_fig.patch.set_facecolor(cfg["fig_face"])
        self.xas_fig.patch.set_facecolor(cfg["fig_face"])
        self.k_fig.patch.set_facecolor(cfg["fig_face"])
        self.fit_k_fig.patch.set_facecolor(cfg["fig_face"])
        self.fit_xas_fig.patch.set_facecolor(cfg["fig_face"])
        self.fit_mca_fig.patch.set_facecolor(cfg["fig_face"])
        self.ft_fig.patch.set_facecolor(cfg["fig_face"])
        
        for ax in [self.mca_ax, self.xas_ax, self.k_ax, self.fit_k_ax, self.fit_xas_ax, self.fit_mca_ax, self.ft_k_ax, self.ft_r_ax]:
            ax.set_facecolor(cfg["ax_face"])
            ax.tick_params(colors=cfg["text"], labelsize=11)
            ax.xaxis.label.set_color(cfg["text"])
            ax.yaxis.label.set_color(cfg["text"])
            ax.title.set_color(cfg["text"])
            ax.grid(True, color=cfg["grid"], linestyle=':', alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color(cfg["spine"])
                
        self.spectrum_line.set_color('#6366f1')
        self.roi_start_line.set_color('#10b981')
        self.roi_end_line.set_color('#10b981')
        self.roi_start_handle.set_color('#10b981')
        self.roi_end_handle.set_color('#10b981')
        self.roi_start_text.set_color('#10b981')
        self.roi_start_text.get_bbox_patch().set_facecolor(cfg["fig_face"])
        self.roi_end_text.set_color('#10b981')
        self.roi_end_text.get_bbox_patch().set_facecolor(cfg["fig_face"])
        
        self.mca_canvas.draw_idle()
        self.xas_canvas.draw_idle()
        self.k_canvas.draw_idle()
        self.fit_k_canvas.draw_idle()
        self.fit_xas_canvas.draw_idle()
        self.fit_mca_canvas.draw_idle()
        self.ft_canvas.draw_idle()
        if self.data_2d is not None:
            self.plot_heatmap()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    gui = XASExplorerGUI()
    gui.show()
    sys.exit(app.exec_())
