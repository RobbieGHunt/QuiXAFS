# QuiXAFS

A suite of standalone, interactive Python tools for raw data exploration, batch processing, and spectral analysis of Extended X-ray Absorption Fine-Structure (EXAFS) and X-ray Absorption Spectroscopy (XAS) data obtained from the **BM28 XMaS beamline** at the **ESRF**.

---

## System Requirements & Dependencies

These tools are built for **Python 3.8+** and depend on standard scientific Python libraries and PyQt5 for user interfaces.

### Dependencies
- **PyQt5** (GUI layouts and event loops)
- **NumPy** (matrix operations, stackings)
- **Pandas** (reading Spec/ZAP CSV files)
- **SciPy** (multivariable optimization and fitting algorithms)
- **Matplotlib** (interactive plot canvases)

### Installation
You can install all dependencies in a single step using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## Scripts

### 1. Raw EDF Explorer (`raw_edf_explorer.py`)
An interactive explorer interface to inspect raw ESRF Data Format (EDF) files. Displays the 2D map and allows for selecting an individual spectrum.

- **Features**:
  - Open any raw `.edf` binary file containing 2D MCA datasets.
  - Scroll through incident energy slices and plot individual MCA spectra.
  - Select regions of interest (ROI) and inspect raw detector counts.
- **Dependency Files**: Requires [data_loader.py](data_loader.py) in the same directory for EDF parsing.
- **Run**:
  ```bash
  python raw_edf_explorer.py
  ```
  Or run from any python IDE (e.g. Spyder).

---

### 2. ZAP Processor (`process_and_plot.py`)
A PyQt5 GUI utility to batch load, align, normalize, and average raw ZAP scans (EDF format) and their SPEC metadata CSV files.

- **Features**:
  - Browse and select inputs for ZAP (raw `.edf` files), CSV (SPEC scan metadata), and save outputs.
  - **Range Selection**: Dynamic start and finish spin boxes to quickly select a range of scans for processing, plus Select All/Clear All overrides.
  - **Robust Error Handling**: Handles EDF/CSV row count mismatches by slicing to the minimum common length, and skips corrupted files gracefully.
  - **Auto-Averaging & Standard Deviation**: Computes average data and standard deviations (`ddof=1`).
  - **Combined Package Output**: Saves all arrays into a single compressed NumPy container (`<basename>.npz`) alongside individual `.npy` files for backwards compatibility.
  - **Heatmap Generation**: Calibrates emission energies using IUPAC reference lines and saves the 2D average heatmap as `<basename>_heatmap.png`.
- **Run**:
  ```bash
  python process_and_plot.py
  ```
  Or run from any python IDE (e.g. Spyder).

---

### 3. QuiXAFS (`QuiXAFS.py`)
An interactive PyQt5 application for visualizing, calibrating, and performing multi-component fits of XAS/EXAFS datasets.

- **Features**:
  - **Dynamic File Format Support**: Loads standard 2D `.npy` files or single compressed `.npz` dataset packages containing average intensity, standard deviations, and energy axes.
  - **Example Data**: Includes an example NumPy dataset in [`example_data/`](example_data/) (`averaged_normalized_zap.npy` with `zap_energy_axis.npy`, `mca_energy_axis.npy`, and `standard_error_zap.npy`).
  - **Integrated ROI Integration Tool**: Draggable boundaries on the 1D spectrum plot to map specific emission lines.
  - **Element Selection**: Select from a periodic table which elements to include and model to compare against (elemental data pulled from https://xraypy.github.io/XrayDB/).
  - **Emission Line Calibration**: Maps MCA channels to emission energy (eV) using IUPAC database lines.
  - **Advanced Fitting**: Solves multi-component models (Tb, Co, Fe, Cr) with background scatter corrections.
- **Dependency Files**: Requires [emission_lines.json](emission_lines.json) in the same directory to load reference database lines.
- **Run**:
  ```bash
  python QuiXAFS.py
  ```
  Or run from any python IDE (e.g. Spyder).

---

## 📊 Example Dataset

An example pre-processed NumPy dataset is provided in the [`example_data/`](example_data/) directory:

> [!IMPORTANT]
> **This example dataset is for the EXAFS GUI script (`QuiXAFS.py`) ONLY.**
> The Raw EDF Explorer (`raw_edf_explorer.py`) and ZAP Processor (`process_and_plot.py`) process raw ESRF `.edf` binary detector files and SPEC metadata CSV files, whereas `QuiXAFS.py` operates on pre-processed/normalized 2D NumPy array packages.

### Files in `example_data/`:
- **`averaged_normalized_zap.npy`**: 2D array of averaged, normalized ZAP fluorescence spectra (incident energy points × MCA channels).
- **`zap_energy_axis.npy`**: 1D array of incident photon energies (eV).
- **`mca_energy_axis.npy`**: 1D array of calibrated MCA detector energies (eV).
- **`standard_error_zap.npy`**: 2D array of standard errors across processed scans.

### Loading the Example Dataset in `QuiXAFS.py`:
1. Launch `QuiXAFS.py`.
2. Click **"Load 2D NumPy"** in the top-left panel.
3. Select `example_data/averaged_normalized_zap.npy`.
4. The GUI will automatically locate the matching energy axes (`zap_energy_axis.npy`, `mca_energy_axis.npy`) and error array (`standard_error_zap.npy`) in the same directory.

## 🌓 Dark / Light Themes

All three applications feature a theme toggle button in the **top-left** of the interface:
- **Toggle Theme**: Instantly switch between the default dark (Charcoal) theme and a light theme. The Matplotlib figures, axes, reference lines, and label colors adjust dynamically.
- **Preference Persistence**: The active theme selection is saved locally next to the script in a JSON configuration file (`quixafs_defaults.json`, `zap_defaults.json`, or `raw_edf_defaults.json`). The tool automatically loads and applies your preferred theme the next time you launch the application.


