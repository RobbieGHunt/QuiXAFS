# QuiXAFS

A suite of standalone, interactive Python tools for raw data exploration, batch processing, and spectral analysis of Extended X-ray Absorption Fine-Structure (EXAFS) and X-ray Absorption Spectroscopy (XAS) data obtained from the **BM28 XMaS beamline** at the **ESRF**.

---

## 📁 Repository Structure

```
QuiXAFS_repo/
├── QuiXAFS.py                      # [Main GUI] EXAFS Spectral Deconvolution & Analysis
├── process_and_plot.py             # [Main GUI] ZAP Batch Normalization & Averaging
├── raw_edf_explorer.py             # [Main GUI] Raw MCA / EDF Frame Inspector
│
├── utils/                          # Helper / Utility modules & scientific loaders
│   ├── __init__.py
│   └── data_loader.py              # In-memory SPEC & EDF parser library
│
├── resources/                      # Static reference databases & assets
│   └── emission_lines.json         # IUPAC atomic line database
│
├── config/                         # User default settings (auto-created)
│   ├── quixafs_defaults.json
│   ├── raw_edf_defaults.json
│   └── zap_defaults.json
│
├── example_data/                   # Pre-processed example datasets
│   ├── averaged_normalized_zap.npy
│   ├── mca_energy_axis.npy
│   ├── standard_error_zap.npy
│   └── zap_energy_axis.npy
│
├── requirements.txt                # Python environment dependencies
├── README.md                       # This documentation guide
├── .gitignore                      # Git ignore rules
└── .gitattributes                  # Git attributes
```

---

## ⚙️ System Requirements & Dependencies

These tools are built for **Python 3.8+** and depend on standard scientific Python libraries and PyQt5 for user interfaces.

### Dependencies
- **PyQt5** (GUI layouts and event loops)
- **NumPy** (matrix operations, stackings)
- **Pandas** (reading SPEC/ZAP metadata)
- **SciPy** (multivariable optimization and fitting algorithms)
- **Matplotlib** (interactive plot canvases)

### Installation
You can install all dependencies in a single step using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## GUI Applications

### 1. Raw EDF Explorer (`raw_edf_explorer.py`)
An interactive explorer interface to inspect raw ESRF Data Format (EDF) files. Displays the 2D map and allows for selecting an individual spectrum.

  - Open any raw `.edf` binary file containing 2D MCA datasets.
  - Select an individual energy slice and plot MCA spectra.
- **Dependencies**: Leverages [`utils/data_loader.py`](utils/data_loader.py) for EDF binary parsing.
- **Run**:
  ```bash
  python raw_edf_explorer.py
  ```
... Or launch from an IDE.
---

### 2. ZAP Processor (`process_and_plot.py`)
A PyQt5 GUI utility to batch load, align, normalize, and average raw ZAP scans (EDF format) and their SPEC metadata. Accepts converted csv files from the BM28 scripts also.

- **Features**:
  - **Dynamic Range Selection**: Start and finish spin boxes to quickly select a range of scans for processing, plus Select All/Clear All overrides. Select which scans to average if many samples or conditions are in the same directories.
  - **Auto-Averaging & Standard Deviation**: Computes average data and standard deviations (`ddof=1`).
  - **Combined Output**: Saves all arrays into a single compressed NumPy container (`<basename>.npz`) alongside individual `.npy` files. This contains an array for the averaged data, the error, and the calibration of the energy channels from the SPEC (CSV) file.
  - **Heatmap Generation**: Calibrates emission energies using IUPAC reference lines and saves the 2D average heatmap as `<basename>_heatmap.png`.
- **Dependencies**: Leverages [`utils/data_loader.py`](utils/data_loader.py) for in-memory SPEC and EDF processing.
- **Run**:
  ```bash
  python process_and_plot.py
  ```
... Or launch from an IDE.
---

### 3. QuiXAFS (`QuiXAFS.py`)
An interactive PyQt5 application for visualizing, calibrating, and performing multi-component fits of XAS/EXAFS datasets. Uses emission lines obtain from xraydb (https://github.com/xraypy/XrayDB).

- **Features**:
  - **Example Data**: Includes an example NumPy dataset in [`example_data/`](example_data/) (`averaged_normalized_zap.npy` with `zap_energy_axis.npy`, `mca_energy_axis.npy`, and `standard_error_zap.npy`).
  - **Integrated ROI Integration Tool**: Draggable boundaries on the 1D spectrum plot to map specific emission lines.
  - **Element Selection**: Select from a periodic table which elements to include and model to compare against (elemental data pulled from IUPAC database).
  - **Emission Line Calibration**: Maps MCA channels to emission energy (eV) using IUPAC database lines.
  - **Advanced Fitting**: Solves multi-component models (selected from periodic table) with background scatter corrections. By fitting gaussians to these emission lines, elemental "intensities" can be resolved within a ROI if elements overlap, helping to remove artifacts.
- **Dependencies**: Loads atomic transition reference data from [`resources/emission_lines.json`](resources/emission_lines.json).
- **Run**:
  ```bash
  python QuiXAFS.py
  ```
... Or launch from an IDE.
---

## 📊 Example Dataset

An example pre-processed NumPy dataset is provided in the [`example_data/`](example_data/) directory:

> [!IMPORTANT]
> **This example dataset is for the EXAFS GUI script (`QuiXAFS.py`) ONLY.**
> The Raw EDF Explorer (`raw_edf_explorer.py`) and ZAP Processor (`process_and_plot.py`) process raw ESRF `.edf` binary detector files and SPEC files, whereas `QuiXAFS.py` operates on pre-processed/normalized 2D NumPy array packages.

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

---

## 🌓 Dark / Light Themes & Persistence

All three applications feature a theme toggle button in the **top-left** of the interface:
- **Toggle Theme**: Instantly switch between the default dark (Charcoal) theme and a light theme. The Matplotlib figures, axes, reference lines, and label colors adjust dynamically.
- **Preference Persistence**: User configurations and theme preferences are saved in the `config/` directory (e.g., `config/quixafs_defaults.json`, `config/zap_defaults.json`, or `config/raw_edf_defaults.json`). The tools automatically load and apply your preferences on launch.
