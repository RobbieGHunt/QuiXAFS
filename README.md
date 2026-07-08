# QuiXAFS

A suite of standalone, interactive Python tools for raw data exploration, batch processing, and spectral analysis of Extended X-ray Absorption Fine-Structure (EXAFS) and X-ray Absorption Spectroscopy (XAS) data obtained from the **BM28 XMaS beamline** at the **ESRF**.

---

## 📋 System Requirements & Dependencies

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

## 🛠️ The Standalone Tools

### 1. QuiXAFS (`QuiXAFS.py`)
An interactive PyQt5 application for visualizing, calibrating, and performing multi-component fits of XAS/EXAFS datasets.

- **Features**:
  - **Premium UI**: Dark mode Charcoal styling.
  - **Dynamic File Format Support**: Loads standard 2D `.npy` files or single compressed `.npz` dataset packages containing average intensity, standard deviations, and energy axes.
  - **Dynamic Index Mapping**: Naturally links incident energy sweep sliders to the raw EDF spectrum orientation.
  - **Integrated ROI Integration Tool**: Draggable boundaries on the 1D spectrum plot to map specific emission lines.
  - **Emission Line Calibration**: Maps MCA channels to emission energy (eV) using IUPAC database lines.
  - **Advanced Fitting**: Solves multi-component models (Tb, Co, Fe, Cr) with background scatter corrections.
- **Dependency Files**: Requires [emission_lines.json](emission_lines.json) in the same directory to load reference database lines.
- **Run**:
  ```bash
  python QuiXAFS.py
  ```

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

---

### 3. Raw EDF Explorer (`raw_edf_explorer.py`)
An interactive explorer interface to inspect raw ESRF Data Format (EDF) files.

- **Features**:
  - Open any raw `.edf` binary file containing 2D MCA datasets.
  - Scroll through incident energy slices and plot individual MCA spectra.
  - Select regions of interest (ROI) and inspect raw detector counts.
- **Dependency Files**: Requires [data_loader.py](data_loader.py) in the same directory for EDF parsing.
- **Run**:
  ```bash
  python raw_edf_explorer.py
  ```
