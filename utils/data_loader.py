"""
data_loader.py

A library to load and process data from the BM28 XMaS beamline at ESRF.
It handles reading EDF binary files (ESRF Data Format containing MCA spectra)
and correlating them with their corresponding SPEC scans (in-memory) or scan CSV files.
"""

import os
import re
import io
import glob
import pandas as pd
import numpy as np


class SpecFile:
    """
    In-memory parser and indexer for ESRF BM28 SPEC files.
    Allows high-performance, on-demand loading of individual scan data
    as pandas DataFrames without writing intermediate CSV files to disk.
    """
    def __init__(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"SPEC file not found: {filepath}")
        self.filepath = os.path.abspath(filepath)
        self.scans = {}  # scan_num -> {'cmd': str, 'labels': list, 'data_offset': int, 'date': str}
        self._index_file()

    def _index_file(self):
        """
        Quickly indexes scan headers and byte offsets for fast on-demand loading.
        """
        with open(self.filepath, 'rb') as f:
            current_scan = None
            current_cmd = ''
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith(b'#S '):
                    parts = line.decode('ascii', errors='ignore').strip().split(maxsplit=2)
                    try:
                        current_scan = int(parts[1])
                    except (ValueError, IndexError):
                        current_scan = None
                        continue
                    current_cmd = parts[2] if len(parts) > 2 else ''
                    self.scans[current_scan] = {
                        'cmd': current_cmd,
                        'date': '',
                        'labels': None,
                        'data_offset': None
                    }
                elif current_scan is not None:
                    if line.startswith(b'#D '):
                        self.scans[current_scan]['date'] = line.decode('ascii', errors='ignore')[3:].strip()
                    elif line.startswith(b'#L '):
                        self.scans[current_scan]['labels'] = line.decode('ascii', errors='ignore')[3:].split()
                        self.scans[current_scan]['data_offset'] = f.tell()

    def get_scan_numbers(self):
        """Returns sorted list of all scan numbers found in the SPEC file."""
        return sorted(list(self.scans.keys()))

    def get_scan_command(self, scan_num):
        """Returns the scan command string for a specific scan number."""
        return self.scans.get(scan_num, {}).get('cmd', '')

    def get_scan_date(self, scan_num):
        """Returns the date string for a specific scan number."""
        return self.scans.get(scan_num, {}).get('date', '')

    def get_scan_df(self, scan_num):
        """
        Loads the data table for a specific scan into a pandas DataFrame.
        Automatically converts 'zap_energy' from keV to eV (multiplied by 1000.0)
        matching the beamline processing pipeline.
        """
        if scan_num not in self.scans:
            raise KeyError(f"Scan {scan_num} not found in SPEC file {self.filepath}")
        
        info = self.scans[scan_num]
        if info['data_offset'] is None or not info['labels']:
            return pd.DataFrame()

        with open(self.filepath, 'rb') as f:
            f.seek(info['data_offset'])
            data_lines = []
            while True:
                line = f.readline()
                if not line or line.startswith(b'#S '):
                    break
                if not line.startswith(b'#') and line.strip():
                    data_lines.append(line.decode('ascii', errors='ignore').strip())

        if not data_lines:
            return pd.DataFrame(columns=info['labels'])

        csv_str = ' '.join(info['labels']) + '\n' + '\n'.join(data_lines)
        df = pd.read_csv(io.StringIO(csv_str), sep=r'\s+')
        if 'zap_energy' in df.columns:
            df['zap_energy'] = df['zap_energy'] * 1000.0
            
        return df


def parse_edf_header(header_bytes):
    """
    Parses the ASCII header from an EDF file.
    """
    header_str = header_bytes.decode('ascii', errors='ignore')
    # Clean the curly braces
    header_str = header_str.strip('{}\n\r ')
    
    header_dict = {}
    for line in header_str.split(';'):
        line = line.strip()
        if not line or '=' not in line:
            continue
        key, val = line.split('=', 1)
        header_dict[key.strip()] = val.strip()
        
    return header_dict


def read_edf(file_path):
    """
    Reads an EDF file, parses its header, and extracts the raw binary data
    as a 2D NumPy array of shape (Dim_2, Dim_1) representing (scan_points, mca_channels).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EDF file not found: {file_path}")
        
    with open(file_path, 'rb') as f:
        # Read the first 1024 bytes (standard header size for these files)
        header_bytes = f.read(1024)
        header = parse_edf_header(header_bytes)
        
        # Determine size and dimensions
        dim_1 = int(header.get('Dim_1', 4096))
        dim_2 = int(header.get('Dim_2', 4701))
        header_size = int(header.get('EDF_HeaderSize', 1024))
        
        # Read the binary data from the offset
        f.seek(header_size)
        
        # Check data type
        data_type = header.get('DataType', 'SignedInteger')
        if data_type == 'SignedInteger':
            dtype = '<i4' # 32-bit signed integer, little endian
        elif data_type == 'UnsignedInteger':
            dtype = '<u4'
        else:
            dtype = '<i4' # Fallback
            
        data = np.fromfile(f, dtype=dtype)
        
        # Reshape to (Dim_2, Dim_1)
        # Dim_2 represents the scan points, Dim_1 represents the MCA channels
        data_2d = data.reshape((dim_2, dim_1))
        
    return header, data_2d


def average_scans(file_paths):
    """
    Averages the binary data across multiple EDF files.
    Assumes all files have identical dimensions.
    Returns the averaged 2D NumPy array.
    """
    if not file_paths:
        raise ValueError("No files provided for averaging.")
        
    accumulated_data = None
    count = 0
    
    for path in file_paths:
        _, data_2d = read_edf(path)
        if accumulated_data is None:
            accumulated_data = data_2d.astype(np.float64)
        else:
            if accumulated_data.shape != data_2d.shape:
                raise ValueError(f"Shape mismatch: {path} has shape {data_2d.shape}, expected {accumulated_data.shape}")
            accumulated_data += data_2d
        count += 1
        
    return accumulated_data / count


def load_scan_csv(csv_path):
    """
    Loads a scan's CSV file containing energy and normalization counters.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_scan_dataframe(scan_info):
    """
    Convenience helper to retrieve the scan's pandas DataFrame from either
    an in-memory SpecFile instance, an explicit spec_df, or a CSV file path.
    """
    if 'spec_df' in scan_info and scan_info['spec_df'] is not None:
        return scan_info['spec_df']
    if 'spec_file' in scan_info and scan_info['spec_file'] is not None and 'scan_number' in scan_info:
        spec_obj = scan_info['spec_file']
        if isinstance(spec_obj, SpecFile):
            return spec_obj.get_scan_df(scan_info['scan_number'])
    if 'csv_path' in scan_info and scan_info['csv_path']:
        return load_scan_csv(scan_info['csv_path'])
    if 'csv' in scan_info and scan_info['csv']:
        return load_scan_csv(scan_info['csv'])
    raise ValueError(f"Unable to load DataFrame from scan_info: {scan_info}")


def get_matching_scans(source, zap_dir):
    """
    Finds scan files that correspond to the same scan numbers between the
    SPEC metadata (SPEC file or CSV directory) and the ZAP directory (EDF binary files).

    Parameters:
    -----------
    source : str, SpecFile, or dict
        Can be:
        - A path to a single SPEC file (e.g. 'BL-align-XAS_batch1.01', '*.spec', etc.)
        - A directory path containing individual scan CSV files (*.csv)
        - An already initialized SpecFile object
    zap_dir : str
        Directory path containing ZAP raw detector EDF files (*.edf)

    Returns:
    --------
    dict: mapping scan_number -> {
        'scan_number': int,
        'edf_path': str,
        'spec_file': SpecFile (if SPEC source),
        'csv_path': str (if CSV directory source),
        'command': str,
        # Backwards-compatible keys:
        'csv': str (or None),
        'edf': str,
    }
    """
    if not zap_dir or not os.path.isdir(zap_dir):
        return {}

    edf_files = glob.glob(os.path.join(zap_dir, "*.edf"))
    # Also search subdirectories of zap_dir if none found in root
    if not edf_files:
        edf_files = glob.glob(os.path.join(zap_dir, "**", "*.edf"), recursive=True)

    edf_map = {}
    for f in edf_files:
        m = re.search(r"xia\d+_(\d+)_0000_0000\.edf", os.path.basename(f))
        if not m:
            m = re.search(r"_(\d+)_0000_0000\.edf", os.path.basename(f))
        if not m:
            m = re.search(r"scan_(\d+)", os.path.basename(f))
        if m:
            scan_num = int(m.group(1))
            edf_map[scan_num] = f

    scan_map = {}

    # Case 1: source is a SpecFile instance
    if isinstance(source, SpecFile):
        spec_obj = source
        for s_num in spec_obj.get_scan_numbers():
            if s_num in edf_map:
                scan_map[s_num] = {
                    'scan_number': s_num,
                    'edf_path': edf_map[s_num],
                    'edf': edf_map[s_num],
                    'spec_file': spec_obj,
                    'csv_path': None,
                    'csv': None,
                    'command': spec_obj.get_scan_command(s_num)
                }

    # Case 2: source is a file path (SPEC file)
    elif isinstance(source, str) and os.path.isfile(source):
        spec_obj = SpecFile(source)
        for s_num in spec_obj.get_scan_numbers():
            if s_num in edf_map:
                scan_map[s_num] = {
                    'scan_number': s_num,
                    'edf_path': edf_map[s_num],
                    'edf': edf_map[s_num],
                    'spec_file': spec_obj,
                    'csv_path': None,
                    'csv': None,
                    'command': spec_obj.get_scan_command(s_num)
                }

    # Case 3: source is a directory path (CSV directory)
    elif isinstance(source, str) and os.path.isdir(source):
        csv_files = glob.glob(os.path.join(source, "*.csv"))
        for f in csv_files:
            m = re.search(r"scan_(\d+)_", os.path.basename(f))
            if not m:
                m = re.search(r"_(\d+)_", os.path.basename(f))
            if m:
                s_num = int(m.group(1))
                if s_num in edf_map:
                    scan_map[s_num] = {
                        'scan_number': s_num,
                        'edf_path': edf_map[s_num],
                        'edf': edf_map[s_num],
                        'spec_file': None,
                        'csv_path': f,
                        'csv': f,
                        'command': os.path.basename(f)
                    }

    return scan_map
