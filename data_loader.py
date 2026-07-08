"""
data_loader.py

A library to load and process data from the BM28 XMaS beamline at ESRF.
It handles reading EDF binary files (ESRF Data Format containing MCA spectra)
and correlating them with their corresponding scan CSV files.
"""

import os
import re
import glob
import pandas as pd
import numpy as np

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

def get_matching_scans(csv_dir, zap_dir):
    """
    Finds scan files in the CSV directory and ZAP directory that correspond to the same scan numbers.
    Returns a dictionary mapping scan_number -> {'csv_path': path, 'edf_path': path}.
    """
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    edf_files = glob.glob(os.path.join(zap_dir, "*.edf"))
    
    scan_map = {}
    
    for f in csv_files:
        m = re.search(r"scan_(\d+)_", os.path.basename(f))
        if m:
            scan_num = int(m.group(1))
            scan_map[scan_num] = {'csv_path': f, 'edf_path': None}
            
    for f in edf_files:
        # Matches files like XAS_batch1_ZAP_raw_xia02_0018_0000_0000.edf
        m = re.search(r"xia\d+_(\d+)_0000_0000\.edf", os.path.basename(f))
        if m:
            scan_num = int(m.group(1))
            if scan_num in scan_map:
                scan_map[scan_num]['edf_path'] = f
                
    # Filter out scans that don't have both files
    complete_scans = {k: v for k, v in scan_map.items() if v['edf_path'] is not None}
    
    return complete_scans
