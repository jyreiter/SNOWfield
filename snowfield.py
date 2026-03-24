
# ============================================================
# snowfield.py
#
# SMU–NORSAR Observational Wavefields
# Utilities for DAS, SmartSolo, and array processing workflows.
# Designed for reuse in notebooks and scripts.
#
# Excludes spectral estimation and transfer function logic.
# ============================================================

# ==========================================================
# Imports
# ==========================================================

import copy
import glob
import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta

from scipy import signal
from scipy.signal import welch, csd, coherence
from scipy.ndimage import median_filter
from scipy.signal.windows import tukey, hann
import pywt

import utm
import simpledas

from obspy import read, Stream, UTCDateTime
from obspy import read_inventory
from obspy.core import AttribDict
from obspy.geodetics import gps2dist_azimuth

from geopy.distance import geodesic
from scipy.spatial import cKDTree


from dataclasses import dataclass
from typing import Optional, Literal

import warnings

# ==========================================================
# DAS Data Loading, Preprocessing, and Visualization
# ==========================================================

# Reading in the DAS data
def load_das_data(
    t1: UTCDateTime,
    t2: UTCDateTime,
    base_dir: str,
    integrate: bool = False,
    sensitivitySelect: int = -1,
    buffer_seconds: int = 10
) -> pd.DataFrame:
    """
    Load and process DAS data from HDF5 files within a time window.
    
    Parameters
    ----------
    t1 : UTCDateTime
        Start time of the time window
    t2 : UTCDateTime
        End time of the time window
    base_dir : str
        Base directory containing HDF5 files
    integrate : bool, optional
        Whether to integrate the DAS data (default: False)
    sensitivitySelect : int, optional
        Sensitivity selection parameter for simpledas (default: -1)
    buffer_seconds : int, optional
        Buffer in seconds to search before t1 and after t2 (default: 10)
    
    Returns
    -------
    dfdas : pd.DataFrame
        Processed DAS data in strain rate units with sequential integer column names,
        trimmed to the exact time window [t1, t2].
        The DataFrame includes a .meta attribute with DAS metadata.
    """
    # Find all HDF5 files in the directory
    all_files = glob.glob(os.path.join(base_dir, '*.hdf5'))
    
    # Extract times from filenames and filter to those within the buffered window
    hdf5_files = []
    t1_buffered = t1 - buffer_seconds
    t2_buffered = t2 + buffer_seconds
    
    # Get the date for constructing full datetime from filename
    base_date = t1.date
    
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        # Extract time from filename (format: HHMMSS.hdf5)
        match = re.match(r'(\d{6})\.hdf5', file_name)
        if match:
            time_str = match.group(1)
            try:
                # Construct full datetime from base_date and time string
                file_time = UTCDateTime(
                    f"{base_date.year}-{base_date.month:02d}-{base_date.day:02d}T"
                    f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                )
                
                # Include files that overlap with the buffered time window
                # Each file represents data starting at that time (typically 10s duration)
                if file_time >= t1_buffered and file_time <= t2_buffered:
                    hdf5_files.append(file_path)
            except:
                # Skip files with invalid time formats
                continue
    
    # Sort files by time to ensure proper order
    hdf5_files.sort()
    
    if not hdf5_files:
        raise FileNotFoundError(
            f"No HDF5 files found in {base_dir} for time window {t1} to {t2}"
        )
    
    # Loading the DAS data
    dfdas = simpledas.load_DAS_files(hdf5_files, integrate=integrate, sensitivitySelect=sensitivitySelect)
    
    # Rename columns to sequential integers starting from 0 (to be consistent with metadata)
    dfdas.columns = range(len(dfdas.columns))
    
    # Extract sensitivity and convert data to strain rate
    sensitivity = dfdas.meta['sensitivities'][0][0]    # This is the sensitivity in units of rad/(strain*m)
    dfdas = dfdas / sensitivity    # Converts data to strain rate (assuming integrate=False above)
    
    # Trim the dataframe to the exact requested time window
    dfdas = dfdas[(dfdas.index >= t1.datetime) & (dfdas.index <= t2.datetime)]
    
    return dfdas

# Preprocessing the DAS data
def das_preproc(dfdas, detrend=True, rmv_mode_noise=True, filter='bandpass', taper='tukey', 
                fmin=0.5, fmax=5.0, order=4, spatial_med=True, kernel_size=7):
    """
    Description:
        Fast, vectorized function to preprocess DAS data
    
    Operations:
      1) Detrend each channel
      2) Remove common-mode noise (time-wise median)
      3) Apply taper
      4) Temporal Butterworth filtering
      5) Spatial median filtering (across channels)
    
    Parameters:
    dfdas : loaded DAS dataframe from simpledas as the raw data to preprocess
    detrend : bool, optional
        Whether to detrend the data (default is True)
    rmv_mode_noise : bool, optional
        Whether to remove common mode noise (default is True)
    filter : str, optional
        Type of filter to apply as bandpass, low-, high-pass or spatial median (default is 'bandpass')
    taper : str, optional
        Type of taper to apply (default is 'tukey')
    fmin : float, optional
        Minimum frequency for filtering (default is 0.5 Hz)
    fmax : float, optional
        Maximum frequency for filtering (default is 5.0 Hz)
    order : int, optional
        Order of the Butterworth filter (default is 4)
    kernel_size : int, optional
        Kernel size for spatial median filter (default is 7)
    """    

    # Copy once to avoid mutating caller data
    dfdas = dfdas.copy()

    # Work in NumPy for speed
    data = dfdas.values  # shape: (ntime, nch)

    # ------------------------------------------------------------------
    # 1. Detrend (per channel)
    # ------------------------------------------------------------------
    if detrend:
        data = signal.detrend(data, axis=0)

    # ------------------------------------------------------------------
    # 2. Remove common-mode noise (median across channels at each time)
    # ------------------------------------------------------------------
    if rmv_mode_noise:
        data = data - np.median(data, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # 3. Sampling rate
    # ------------------------------------------------------------------
    if hasattr(dfdas, "meta") and "dt" in dfdas.meta:
        fs = 1.0 / dfdas.meta["dt"]
    else:
        fs = 1.0 / dfdas.index.freq.delta.total_seconds()

    # ------------------------------------------------------------------
    # 4. Temporal filtering
    # ------------------------------------------------------------------
    if filter in {"bandpass", "lowpass", "highpass"}:

        if filter == "bandpass":
            b, a = signal.butter(
                order, [fmin, fmax], btype="bandpass", fs=fs
            )
        elif filter == "lowpass":
            b, a = signal.butter(
                order, fmax, btype="lowpass", fs=fs
            )
        elif filter == "highpass":
            b, a = signal.butter(
                order, fmin, btype="highpass", fs=fs
            )

        # -----------------------------
        # Taper
        # -----------------------------
        ntime = data.shape[0]
        if taper == "tukey":
            taper_win = tukey(ntime, alpha=0.01)
        elif taper == "hann":
            taper_win = hann(ntime)
        else:
            taper_win = np.ones(ntime)

        # Apply taper and filter (vectorized)
        data *= taper_win[:, None]
        data = signal.filtfilt(b, a, data, axis=0)

    # ------------------------------------------------------------------
    # 5. Spatial median filtering (across channels)
    # ------------------------------------------------------------------
    if spatial_med:
        kernel_size = kernel_size | 1  # force odd
        data = median_filter(
            data,
            size=(1, kernel_size),
            mode="nearest",
        )

    # ------------------------------------------------------------------
    # Write back to DataFrame
    # ------------------------------------------------------------------
    dfdas[:] = data
    return dfdas

# Plotting a waterfall-like image of the DAS data
def plot_das_image(dfdas, percentile_clip=60, cmap='RdBu_r', ax=None):
    """
    Plot DAS data as an image for visual inspection only.

    Parameters
    ----------
    dfdas : pandas.DataFrame
        DAS data (time index x channels)
    percentile_clip : float
        Percentile for symmetric color scaling
    cmap : str
        Matplotlib colormap
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    vmax = np.percentile(np.abs(dfdas.values), percentile_clip)
    vmin = -vmax

    times_num = mdates.date2num(dfdas.index)

    im = ax.imshow(
        dfdas.values.T,
        aspect='auto',
        cmap=cmap,
        extent=[times_num[0], times_num[-1],
                dfdas.columns[0], dfdas.columns[-1]],
        origin='lower',
        vmin=vmin,
        vmax=vmax,
        interpolation='none'
    )

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Channel')
    ax.set_title('DAS Image (Visualization Only)')

    plt.colorbar(im, ax=ax, label='Relative Strain Rate')

    return ax

# Functions to plot channel-level trances of the DAS data
def aggregate_das_channels(
    dfdas: pd.DataFrame,
    center_channel: int,
    n_channels: int,
    method: str = 'mean' or 'median'
):
    """
    Aggregate DAS channels around a center channel.
    
    IMPORTANT:
    - Aggregation is allowed ONLY for visualization and noise reduction
    - Raw channel data must be retained for spectral analysis if needed
    """
    channels = dfdas.columns.tolist()
    idx = channels.index(center_channel)
    half = n_channels // 2
    
    sel = channels[max(0, idx - half): min(len(channels), idx + half)]
    data = dfdas[sel].values
    
    if method == 'median':
        agg = np.median(data, axis=1)
    elif method == 'mean':
        agg = np.mean(data, axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}; use 'median' or 'mean'.")
    
    mad = np.median(np.abs(data - agg[:, None]), axis=1)
    return agg, mad, sel

def pandas_times_relative_to_reference(datetime_index, ref_time):
    """
    Convert pandas DateTimeIndex to seconds relative to a reference time.
    
        Parameters
    ----------
    datetime_index : pandas.DatetimeIndex
    ref_time : UTCDateTime
        
    Returns
    -------
    np.ndarray
        Time vector in seconds
    """
    return np.array([
        (UTCDateTime(t.to_pydatetime()) - ref_time)
        for t in datetime_index
    ])
    
def plot_das_channels(
    dfdas: pd.DataFrame,
    channels: list,
    time_rel: np.ndarray,
    normalize: bool = True,
    xlim: tuple = None,
    figsize: tuple = (14, 10),
    title: Optional[str] = None
):
    """
    Plot individual DAS channels.
    
    Parameters
    ----------
    dfdas : pd.DataFrame
        DAS data with channels as columns
    channels : list of int
        Channel numbers to plot
    time_rel : np.ndarray
        Time vector relative to reference (seconds)
    normalize : bool, optional
        Normalize each channel by its peak amplitude (default: True)
    xlim : tuple, optional
        (tmin, tmax) time limits
    figsize : tuple, optional
        Figure size (default: (14, 10))
    title : Optional[str], optional
        Figure title (default: 'DAS Channel-Level Waveforms')
    !!!WARNING!!!: Normalization destroys physical amplitudes.
    Use ONLY for timing and waveform-shape comparison.
    """
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, sharex=True, figsize=figsize)
    
    if n_channels == 1:
        axes = [axes]
    
    for ax, ch in zip(axes, channels):
        d_plot = dfdas[ch].values.copy()
        
        if normalize:
            peak = np.max(np.abs(d_plot))
            if peak > 0:
                d_plot /= peak
        
        ax.plot(time_rel, d_plot, 'k-', linewidth=0.7)
        ax.set_ylabel(f'Channel {ch}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    if xlim is not None:
        axes[-1].set_xlim(xlim)
    
    axes[-1].set_xlabel('Time (s) relative to reference')
    
    title = 'DAS Channel-Level Waveforms' if title is None else title
    if normalize:
        title += ' (Normalized — No Amplitude Meaning)'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
# ==========================================================
# Sapphire Data Loading, Preprocessing, and Visualization
# ==========================================================

# Functions to apply location and metadata
def get_array_coords(st, ref_station=None):
    '''
    Returns the array coordinates for an array, in km with respect to the reference array provided
    
    Inputs:
    st - ObsPy Stream object containing array data
    ref_station - A String containing the name of the reference station (optional)
                 If not provided, uses the first station in the stream
    
    Outputs:
    X - [Nx2] NumPy array of array coordinates in km
    stnm - [Nx1] list of element names
    
    Stephen Arrowsmith (sarrowsmith@smu.edu)
    '''
    
    X = np.zeros((len(st), 2))
    stnm = []
    for i in range(0, len(st)):
        #print(st[i].stats.station, st[i].stats.sac.stla, st[i].stats.sac.stlo)
        E, N, _, _ = utm.from_latlon(st[i].stats.sac.stla, st[i].stats.sac.stlo)
        X[i,0] = E; X[i,1] = N
        stnm.append(st[i].stats.station)

    # Use first station as reference if not specified
    if ref_station is None:
        ref_station = stnm[0]
    
    # Adjusting to the reference station, and converting to km:
    ref_station_ix = np.where(np.array(stnm) == ref_station)[0][0]    # index of reference station
    X[:,0] = (X[:,0] - X[ref_station_ix,0])
    X[:,1] = (X[:,1] - X[ref_station_ix,1])
    X = X/1000.

    return X, stnm

def dms_to_decimal(coord_str):
    """
    Convert coordinates from degrees, minutes, seconds to decimal degrees
    Input format: 'N 60 44' 6.4892"' or 'E 11 32' 21.0378"'
    """
    # Remove extra quotes and spaces
    coord_str = coord_str.strip().replace('"', '')
    
    # Parse the coordinate string
    # Pattern: Direction Degrees Minutes Seconds
    pattern = r'([NSEW])\s+(\d+)\s+(\d+)\'\s+([\d.]+)'
    match = re.match(pattern, coord_str)
    
    if not match:
        return None
    
    direction, degrees, minutes, seconds = match.groups()
    
    # Convert to decimal
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    
    # Apply direction (negative for South and West)
    if direction in ['S', 'W']:
        decimal = -decimal
    
    return decimal

def sapphire_location(df_locations, st):
    """
    Function to assign station locations from a location dataframe to an ObsPy Stream object.
    Parameters:
    df_locations : pandas DataFrame
        DataFrame containing station location information with columns 'Station', 'Latitude', 'Longitude', 'Elevation'
    st : obspy Stream object
        Stream object containing station traces to which locations will be assigned
    """
    # Create a dictionary mapping station number to location
    station_locations = {}

    for idx, row in df_locations.iterrows():
        station_id = str(row['sensor'])  # Convert to string for consistency
        lat_decimal = dms_to_decimal(row['lat'])
        lon_decimal = dms_to_decimal(row['lon'])
        elevation = float(row['elev'])
        
        # Store as decimal degrees
        station_locations[station_id] = {
            'latitude': lat_decimal,
            'longitude': lon_decimal,
            'elevation': elevation
        }
    
    # Adding location data to the ObsPy Stream:
    for tr in st:
        try:
            sacAttrib = AttribDict({"stla": station_locations[tr.stats.station[2:]]['latitude'],
                                    "stlo": station_locations[tr.stats.station[2:]]['longitude'],
                                    "stel": station_locations[tr.stats.station[2:]]['elevation']})
            tr.stats.sac = sacAttrib
        except:
            print(tr.stats.station + " not found in station locations")
            
    return st

def stat_2_event(st, station='SM325', evla=None, evlo=None):
    """
    Function to compute distance and azimuth from a station to an event location.
    
    Parameters:
    st : obspy Stream object
        Stream object containing station traces with SAC headers.
    station : str
        Station name to compute distance and azimuth from.
    evla : float
        Event latitude in decimal degrees.
    evlo : float
        Event longitude in decimal degrees.
    """
    # Get station location
    station = st.select(station=station)[0]
    station_lat = station.stats.sac.stla
    station_lon = station.stats.sac.stlo

    # Calculate distance and azimuths
    dist_m, az, baz = gps2dist_azimuth(station_lat, station_lon, evla, evlo)

    # Convert distance to km
    dist_km = dist_m / 1000.0

    print(f"Station {station.stats.station} location: {station_lat:.6f}°N, {station_lon:.6f}°E")
    print(f"Event location: {evla:.6f}°N, {evlo:.6f}°E")
    print(f"Great-circle distance: {dist_km:.2f} km ({dist_m:.0f} m)")
    print(f"Backazimuth (station to event): {az:.2f}°")
    print(f"Azimuth (event to station): {baz:.2f}°")

# Reading in the Sapphire data
def load_sapphire_data(
    t1: UTCDateTime,
    t2: UTCDateTime,
    base_dir: str,
    sub_paths: list,
    df_locations: pd.DataFrame,
    channel: str = "BDF",
    stations_to_remove: list = None,
) -> Stream:
    """ 
    Load Sapphire infrasound data into an ObsPy Stream with station locations attached.
    
    Parameters
    ----------  
    t1, t2 : UTCDateTime
        Start and end times for data loading
    base_dir : str
        Root directory containing Sapphire data along fiber subdirectories
    sub_paths : list of str
        List of subdirectory names corresponding to different fiber segments (e.g., '/Fiber_A/Sapphires')
    df_locations : pd.DataFrame
        DataFrame with sensor locations (sensor, lat, lon, elev)
    channel : str, optional
        Channel code to read (default: "BDF")
    stations_to_remove : list of str, optional
        Stations lacking metadata or known to be bad
        
    Returns
    -------
    st : obspy.Stream
        Loaded Sapphire data with SAC headers populated (stla, stlo, stel)
    """
    
    st = Stream()
    
    # Determine file date convention (files roll at 04:00 UTC)
    file_date = t1 if t1.hour >= 4 else t1 - 86400
    date_string = file_date.strftime('%y%m%d')
    
    for sub_path in sub_paths:
        fiber_path = base_dir + sub_path  # Direct concatenation like your working code
        
        # FIX: Added '/*' to match your working pattern
        sensors = [path.split('/')[-1] for path in glob.glob(fiber_path + '/*')]
        
        for sensor in sensors:
            file_pattern = f"{fiber_path}/{sensor}/S{sensor}*{date_string}*.msd"
            try:
                st_temp = read(file_pattern, starttime=t1, endtime=t2)
                st_temp = st_temp.select(channel=channel)
                for tr in st_temp:
                    st.append(tr)
            except Exception as e:
                print(f"Warning: Could not read data for sensor {sensor}: {e}")
                continue
            
    st.trim(t1, t2)
    
    # Remove stations without metadata
    if stations_to_remove is not None:
        for sta in stations_to_remove:
            try:
                st.remove(st.select(station=sta)[0])
            except Exception:
                pass
        
    # Attach station metadata
    st = sapphire_location(df_locations, st)
    return st
    
def load_sapphire_data_by_fiber(
    t1: UTCDateTime,
    t2: UTCDateTime,
    base_dir: str,
    sensors_by_fiber: dict,
    df_locations: pd.DataFrame,
    channel: str = "BDF"
) -> Stream:
    """
    Load Sapphire infrasound data into an ObsPy Stream with station locations.

    Parameters
    ----------
    t1, t2 : UTCDateTime
        Start and end times for data loading.
    base_dir : str
        Root folder containing Fiber_X/Sapphires folders.
    sensors_by_fiber : dict
        Dictionary mapping fiber keys ('A','B',...) to sensor list (e.g., ['SM384', ...]).
    df_locations : pd.DataFrame
        DataFrame containing sensor metadata (station, lat, lon, elev).
    channel : str
        Channel code to read (default: "BDF").
    
    Returns
    -------
    st : obspy.Stream
        Loaded Sapphire data with metadata.
    """
    
    st = Stream()
    
    # Determine the file date convention (files roll at 04:00 UTC)
    file_date = t1 if t1.hour >= 4 else t1 - 86400
    date_string = file_date.strftime('%y%m%d')
    
    for fiber_key, sensors in sensors_by_fiber.items():
        sub_path = f"Fiber_{fiber_key}/Sapphires"
        fiber_path = os.path.join(base_dir, sub_path)
        
        # Convert sensor names from 'SM384' → '384' to match folder names
        sensors_num = [s.replace("SM", "") for s in sensors]
        print(f"🔹 Fiber {fiber_key}: loading {len(sensors_num)} sensors")
        
        for sensor in sensors_num:
            sensor_path = os.path.join(fiber_path, sensor)
            
            if not os.path.isdir(sensor_path):
                print(f"     ⚠️ Sensor folder missing: {sensor_path}")
                continue
            
            # File pattern: S384_R001_250810_N001.msd
            file_pattern = os.path.join(sensor_path, f"S{sensor}*{date_string}*.msd")
            matches = glob.glob(file_pattern)
            
            if len(matches) == 0:
                print(f"     ⚠️ No files matched: {file_pattern}")
                continue
            else:
                print(f"     Found {len(matches)} file(s) for sensor {sensor}")
            
            for fname in matches:
                try:
                    st_temp = read(fname, starttime=t1, endtime=t2)
                    st_temp = st_temp.select(channel=channel)
                    if len(st_temp) > 0:
                        st += st_temp
                except Exception as e:
                    print(f"       ⚠️ Could not read {fname}: {e}")
    
    # Trim to exact time window
    st.trim(t1, t2)
    
    # Attach station metadata
    st = sapphire_location(df_locations, st)
    
    print(f"\n✅ Finished loading. Total traces: {len(st)}")
    return st
    
def summarize_stream(st: Stream):
    """ 
    Print a lightweight summary of a STream for sanity checking
    """
    print(f"Number of traces: {len(st)}")
    print(f"Start time: {st[0].stats.starttime}")
    print(f"End time: {st[0].stats.endtime}")
    print(f"Sampling rate: {st[0].stats.sampling_rate} Hz")
    
    stations = sorted({tr.stats.station for tr in st})
    print(f"Stations ({len(stations)}): {stations[:6]}{'...' if len(stations) > 6 else ''}")
    
def plot_stream_section(
    st: Stream,
    evloc: tuple = None,  # (lat, lon) for record section
    ref_time: UTCDateTime = None,  # reference time for x-axis
    normalize: bool = True,
    spacing: float = 1.0,
    min_spacing: float = 0.5,  # NEW: minimum spacing for record sections
    scale_distances: float = 1.0,  # NEW: scale factor for distance differences
    linewidth: float = 0.5,
    color: str = 'k',
    title: str = None,
    figsize: tuple = (12, 8),
    vlines: list = None,
    vline_color: str = 'red',
    vline_style: str = '-',
    vline_width: float = 0.5,
    show_station_labels: bool = False,
    amplitude_scale: float = 0.8  # NEW: scale factor for trace amplitudes
):
    """
    Plot waveform section with time in seconds on x-axis.
    
    Can plot either a regular section (sequential traces) or a record section
    (sorted by distance from event location).
    
    Parameters
    ----------
    st : obspy.Stream
        Input waveforms with SAC headers (stla, stlo) if using record section
    evloc : tuple, optional
        (latitude, longitude) of event for record section plotting.
        If provided, traces are sorted by distance from event and y-axis shows distance.
        Requires traces to have SAC headers with station coordinates.
    ref_time : UTCDateTime, optional
        Reference time for x-axis. If None, uses earliest trace start time.
    normalize : bool
        Normalize each trace by its peak absolute value
    spacing : float
        Vertical spacing between traces (for regular section)
    min_spacing : float
        Minimum vertical spacing for record sections (km). Ensures traces don't overlap
        even when stations are physically close together.
    scale_distances : float
        Multiplicative factor to amplify distance differences in record sections.
        Values > 1.0 spread traces apart more.
    linewidth : float
        Line width for plotting
    color : str
        Line color for plotting
    title : str, optional
        Plot title. If None, generates automatic title.
    figsize : tuple
        Figure size (width, height)
    vlines : list, optional
        List of x values (in seconds) to draw vertical lines
    vline_color : str
        Color for vertical lines
    vline_style : str
        Line style for vertical lines ('-', '--', ':', etc.)
    vline_width : float
        Line width for vertical lines
    show_station_labels : bool
        Whether to show station names on y-axis (for regular section only)
    amplitude_scale : float
        Scaling factor for trace amplitudes (0-1). Lower values make traces smaller
        and reduce overlap. Default 0.8 gives some breathing room.
        
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    
    Examples
    --------
    # Record section with better spacing
    fig, ax = plot_stream_section(
        st, 
        evloc=(60.73, 11.54),
        min_spacing=1.0,  # Enforce 1 km minimum separation
        scale_distances=2.0,  # Spread traces 2x more than actual distances
        amplitude_scale=0.6,  # Make traces 60% of full size
        title='Record Section',
        vlines=[147, 163]
    )
    """
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Set reference time
    if ref_time is None:
        ref_time = min([tr.stats.starttime for tr in st])
    
    # Prepare trace-distance pairs if record section
    if evloc is not None:
        trace_dist_pairs = []
        for tr in st:
            try:
                sta_lat = tr.stats.sac.stla
                sta_lon = tr.stats.sac.stlo
                dist_m = geodesic_distance_m((sta_lat, sta_lon), evloc)
                dist_km = dist_m / 1000.0
                trace_dist_pairs.append((tr, dist_km))
            except AttributeError:
                warnings.warn(f"Station {tr.stats.station} missing SAC coordinates, skipping")
                continue
        
        # Sort by distance
        trace_dist_pairs.sort(key=lambda x: x[1])
        
        # Apply distance scaling and minimum spacing enforcement
        adjusted_pairs = []
        for i, (tr, actual_dist) in enumerate(trace_dist_pairs):
            if i == 0:
                # First trace stays at its actual distance
                plot_dist = actual_dist * scale_distances
            else:
                # Ensure minimum spacing from previous trace
                prev_plot_dist = adjusted_pairs[-1][1]
                candidate_dist = actual_dist * scale_distances
                plot_dist = max(candidate_dist, prev_plot_dist + min_spacing)
            
            adjusted_pairs.append((tr, plot_dist, actual_dist))
        
        traces_to_plot = adjusted_pairs
        is_record_section = True
    else:
        # Regular section: just enumerate
        traces_to_plot = [(tr, i * spacing, None) for i, tr in enumerate(st)]
        is_record_section = False
    
    # Plot each trace
    station_names = []
    actual_distances = []
    for item in traces_to_plot:
        if is_record_section:
            tr, plot_offset, actual_dist = item
            actual_distances.append(actual_dist)
        else:
            tr, plot_offset, _ = item
        
        # Calculate time in seconds relative to reference time
        times_sec = tr.times() + (tr.stats.starttime - ref_time)
        data = tr.data.copy()
        
        # Normalize if requested
        if normalize:
            max_amp = abs(data).max()
            if max_amp > 0:
                data = data / max_amp  # Normalize to [-1, 1]
        
        # Apply amplitude scaling to prevent overlap
        data = data * amplitude_scale
        
        # Plot with offset
        ax.plot(times_sec, data + plot_offset, color=color, linewidth=linewidth)
        station_names.append(tr.stats.station)
    
    # Add vertical lines if requested
    if vlines is not None:
        for vline_x in vlines:
            ax.axvline(x=vline_x, color=vline_color, linestyle=vline_style, linewidth=vline_width)
    
    # Customize the plot
    ax.set_xlabel('Time (seconds)', fontsize=12)
    
    if evloc is not None:
        # Record section
        ax.set_ylabel('Distance (km)', fontsize=12)
        if title is None:
            title = f'Record Section (Event: {evloc[0]:.3f}°N, {evloc[1]:.3f}°E)'
        
        # Add note if spacing was adjusted
        dist_range = f"{actual_distances[0]:.2f} - {actual_distances[-1]:.2f} km"
        if min_spacing > 0 or scale_distances != 1.0:
            note = f"\n(Actual distance range: {dist_range}"
            if scale_distances != 1.0:
                note += f", scaled {scale_distances}x"
            if min_spacing > 0:
                note += f", min spacing {min_spacing} km"
            note += ")"
            title += note
    else:
        # Regular section
        ax.set_ylabel('Trace Number' if not show_station_labels else 'Station', fontsize=12)
        if show_station_labels:
            ax.set_yticks([i * spacing for i in range(len(st))])
            ax.set_yticklabels(station_names, fontsize=8)
        else:
            ax.set_yticks([i * spacing for i in range(len(st))])
            ax.set_yticklabels([])
        if title is None:
            title = 'Waveform Section'
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print(f"Displayed {len(traces_to_plot)} traces")
    if evloc is not None:
        print(f"Actual distance range: {actual_distances[0]:.2f} - {actual_distances[-1]:.2f} km")
        if min_spacing > 0 or scale_distances != 1.0:
            print(f"Adjusted for visualization (scale={scale_distances}x, min_spacing={min_spacing} km)")
    
    return fig, ax

# ==========================================================
# Sapphire Data Loading, Preprocessing, and Visualization
# ==========================================================

# Functions to read in the SmartSolo data
def parse_starttime_from_filename(fname):
    """
    Extract start time from SmartSolo MiniSEED filename.

    Example:
    453021777.0001.2025.08.07.13.09.30.000.Z.miniseed
    """
    parts = os.path.basename(fname).split('.')
    return UTCDateTime(
        f"{parts[2]}-{parts[3]}-{parts[4]} "
        f"{parts[5]}:{parts[6]}:{parts[7]}"
    )

def overlaps_window(t_file, t1, t2, file_len=86400):
    """
    Fast overlap test to avoid reading non-overlapping files.
    """
    return not (t_file > t2 or (t_file + file_len) < t1)

def extract_serial_from_filename(fname: str) -> int:
    serial_full = os.path.basename(fname).split(".")[0]  # e.g. 453025444
    return int(serial_full[-5:])                         # e.g. 25444

def _normalize_components(channel: str) -> set[str]:
    """
    Accepts: '*', '*Z', 'Z', 'HHZ', '*N', etc.
    Returns one or more of {'Z','N','E'}.
    """
    channel = channel.strip().upper()
    if channel == "*":
        return {"Z", "N", "E"}

    # If last char is one of Z/N/E, use that
    if channel and channel[-1] in {"Z", "N", "E"}:
        return {channel[-1]}

    # Fallback: all
    return {"Z", "N", "E"}

def load_smartsolo_data(
    t1: UTCDateTime,
    t2: UTCDateTime,
    base_dir: str,
    channel: str = "*Z",
    file_len_est: float = 86400.0,
    serial_whitelist: list[int] | None = None,
    verbose: bool = False,
) -> Stream:
    """
    Load SmartSolo MiniSEED data in [t1, t2], filtered by component and optional serial whitelist.
    """

    components = _normalize_components(channel)

    # Build file list by component suffix in filename: .N.miniseed / .E.miniseed / .Z.miniseed
    file_list = []
    for comp in sorted(components):
        file_list.extend(glob.glob(os.path.join(base_dir, f"*.{comp}.miniseed")))
    file_list = sorted(file_list)

    st = Stream()
    n_kept = 0
    n_skipped_serial = 0
    n_skipped_time = 0

    for fname in file_list:
        try:
            if serial_whitelist is not None:
                serial = extract_serial_from_filename(fname)
                if serial not in serial_whitelist:
                    n_skipped_serial += 1
                    continue

            t_file = parse_starttime_from_filename(fname)
            if not overlaps_window(t_file, t1, t2, file_len_est):
                n_skipped_time += 1
                continue

            st_tmp = read(fname, starttime=t1, endtime=t2)
            for tr in st_tmp:
                st.append(tr)
                n_kept += 1

        except Exception as e:
            if verbose:
                print(f"Warning: could not read {fname}: {e}")
            continue

    if len(st) > 0:
        st.trim(t1, t2)

    if verbose:
        print(f"Components: {sorted(components)}")
        print(f"Files considered: {len(file_list)}")
        print(f"Skipped by serial: {n_skipped_serial}")
        print(f"Skipped by time: {n_skipped_time}")
        print(f"Traces loaded: {n_kept}")

    return st

# Preprocessing the SmartSolo data
def smartsolo_preproc(
    st: Stream,
    rmv_response: bool = False,
    inventory_path: str = None,
    detrend: bool = True,
    taper: bool = True,
    filter: str | None = 'bandpass',
    fmin: float = 0.5,
    fmax: float = 20.0,
    order: int = 4,
) -> Stream:
    """
    Light preprocessing for SmartSolo seismic data.

    Operations
    ----------
    1) Optional instrument response removal
    2) Detrend
    3) Optional taper
    4) Optional Butterworth filtering

    Notes
    -----
    No amplitude normalization or AGC is applied.
    This preserves physical amplitudes for coherence
    and transfer-function estimation.
    """

    st = st.copy()

    for tr in st:
        if rmv_response and inventory_path is not None:
            inv = read_inventory(inventory_path)
            tr = tr.remove_response(inventory=inv, output='VEL', pre_filt = [0.01, 0.05, 100, 125], water_level=60, plot=False)
        if detrend:
            tr.detrend("linear")

        if taper:
            tr.taper(max_percentage=0.01, type="hann")

        if filter is not None:
            if filter == "bandpass":
                tr.filter(
                    "bandpass",
                    freqmin=fmin,
                    freqmax=fmax,
                    corners=order,
                    zerophase=True,
                )
            elif filter == "lowpass":
                tr.filter(
                    "lowpass",
                    freq=fmax,
                    corners=order,
                    zerophase=True,
                )
            elif filter == "highpass":
                tr.filter(
                    "highpass",
                    freq=fmin,
                    corners=order,
                    zerophase=True,
                )

    return st

# Attaching the SAC headers with station locations
def attach_sac_locations(
    st: Stream,
    station_locations: dict,
    station_key_slice=slice(1, None),
    inplace: bool = False,
    verbose: bool = True,
):
    """
    Attach SAC-compatible location metadata (stla, stlo, stel)
    to an ObsPy Stream.

    Parameters
    ----------
    st : obspy.Stream
        Input stream (SmartSolo or Sapphire)
    station_locations : dict
        Dictionary keyed by station ID containing:
        {
            'latitude': float,
            'longitude': float,
            'elevation': float
        }
    station_key_slice : slice
        Slice applied to tr.stats.station to match station_locations keys.
        Default removes leading character (e.g., 'S24354' -> '24354').
    inplace : bool
        If True, modify stream in place. Otherwise return a copy.
    verbose : bool
        Print warnings for missing stations.

    Returns
    -------
    st_out : obspy.Stream
        Stream with SAC headers attached where possible
    """

    st_out = st if inplace else st.copy()

    missing = []

    for tr in st_out:
        sta_raw = tr.stats.station
        sta_key = sta_raw[station_key_slice]

        try:
            loc = station_locations[sta_key]
            tr.stats.sac = AttribDict(
                stla=loc["latitude"],
                stlo=loc["longitude"],
                stel=loc["elevation"],
            )
        except KeyError:
            missing.append(sta_raw)

    if missing and verbose:
        warnings.warn(
            f"{len(missing)} stations missing SAC metadata: "
            f"{sorted(set(missing))}",
            RuntimeWarning
        )

    return st_out

# ==========================================================
# Collocation Functions
# ==========================================================

def extract_stream_coords(stream):
    """
    Extract station names and coordinates from ObsPy stream SAC headers.
    Skips traces without SAC info.
    Returns names (list) and coords (Nx2 array of (lat, lon)).
    """
    names = []
    coords = []

    for tr in stream:
        try:
            lat = tr.stats.sac.stla
            lon = tr.stats.sac.stlo
            # Skip traces with NaNs
            if np.isnan(lat) or np.isnan(lon):
                continue
            names.append(tr.stats.station)
            coords.append([lat, lon])
        except AttributeError:
            # This trace has no .stats.sac, skip it
            print(f"Skipping {tr.stats.station}: no SAC header")
            continue

    return names, np.array(coords)

def extract_das_coords(df_cal, arm="A"):
    """
    Extract DAS channel coordinates for a specific arm.
    Returns df with 'channel', 'latitude', 'longitude'.
    """
    if arm.lower() == "all":
        dfs = []
        for a in ["A", "B", "C", "D", "E"]:
            dfs.append(extract_das_coords(df_cal, arm=a))
        return pd.concat(dfs, ignore_index=True)

    start_idx = df_cal[df_cal["note"] == f"Arm{arm}_1chn"].index[0]
    end_idx   = df_cal[df_cal["note"] == f"Arm{arm}_2chn"].index[0]
    df_arm = df_cal.loc[start_idx:end_idx].copy()
    return df_arm[["channel", "latitude", "longitude"]]

def geodesic_distance_m(latlon_a, latlon_b):
    """
    Compute geodesic distance between two (lat, lon) points in meters.
    """
    return geodesic(latlon_a, latlon_b).meters

def find_nearest_point(target_coord, coords, labels=None):
    """
    Find nearest coordinate to a target point using geodesic distance.

    Parameters
    ----------
    target_coord : tuple
        (lat, lon)
    coords : list of tuples
        List of (lat, lon)
    labels : list, optional
        Labels corresponding to coords

    Returns
    -------
    dict
        {
          'index': int,
          'label': label or None,
          'distance_m': float
        }
    """
    distances = [geodesic_distance_m(target_coord, c) for c in coords]
    idx = int(np.argmin(distances))

    return {
        'index': idx,
        'label': labels[idx] if labels is not None else None,
        'distance_m': distances[idx]
    }
    
def build_collocated_triplets(solo_names, solo_coords,
                              sap_names, sap_coords,
                              das_channels, das_coords,
                              verbose=False):
    """
    Build triplets by finding the nearest Sapphire and DAS points for each SmartSolo.
    Uses geodesic distances for accuracy.
    Returns a pandas DataFrame.
    
    Parameters
    ----------
    solo_names : list of str
        Names of SmartSolo stations
    solo_coords : array-like (Nx2)
        SmartSolo coordinates (lat, lon)
    sap_names : list of str
        Names of Sapphire stations
    sap_coords : array-like (Nx2)
        Sapphire coordinates (lat, lon)
    das_channels : list
        DAS channel identifiers
    das_coords : array-like (Nx2)
        DAS coordinates (lat, lon)
    verbose : bool
        Print progress if True
    """
    triplets = []

    for i, ss_name in enumerate(solo_names):
        ss_coord = solo_coords[i]

        # Find nearest Sapphire
        nearest_sap = find_nearest_point(ss_coord, sap_coords, sap_names)
        sap_coord = sap_coords[nearest_sap['index']]

        # Find nearest DAS
        nearest_das = find_nearest_point(ss_coord, das_coords, das_channels)
        das_coord = das_coords[nearest_das['index']]

        triplets.append({
            'smartsolo': ss_name,
            'smartsolo_lat': ss_coord[0],
            'smartsolo_lon': ss_coord[1],

            'sapphire': nearest_sap['label'],
            'sapphire_lat': sap_coord[0],
            'sapphire_lon': sap_coord[1],
            'sapphire_dist_m': nearest_sap['distance_m'],

            'das_channel': nearest_das['label'],
            'das_lat': das_coord[0],
            'das_lon': das_coord[1],
            'das_dist_m': nearest_das['distance_m']
        })

        if verbose:
            print(f"{ss_name} → Sapphire: {nearest_sap['label']} ({nearest_sap['distance_m']:.2f} m), "
                  f"DAS: {nearest_das['label']} ({nearest_das['distance_m']:.2f} m)")

    return pd.DataFrame(triplets)

def rotate_to_fiber(east, north, azimuth_deg):
    """
    Rotate horizontal components (E, N) into fiber-aligned (P, O).

    Parameters
    ----------
    east, north : np.ndarray
        East and North component data arrays.
    azimuth_deg : float
        Fiber azimuth in degrees (clockwise from North).

    Returns
    -------
    fiber_parallel, fiber_perp : np.ndarray, np.ndarray
        Parallel (P) and perpendicular (O) components.
    """
    a = np.deg2rad(azimuth_deg)
    fiber_parallel = north * np.cos(a) + east * np.sin(a)
    fiber_perp = -north * np.sin(a) + east * np.cos(a)
    return fiber_parallel, fiber_perp

def rotate_smartsolo_to_fiber(
    st_solo_orient,
    azimuth_csv_path,
    smartsolo_col="smartsolo",
    azimuth_col="fiber_azimuth_deg",
    include_vertical=True,
):
    """
    Rotate SmartSolo N/E traces into fiber-parallel (P) and fiber-perp (O) traces.

    Parameters
    ----------
    st_solo_orient : obspy.Stream
        Input stream containing station traces with N/E (and optionally Z) channels.
    azimuth_csv_path : str
        Path to CSV containing station azimuth metadata.
    smartsolo_col : str
        CSV column name for station code.
    azimuth_col : str
        CSV column name for fiber azimuth (deg clockwise from North).
    include_vertical : bool
        If True, include original Z trace (copied) in output stream.

    Returns
    -------
    st_rot : obspy.Stream
        Rotated stream with channels ending in:
        - P : fiber-parallel
        - O : fiber-perpendicular
        Plus Z if include_vertical=True and available.
        
    Usage
    -----
    st_rot = rotate_smartsolo_to_fiber(
        st_solo_orient=st_solo_orient,
        azimuth_csv_path="/Users/49573102/Documents/Infrasound Work/Norway/collocated_triplets_with_azimuth.csv",
    )
    print(st_rot)
    """
    meta = pd.read_csv(azimuth_csv_path)
    meta[smartsolo_col] = meta[smartsolo_col].astype(str).str.strip()
    azimuth_map = dict(zip(meta[smartsolo_col], meta[azimuth_col]))

    st_rot = Stream()

    for sta in sorted({tr.stats.station for tr in st_solo_orient}):
        if sta not in azimuth_map:
            continue

        st_sta = st_solo_orient.select(station=sta)
        tr_n_sel = st_sta.select(channel="*N")
        tr_e_sel = st_sta.select(channel="*E")
        tr_z_sel = st_sta.select(channel="*Z")

        if len(tr_n_sel) == 0 or len(tr_e_sel) == 0:
            continue

        tr_n = tr_n_sel[0].copy()
        tr_e = tr_e_sel[0].copy()

        # Ensure matching sample lengths if needed
        npts = min(len(tr_n.data), len(tr_e.data))
        north = tr_n.data[:npts]
        east = tr_e.data[:npts]

        az = float(azimuth_map[sta])
        par, perp = rotate_to_fiber(east=east, north=north, azimuth_deg=az)

        tr_par = tr_n.copy()
        tr_perp = tr_e.copy()

        tr_par.data = par
        tr_perp.data = perp
        tr_par.stats.npts = len(par)
        tr_perp.stats.npts = len(perp)

        # Rename channels: last char -> P/O
        tr_par.stats.channel = tr_par.stats.channel[:-1] + "P"
        tr_perp.stats.channel = tr_perp.stats.channel[:-1] + "O"

        st_rot += tr_par
        st_rot += tr_perp

        if include_vertical and len(tr_z_sel):
            st_rot += tr_z_sel[0].copy()

    return st_rot
    
def extract_triplet_waveforms(triplet, st_solo, st_sap, dfdas, n_das_channels=20, manual_channels=None):
    """
    Extract SmartSolo, Sapphire, and DAS waveforms for a single triplet.
    DAS waveform returns median and MAD across selected channels.
    """
    # SmartSolo
    tr_ss = st_solo.select(station=triplet["smartsolo"])[0]

    # Sapphire
    tr_sap = st_sap.select(station=triplet["sapphire"])[0]

    # DAS channel selection
    center_ch = int(triplet["das_channel"])
    all_ch = dfdas.columns.tolist()

    if manual_channels is not None:
        sel_ch = manual_channels
    else:
        half = n_das_channels // 2
        idx = all_ch.index(center_ch)
        sel_ch = all_ch[max(0, idx-half):min(len(all_ch), idx+half)]

    das_data = dfdas[sel_ch].values
    das_median = np.median(das_data, axis=1)
    das_mad = np.median(np.abs(das_data - das_median[:, None]), axis=1)

    return {
        "smartsolo": tr_ss,
        "sapphire": tr_sap,
        "das_channels": sel_ch,
        "das_median": das_median,
        "das_mad": das_mad
    }

# ==========================================================
# Power Spectral Density (Single and Beamformed)
# ==========================================================
def compute_psd_trace(tr: Trace, nperseg: int = 2048):
    """
    Compute PSD for a single trace.
    """
    fs = tr.stats.sampling_rate
    f, Pxx = welch(tr.data.astype(float), fs=fs, nperseg=nperseg)
    return f, Pxx


def compute_psd_stream(st: Stream, method="median", nperseg=2048):
    """
    Compute PSD across a stream (array-level PSD).

    Parameters
    ----------
    method : 'median' or 'mean'
    """
    psds = []
    for tr in st:
        f, P = compute_psd_trace(tr, nperseg=nperseg)
        psds.append(P)

    psds = np.array(psds)

    if method == "median":
        P_out = np.median(psds, axis=0)
    elif method == "mean":
        P_out = np.mean(psds, axis=0)
    else:
        raise ValueError

    return f, P_out


def plot_psd(f, P, label=None, ax=None, loglog=True):
    if ax is None:
        fig, ax = plt.subplots()

    if loglog:
        ax.loglog(f, P, label=label)
    else:
        ax.plot(f, P, label=label)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    if label:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    return ax

# ==========================================================
# Frequency-Wavenumber Analysis
# ==========================================================
def stream_to_array_data(st: Stream):
    """
    Convert stream to matrix + coordinates for FK.
    """
    st = st.copy()
    st.sort()

    # Ensure equal lengths
    npts = min([tr.stats.npts for tr in st])
    data = np.array([tr.data[:npts] for tr in st])

    # Coordinates
    x_km, stations = get_array_coords(st)
    x = x_km[:, 0]
    y = x_km[:, 1]

    fs = st[0].stats.sampling_rate

    return data, x, y, fs

def fk_analysis(data, x_km, y_km, fs, fmin, fmax, smax=0.4, ngrid=51):
    """
    Wide-band F-K analysis.

    Parameters
    ----------
    data   : (N_sta, N_samp) array, pre-filtered waveforms
    x_km   : (N_sta,) East offsets from centroid [km]
    y_km   : (N_sta,) North offsets from centroid [km]
    fs     : sampling rate [Hz]
    fmin, fmax : frequency band [Hz]
    smax   : slowness grid limit [s/km]
    ngrid  : number of grid points per axis

    Returns
    -------
    sx_vec, sy_vec : 1-D slowness axes [s/km]
    power_grid     : (ngrid, ngrid) absolute beam power
    semblance_grid : (ngrid, ngrid) semblance in [0,1]
    """
    N_sta, N_samp = data.shape

    # ── Detrend each trace (critical for unfiltered data) ─────────────────────
    # Copy to avoid modifying original, then remove linear trend
    data = data.copy()
    for i in range(N_sta):
        # Remove best-fit line
        x = np.arange(N_samp)
        coeffs = np.polyfit(x, data[i], 1)
        data[i] -= np.polyval(coeffs, x)
    
    # ── FFT ────────────────────────────────────────────────────────────────────
    NFFT = N_samp
    freqs = np.fft.rfftfreq(NFFT, d=1.0/fs)          # shape: (NFFT//2+1,)
    U = np.fft.rfft(data, n=NFFT, axis=1)             # shape: (N_sta, NFFT//2+1)

    # ── Frequency mask ─────────────────────────────────────────────────────────
    f_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_sel = freqs[f_mask]                          # selected frequencies
    U_sel = U[:, f_mask]                               # (N_sta, N_f)
    df = freqs[1] - freqs[0]

    # ── Incoherent (denominator) power ─────────────────────────────────────────
    incoherent_power = np.sum(np.abs(U_sel)**2) * df / N_sta   # scalar

    # ── Slowness grid ──────────────────────────────────────────────────────────
    sx_vec = np.linspace(-smax, smax, ngrid)           # E-W slowness
    sy_vec = np.linspace(-smax, smax, ngrid)           # N-S slowness
    SX, SY = np.meshgrid(sx_vec, sy_vec)               # (ngrid, ngrid)

    power_grid     = np.zeros((ngrid, ngrid))
    semblance_grid = np.zeros((ngrid, ngrid))

    # ── Phase shift matrix  (N_sta, N_f, ngrid, ngrid)  ── too large!
    # Use vectorised approach over frequency, loop over slowness grid
    # For each frequency bin build the phase matrix:
    #   phase[i, gx, gy] = 2π f (sx[gx,gy] * x[i] + sy[gx,gy] * y[i])

    # Pre-compute position outer products with the flattened slowness vectors
    # shapes: sx_flat (ngrid²,), x_km (N_sta,)
    sx_flat = SX.ravel()   # (ngrid²,)
    sy_flat = SY.ravel()   # (ngrid²,)

    # delay[i, g] = sx_flat[g]*x[i] + sy_flat[g]*y[i]  →  (N_sta, ngrid²)
    delay = np.outer(x_km, sx_flat) + np.outer(y_km, sy_flat)  # (N_sta, ngrid²)

    beam_power_flat = np.zeros(ngrid * ngrid)

    for k, f in enumerate(freqs_sel):
        # Steering phase for all stations and all grid points
        steer = np.exp(1j * 2 * np.pi * f * delay)    # (N_sta, ngrid²)
        # Beam: sum over stations
        # U_sel[:, k] shape: (N_sta,)
        beam = (U_sel[:, k, np.newaxis] * steer).sum(axis=0) / N_sta  # (ngrid²,)
        beam_power_flat += np.abs(beam)**2 * df

    power_grid     = beam_power_flat.reshape(ngrid, ngrid)
    semblance_grid = power_grid / incoherent_power

    return sx_vec, sy_vec, power_grid, semblance_grid

def run_fk(
    st: Stream,
    fmin=0.5,
    fmax=5.0,
    smax=0.4,
    ngrid=51
):
    """
    High-level FK interface.
    """
    data, x, y, fs = stream_to_array_data(st)

    sx, sy, power, semblance = fk_analysis(
        data, x, y, fs, fmin, fmax,
        smax=smax, ngrid=ngrid
    )

    # Peak
    idx = np.unravel_index(np.argmax(power), power.shape)
    sx_peak = sx[idx[1]]
    sy_peak = sy[idx[0]]

    slow = np.sqrt(sx_peak**2 + sy_peak**2)
    vapp = 1 / slow if slow > 0 else np.inf
    baz = (np.degrees(np.arctan2(sx_peak, sy_peak)) + 180) % 360

    return {
        "sx": sx,
        "sy": sy,
        "power": power,
        "semblance": semblance,
        "sx_peak": sx_peak,
        "sy_peak": sy_peak,
        "vapp": vapp,
        "baz": baz,
    }

def add_velocity_circles(ax, sx_vec, sy_vec, velocities, color='white', lw=0.8, ls='--'):
    """Overplot constant-apparent-velocity circles on a slowness map."""
    for v in velocities:
        s_ref = 1.0 / v  # slowness magnitude = 1/velocity
        theta = np.linspace(0, 2*np.pi, 360)
        cx = s_ref * np.sin(theta)
        cy = s_ref * np.cos(theta)
        # only plot if circle is inside the grid
        if s_ref <= sx_vec.max():
            ax.plot(cx, cy, color=color, lw=lw, ls=ls)
            # label at top of circle
            ax.text(0, s_ref*1.02, f'{v:.2f} km/s',
                    ha='center', va='bottom', fontsize=7, color=color)


def plot_fk(sx_vec, sy_vec, grid, title='F-K Power',
            cmap='jet', vmin_db=-10, vmax_db=0,
            half_contour_db=None,
            ref_velocities=None,
            peak_sx=None, peak_sy=None,
            use_db=True, vmin_linear=0, vmax_linear=1):
    """
    Plot a slowness map (power or semblance).

    Parameters
    ----------
    grid           : 2-D array on (sy, sx) axes
    half_contour_db: if not None, draw contour at this dB level (e.g. -3 or -6)
    use_db         : if True, plot in dB scale; if False, use linear scale
    vmin_linear, vmax_linear : limits for linear scale (when use_db=False)
    """
    if use_db:
        grid_plot = 10 * np.log10(grid / grid.max() + 1e-30)
        vmin, vmax = vmin_db, vmax_db
        cbar_label = 'Relative Power  [dB]'
    else:
        grid_plot = grid
        vmin, vmax = vmin_linear, vmax_linear
        cbar_label = 'Semblance'

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.pcolormesh(sx_vec, sy_vec, grid_plot,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='auto')
    cb = fig.colorbar(im, ax=ax, label=cbar_label)

    # Contours
    if use_db:
        # Standard dB contours (exclude half_contour_db to avoid overlap)
        levels_db = np.arange(vmin_db, 0, 1)
        if half_contour_db is not None and half_contour_db in levels_db:
            levels_db = levels_db[levels_db != half_contour_db]
        cs = ax.contour(sx_vec, sy_vec, grid_plot, levels=levels_db,
                        colors='white', linewidths=0.5, alpha=0.5)

        # Half-power / half-amplitude contour
        if half_contour_db is not None:
            # Check if this level exists in the data
            if grid_plot.min() <= half_contour_db <= grid_plot.max():
                cs_half = ax.contour(sx_vec, sy_vec, grid_plot,
                                     levels=[half_contour_db],
                                     colors='black', linewidths=1.2, linestyles='--', zorder=10)
                label_str = f'{half_contour_db} dB'
                if half_contour_db == -3:
                    label_str += '  (½ power)'
                elif half_contour_db == -6:
                    label_str += '  (½ amplitude / ¼ power)'
                # Label the contour
                ax.clabel(cs_half, fmt=label_str, fontsize=8, inline=True)
    else:
        # Linear contours for semblance
        levels_linear = np.arange(0.1, 1.0, 0.1)
        cs = ax.contour(sx_vec, sy_vec, grid_plot, levels=levels_linear,
                        colors='white', linewidths=0.5, alpha=0.5)

    # Velocity reference circles
    if ref_velocities:
        add_velocity_circles(ax, sx_vec, sy_vec, ref_velocities)

    # Peak marker
    if peak_sx is not None:
        ax.plot(peak_sx, peak_sy, 'w+', ms=12, mew=2, zorder=10)

    ax.set_xlabel('Slowness $s_x$ (E–W)  [s/km]')
    ax.set_ylabel('Slowness $s_y$ (N–S)  [s/km]')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.axvline(0, color='k', lw=0.5, ls=':')
    ax.invert_xaxis()  # Invert x-axis so East (+) is right, West (-) is left
    ax.invert_yaxis()  # Invert y-axis to match standard map projection 
    plt.tight_layout()
    return fig

def sliding_fk(
    st: Stream,
    win_len: float,
    step: float,
    fmin=0.5,
    fmax=5.0,
    smax=0.4,
    ngrid=51,
):
    """
    Sliding window FK detection.

    Returns list of detections:
        time, baz, velocity, semblance
    """

    t0 = min(tr.stats.starttime for tr in st)
    t1 = max(tr.stats.endtime for tr in st)

    results = []

    t = t0
    while t + win_len <= t1:
        st_win = st.copy().trim(t, t + win_len)

        try:
            fk = run_fk(st_win, fmin, fmax, smax, ngrid)

            results.append({
                "time": t,
                "baz": fk["baz"],
                "vapp": fk["vapp"],
                "semblance": np.max(fk["semblance"])
            })

        except Exception:
            pass

        t += step

    return pd.DataFrame(results)

def plot_sliding_fk_results(df, baz_range=(0, 360), vapp_range=(0, 5), semblance_threshold=0.5):
    """
    Plot sliding FK results as a scatter plot of backazimuth through time, with points
    colored by semblance. Only points above the semblance threshold are shown.
    
    A reference beamformed trace is plotted above the scatter plot, showing the semblance over time, with a horizontal line indicating the threshold.
    """
    fig, (ax_scatter, ax_beam) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                              gridspec_kw={'height_ratios': [3, 1]})

    # Scatter plot of detections
    df_detect = df[df['semblance'] >= semblance_threshold]
    sc = ax_scatter.scatter(df_detect['time'], df_detect['baz'], c=df_detect['semblance'],
                            cmap='viridis', vmin=semblance_threshold, vmax=1.0, edgecolor='k')
    ax_scatter.set_ylabel('Backazimuth (°)')
    ax_scatter.set_title('Sliding FK Detections')
    ax_scatter.set_ylim(baz_range)
    ax_scatter.grid(True)

    # Colorbar for scatter plot
    cbar = plt.colorbar(sc, ax=ax_scatter, label='Semblance')

    # Beamformed trace (semblance over time)
    ax_beam.plot(df['time'], df['semblance'], color='blue')
    ax_beam.axhline(semblance_threshold, color='red', linestyle='--', label='Threshold')
    ax_beam.set_xlabel('Time')
    ax_beam.set_ylabel('Semblance')
    ax_beam.set_ylim(vapp_range)
    ax_beam.legend()
    ax_beam.grid(True)

    plt.tight_layout()
    return fig

# ==========================================================
# Spectrograms
# ==========================================================
def compute_spectrogram(tr: Trace, nfft=256):
    f, t, Sxx = signal.spectrogram(
        tr.data.astype(float),
        fs=tr.stats.sampling_rate,
        nperseg=nfft,
        noverlap=nfft // 2,
    )
    return f, t, np.log10(Sxx + 1e-12)


def plot_spectrogram(tr: Trace, nfft=256, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    f, t, Sxx = compute_spectrogram(tr, nfft=nfft)

    t = t + (tr.stats.starttime - tr.stats.starttime)

    pcm = ax.pcolormesh(t, f, Sxx, shading="auto", cmap="nipy_spectral")
    ax.set_ylabel("Freq (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([0.5, f.max()])

    plt.colorbar(pcm, ax=ax, label="log10(Power)")
    return ax

# ==========================================================
# Scalogram
# ==========================================================
def compute_scalogram(
    tr: Trace,
    wavelet="morl",
    n_freqs=64,
    fmin=0.5,
    fmax=20,
):
    dt = tr.stats.delta
    freqs = np.linspace(fmin, fmax, n_freqs)

    scales = pywt.central_frequency(wavelet) / (freqs * dt)

    coeffs, _ = pywt.cwt(
        tr.data.astype(float),
        scales,
        wavelet,
        sampling_period=dt,
    )

    power = np.log10(np.abs(coeffs)**2 + 1e-12)

    t = np.arange(tr.stats.npts) * dt

    return t, freqs, power


def plot_scalogram(tr: Trace, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    t, f, P = compute_scalogram(tr)

    pcm = ax.pcolormesh(t, f, P, shading="auto", cmap="nipy_spectral")
    ax.set_ylabel("Freq (Hz)")
    ax.set_xlabel("Time (s)")

    plt.colorbar(pcm, ax=ax, label="log10(CWT Power)")
    return ax
    
