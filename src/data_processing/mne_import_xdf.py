import gzip
import logging
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from .mne_write_edf import write_edf
import mne
import numpy as np
from pyxdf import load_xdf
logger = logging.getLogger()
def open_xdf(filename):
    """Open XDF file for reading."""
    filename = Path(filename)  # convert to pathlib object
    if filename.suffix == ".xdfz" or filename.suffixes == [".xdf", ".gz"]:
        f = gzip.open(filename, "rb")
    else:
        f = open(filename, "rb")
    if f.read(4) != b"XDF:":  # magic bytes
        raise IOError("Invalid XDF file {}".format(filename))
    return f
def match_streaminfos(stream_infos, parameters):
    """Find stream IDs matching specified criteria.
    Parameters
    ----------
    stream_infos : list of dicts
        List of dicts containing information on each stream. This information
        can be obtained using the function resolve_streams.
    parameters : list of dicts
        List of dicts containing key/values that should be present in streams.
        Examples: [{"name": "Keyboard"}] matches all streams with a "name"
                  field equal to "Keyboard".
                  [{"name": "Keyboard"}, {"type": "EEG"}] matches all streams
                  with a "name" field equal to "Keyboard" and all streams with
                  a "type" field equal to "EEG".
    """
    matches = []
    for request in parameters:
        for info in stream_infos:
            for key in request.keys():
                match = info[key] == request[key]
                if not match:
                    break
            if match:
                matches.append(info["stream_id"])
    return list(set(matches))  # return unique values
def resolve_streams(fname):
    """Resolve streams in given XDF file.
    Parameters
    ----------
    fname : str
        Name of the XDF file.
    Returns
    -------
    stream_infos : list of dicts
        List of dicts containing information on each stream.
    """
    return parse_chunks(parse_xdf(fname))
def parse_xdf(fname):
    """Parse and return chunks contained in an XDF file.
    Parameters
    ----------
    fname : str
        Name of the XDF file.
    Returns
    -------
    chunks : list
        List of all chunks contained in the XDF file.
    """
    chunks = []
    with open_xdf(fname) as f:
        for chunk in _read_chunks(f):
            chunks.append(chunk)
    return chunks
def _read_chunks(f):
    """Read and yield XDF chunks.
    Parameters
    ----------
    f : file handle
        File handle of XDF file.
    Yields
    ------
    chunk : dict
        XDF chunk.
    """
    while True:
        chunk = dict()
        try:
            chunk["nbytes"] = _read_varlen_int(f)
        except EOFError:
            return
        chunk["tag"] = struct.unpack("<H", f.read(2))[0]
        if chunk["tag"] in [2, 3, 4, 6]:
            chunk["stream_id"] = struct.unpack("<I", f.read(4))[0]
            if chunk["tag"] == 2:  # parse StreamHeader chunk
                xml = ET.fromstring(f.read(chunk["nbytes"] - 6).decode())
                chunk = {**chunk, **_parse_streamheader(xml)}
            else:  # skip remaining chunk contents
                f.seek(chunk["nbytes"] - 6, 1)
        else:
            f.seek(chunk["nbytes"] - 2, 1)  # skip remaining chunk contents
        yield chunk
def _parse_streamheader(xml):
    """Parse stream header XML."""
    return {el.tag: el.text for el in xml if el.tag != "desc"}
def parse_chunks(chunks):
    """Parse chunks and extract information on individual streams."""
    streams = []
    for chunk in chunks:
        if chunk["tag"] == 2:  # stream header chunk
            streams.append(
                dict(
                    stream_id=chunk["stream_id"],
                    name=chunk.get("name"),  # optional
                    type=chunk.get("type"),  # optional
                    source_id=chunk.get("source_id"),  # optional
                    created_at=chunk.get("created_at"),  # optional
                    uid=chunk.get("uid"),  # optional
                    session_id=chunk.get("session_id"),  # optional
                    hostname=chunk.get("hostname"),  # optional
                    channel_count=int(chunk["channel_count"]),
                    channel_format=chunk["channel_format"],
                    nominal_srate=int(chunk["nominal_srate"]),
                )
            )
    return streams
def read_raw_xdf(fname, stream_id=None):
    """Read XDF file.
    Parameters
    ----------
    fname : str
        Name of the XDF file.
    stream_id : int | str | None
        ID (number) or name of the stream to load (optional). If None, the
        first stream of type "EEG" will be read.
    Returns
    -------
    raw : mne.io.Raw
        XDF file data.
    """
    streams, header = load_xdf(fname)
    if stream_id is not None:
        if isinstance(stream_id, str):
            stream = _find_stream_by_name(streams, stream_id)
        elif isinstance(stream_id, int):
            stream = _find_stream_by_id(streams, stream_id)
    else:
        stream = _find_stream_by_type(streams, stream_type="EEG")
    if 'footer' in stream.keys():
        if stream['footer']['info']['sample_count'][0] != '0':
            if stream is not None:
                name = stream["info"]["name"][0]
                n_chans = int(stream["info"]["channel_count"][0])
                # fs = float(stream["info"]["nominal_srate"][0])
                # NOTE: Have used calculated rate
                # fs = float(stream["info"]["effective_srate"])
                fs = 256
                logger.info(
                    f"Found EEG stream '{name}' ({n_chans} channels, " f"sampling rate {fs}Hz)."
                )
                labels, units, types = _get_ch_info(stream)
                if not labels:
                    labels = [str(n) for n in range(n_chans)]
                if not units:
                    units = ["NA" for _ in range(n_chans)]
                info = mne.create_info(ch_names=labels, sfreq=fs, verbose=False)
                # convert from microvolts to volts if necessary
                scale = np.array([1e-6 if u == "microvolts" else 1 for u in units])
                raw = mne.io.RawArray((stream["time_series"] * scale).T, info)
                first_samp = stream["time_stamps"][0]
            else:
                logger.info("No EEG stream found.")
                return
    markers = _find_stream_by_type(streams, stream_type="Markers")
    if markers is not None:
        onsets = markers["time_stamps"] - first_samp
        logger.info(f"Adding {len(onsets)} annotations.")
        descriptions = markers["time_series"]
        annotations = mne.Annotations(onsets, [0] * len(onsets), descriptions)
        raw.set_annotations(annotations)
    if 'footer' in stream.keys():
        if stream['footer']['info']['sample_count'][0] == '0':
            raw = []
    if 'footer' not in stream.keys():
        raw = []
    return raw, stream["time_stamps"]

def _find_stream_by_name(streams, stream_name):
    """Find the first stream that matches the given name."""
    for stream in streams:
        if stream["info"]["name"][0] == stream_name:
            return stream

def _find_stream_by_id(streams, stream_id):
    """Find the stream that matches the given ID."""
    for stream in streams:
        if stream["info"]["stream_id"] == stream_id:
            return stream

def _find_stream_by_type(streams, stream_type="EEG"):
    """Find the first stream that matches the given type."""
    for stream in streams:
        if stream["info"]["type"][0] == stream_type:
            return stream

def _tobii_ch_info():
    channels = []
    eye_channels = [
        "dev_time_stamp",
        "avg_x",
        "avg_y",
        "avg_pupil_dia",
        "avg_eye_pos_x",
        "avg_eye_pos_y",
        "avg_eye_pos_z",
        "avg_eye_dist",
        "eye_valid",
        "dev_timestamp",
        "sys_time_stamp",
        "l_disp_area_x",
        "l_disp_area_y",
        "l_user_x",
        "l_user_y",
        "l_user_z",
        "l_valid",
        "l_pupil_dia",
        "l_pupil_valid",
        "l_or_user_x",
        "l_or_user_y",
        "l_or_user_z",
        "l_or_track_x",
        "l_or_track_y",
        "l_or_track_z",
        "l_or_valid",
        "r_disp_area_x",
        "r_disp_area_y",
        "r_user_x",
        "r_user_y",
        "r_user_z",
        "r_valid",
        "r_pupil_dia",
        "r_pupil_valid",
        "r_or_user_x",
        "r_or_user_y",
        "r_or_user_z",
        "r_or_track_x",
        "r_or_track_y",
        "r_or_track_z",
        "r_or_valid",
        "r_pixel_x",
        "r_pixel_y",
        "l_pixel_x",
        "l_pixel_y",
    ]
    for ch in eye_channels:
        channels.append({"label": [ch], "type": ["misc"], "unit": [None]})
    return channels

def _b_alert_ch_info():
    channels = []
    time_channels = ["Epoch", "Offset","Hour", "Min","Sec","MilliSec"]
    for ch in time_channels:
        channels.append({"label": [ch], "type": ["time"], "unit": ["microvolts"]})
    # EEG channels
    eeg_channels = [
        "Fp1",
        "F7",
        "F8",
        "T4",
        "T6",
        "T5",
        "T3",
        "Fp2",
        "O1",
        "P3",
        "Pz",
        "F3",
        "Fz",
        "F4",
        "C4",
        "P4",
        "POz",
        "C3",
        "Cz",
        "O2",
    ]
    for ch in eeg_channels:
        channels.append({"label": [ch], "type": ["EEG"], "unit": ["microvolts"]})
    # ECG channel
    channels.append({"label": ["ECG"], "type": ["ECG"], "unit": ["microvolts"]})
    # Auxillary channels
    aux_channels = ["AUX1", "AUX2", "AUX3"]
    for ch in aux_channels:
        channels.append({"label": [ch], "type": ["AUX"], "unit": ["microvolts"]})
    return channels

def _get_ch_info(stream):
    # print(stream["info"]["desc"])
    # print(stream["info"]["desc"][0]["channels"])
    labels, units, types = [], [], []
    if stream["info"]["desc"]:
        if stream["info"]["type"][0] == "Eye tacking":
            channels = _tobii_ch_info()
        else:
            # channels = stream["info"]["desc"][0]["channels"][0]["channel"]
            channels = _b_alert_ch_info()
        for ch in channels:
            labels.append(str(ch["label"][0]))
            # types.append(ch["type"][0])
            units.append("microvolts")
    return labels, units, types

def _read_varlen_int(f):
    """Read a variable-length integer."""
    nbytes = f.read(1)
    if nbytes == b"\x01":
        return ord(f.read(1))
    elif nbytes == b"\x04":
        return struct.unpack("<I", f.read(4))[0]
    elif nbytes == b"\x08":
        return struct.unpack("<Q", f.read(8))[0]
    elif not nbytes:  # EOF
        raise EOFError
    else:
        raise RuntimeError("Invalid variable-length integer encountered.")