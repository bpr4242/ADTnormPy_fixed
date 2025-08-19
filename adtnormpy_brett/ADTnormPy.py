# adtnorm_bridge.py

import os
import sys
import tempfile
import shutil
from typing import Union, Optional, List, Dict

import numpy as np
import pandas as pd
import anndata

try:
    import mudata
except ImportError:
    class _EmptyModule:
        def __init__(self):
            self.MuData = type(None)
    mudata = _EmptyModule()

import rpy2.robjects
import rpy2.robjects.pandas2ri
import rpy2.robjects.packages
from joblib import Parallel, delayed

# ======================================================================================
# Global R package handle (one import per Python process; each libpython worker gets one)
# ======================================================================================

_ADTNORM_R = None

def _load_adtnorm_R(verbose: int = 1):
    """
    Load ADTnorm exactly once per Python process.
    Each joblib process will get its own interpreter & R session,
    so this caches per-worker, not globally across all workers.
    """
    global _ADTNORM_R
    if _ADTNORM_R is not None:
        return _ADTNORM_R

    try:
        ADTnormR = rpy2.robjects.packages.importr('ADTnorm')
    except rpy2.robjects.packages.PackageNotInstalledError:
        _install_adtnorm_R()
        ADTnormR = rpy2.robjects.packages.importr('ADTnorm')

    _ADTNORM_R = ADTnormR
    if verbose > 1:
        # Only once per worker
        print(rpy2.robjects.r('sessionInfo()'))
    return _ADTNORM_R


def _install_adtnorm_R():
    """
    Fallback installer for ADTnorm via devtools::install_github.
    Not automatically called unless ADTnorm isn't found.
    """
    res = input(f"Are you sure you would like to install ADTnorm into R at: {os.environ.get('R_HOME','<unknown R_HOME>')} ? "
                "Type anything to continue...")
    try:
        devtools = rpy2.robjects.packages.importr('devtools')
    except rpy2.robjects.packages.PackageNotInstalledError:
        utils = rpy2.robjects.packages.importr('utils')
        print("Installing devtools prior to installing ADTnorm. This may take a while...")
        utils.install_packages('devtools', quiet=True, quick=True, upgrade_dependencies=True, keep_source=False)
        devtools = rpy2.robjects.packages.importr('devtools')

    print("Installing ADTnorm and dependencies. This may take a while...")
    devtools.install_github("yezhengSTAT/ADTnorm", build_vignettes=False, quiet=True)


# ======================================================================================
# Utilities: logging, optional R sink capture, and file printing
# ======================================================================================

def _log(msg: str, marker: str = None, ts: bool = False):
    if marker:
        msg = f"[{marker}] {msg}"
    if ts:
        from datetime import datetime
        msg = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _print_r_file(path: str, marker: str):
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    _log(f"R> {line}", marker=marker)
    except Exception as e:
        _log(f"(could not read R sink file {path}: {e})", marker=marker)


def _r_capture_context(stdout_path: str, stderr_path: str):
    """
    Context manager that redirects R stdout/stderr to files via sink().
    Disabled by default in workers to avoid 'sink stack is full' errors,
    but available if you want to capture R chatter for debugging.
    """
    import contextlib
    import rpy2.robjects as ro

    @contextlib.contextmanager
    def _ctx():
        file_fun = ro.r["file"]
        sink_fun = ro.r["sink"]
        close_fun = ro.r["close"]
        sink_number = ro.r["sink.number"]

        # PRE-DRAIN any existing sinks (defensive)
        try:
            while int(sink_number(type="output")[0]) > 0:
                sink_fun(None, type="output")
            while int(sink_number(type="message")[0]) > 0:
                sink_fun(None, type="message")
        except Exception:
            pass

        # Open file connections explicitly
        out_con = file_fun(stdout_path, open="wt")
        err_con = file_fun(stderr_path, open="wt")

        # Sink both streams
        sink_fun(out_con, type="output")
        sink_fun(err_con, type="message")
        try:
            yield
        finally:
            # Always unwind all sinks
            try:
                while int(sink_number(type="output")[0]) > 0:
                    sink_fun(None, type="output")
                while int(sink_number(type="message")[0]) > 0:
                    sink_fun(None, type="message")
            finally:
                # Close file connections
                try: close_fun(out_con)
                except Exception: pass
                try: close_fun(err_con)
                except Exception: pass

    return _ctx()


# ======================================================================================
# Python ↔ R argument conversion & ADTnorm call
# ======================================================================================

def _process_kwargs(kwargs: Dict):
    """
    Convert Python kwargs into rpy2-friendly objects.
    Maps None → R NULL. Handles lists/tuples of scalars.
    Provides default save_outpath and study_name if missing.
    """
    default_kwargs = dict(save_outpath='ADTnorm', study_name='ADTnormPy')

    for k, v in default_kwargs.items():
        if k not in kwargs:
            kwargs[k] = v

    for k in list(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, (str, bool, float, int)):
            continue
        elif isinstance(v, (tuple, list)) and len(v) > 0:
            if isinstance(v[0], int):
                kwargs[k] = rpy2.robjects.vectors.IntVector(v)
            elif isinstance(v[0], float):
                kwargs[k] = rpy2.robjects.vectors.FloatVector(v)
            elif isinstance(v[0], str):
                kwargs[k] = rpy2.robjects.vectors.StrVector(v)
            else:
                raise NotImplementedError(f"Rpy2 conversion for {k}: list/tuple element type {type(v[0])}")
        elif v is None:
            kwargs[k] = rpy2.rinterface.NULL
        else:
            raise NotImplementedError(f"Rpy2 conversion for {k}: {type(v)}")

    return kwargs


def _adtnorm_core(ADT_data: pd.DataFrame,
                  obs: pd.DataFrame,
                  marker_to_process: List[str],
                  customize_landmark: bool,
                  save_landmark: bool,
                  verbose: int,
                  override_landmark,
                  **kwargs) -> pd.DataFrame:
    """
    Call ADTnorm::ADTnorm() on a pandas frame + obs.
    Returns a pandas DataFrame (cells x selected markers).
    Any exception is raised to the caller for fallback handling.
    """
    ADTnormR = _load_adtnorm_R(verbose)
    kwargs = _process_kwargs(kwargs)

    # Landmark override plumbing
    if override_landmark is not None:
        if isinstance(override_landmark, str):
            # try .rds -> else .csv bundle
            try:
                _log('Attempting to load override_landmark from .rds')
                res = load_landmarks_r(override_landmark, append_rds=False)
                assert len(res) > 0
                _log(f"Success, found overrides for: {list(res.names)}")
                override_landmark = res
            except Exception:
                _log('Failed .rds. Attempting to load override_landmark from .csv')
                override_landmark = load_python_landmarks(override_landmark, study_name='ADTnormPy', append_csv=False)
                _log(f"Success, found overrides for: {list(override_landmark.keys())}")
        if isinstance(override_landmark, dict):
            override_landmark = landmarks_to_r(override_landmark)
        kwargs['override_landmark'] = override_landmark

    with rpy2.robjects.conversion.localconverter(
        rpy2.robjects.default_converter + rpy2.robjects.pandas2ri.converter
    ):
        # Allow interactive tuning only in serial contexts
        try:
            if customize_landmark:
                base = rpy2.robjects.packages.importr('base')
                if list(base.getOption('browser')) == ['']:
                    base.options(rpy2.robjects.ListVector(dict(browser='firefox')))

            cell_x_adtnorm = ADTnormR.ADTnorm(
                cell_x_adt=ADT_data,
                cell_x_feature=obs,
                marker_to_process=marker_to_process,
                customize_landmark=customize_landmark,
                save_landmark=save_landmark,
                verbose=bool(verbose - 1),
                **kwargs
            )
        except Exception:
            # Bubble up a clean exception for outer fallback logic
            raise RuntimeError('R Runtime Error inside ADTnorm')

    return cell_x_adtnorm


# ======================================================================================
# Public single-call API (serial): mirrors original but more defensive
# ======================================================================================

def adtnorm(data: Union[pd.DataFrame, anndata.AnnData, mudata.MuData],
            sample_column: str = 'sample',
            marker_to_process: Optional[Union[str, List[str]]] = None,
            obs: Optional[pd.DataFrame] = None,
            batch_column: Optional[str] = None,
            ADT_location: Optional[str] = 'protein',
            return_location: Optional[str] = 'ADTnorm',
            customize_landmark: bool = False,
            save_landmark: bool = True,
            override_landmark: Optional[Union[str, Dict[str, Dict[str, pd.DataFrame]]]] = None,
            verbose: int = 1,
            **kwargs) -> Union[pd.DataFrame, anndata.AnnData, mudata.MuData]:
    """
    Run ADTnorm once over possibly many markers (serial path).
    For large panels, prefer `adtnorm_parallel_markers` below.
    """

    # ------- 1) Extract ADT matrix + obs as pandas --------
    if isinstance(data, pd.DataFrame):
        assert isinstance(obs, pd.DataFrame) and len(data) == len(obs) and all(data.index == obs.index), \
            "If ADT expression is a DataFrame, provide an obs DataFrame with matching indices."
        ADT_data = data
        obs = obs.copy()
    elif isinstance(data, mudata.MuData):
        ADT_data = pd.DataFrame(
            data.mod[ADT_location].X,
            columns=data.mod[ADT_location].var_names,
            index=data.mod[ADT_location].obs_names
        )
        if obs is None:
            obs = data.mod[ADT_location].obs.copy()
    elif isinstance(data, anndata.AnnData):
        return_to_layer = True
        if ADT_location is None:
            ADT_data = pd.DataFrame(data.X, columns=data.var_names, index=data.obs_names)
        elif ADT_location in data.layers.keys():
            ADT_data = pd.DataFrame(data.layers[ADT_location], columns=data.var_names, index=data.obs_names)
        elif ADT_location in data.obsm.keys():
            ADT_data = data.obsm[ADT_location]
            return_to_layer = False
        else:
            raise AssertionError(f"Could not find ADT expression in '{ADT_location}'")
        if obs is None:
            obs = data.obs.copy()
    else:
        raise AssertionError("Provide AnnData, MuData, or DataFrame (+obs)")

    # ------- 2) Sample/batch columns -------
    assert sample_column in obs.columns, f"Could not find '{sample_column}' in obs"
    obs = obs.copy()
    obs['sample'] = obs[sample_column]
    if batch_column is None:
        obs.drop(['batch'], axis=1, errors='ignore', inplace=True)
    else:
        obs['batch'] = obs[batch_column]

    # ------- 3) Marker selection -------
    if marker_to_process is None:
        marker_to_process = ADT_data.columns
    elif isinstance(marker_to_process, str):
        marker_to_process = [marker_to_process]

    index_dtype = type(ADT_data.index[0])

    # ------- 4) Call ADTnorm in one shot -------
    adtnorm_res = _adtnorm_core(
        ADT_data=ADT_data,
        obs=obs,
        marker_to_process=list(marker_to_process),
        customize_landmark=customize_landmark,
        save_landmark=save_landmark,
        verbose=verbose,
        override_landmark=override_landmark,
        **kwargs
    )
    adtnorm_res.index = ADT_data.index

    # ------- 5) Optional name cleaning (R helper) -------
    if kwargs.get('clean_adt_name', False):
        ADTnormR = _load_adtnorm_R(verbose)
        with rpy2.robjects.conversion.localconverter(
            rpy2.robjects.default_converter + rpy2.robjects.pandas2ri.converter
        ):
            marker_to_process = list(ADTnormR.clean_adt_name(marker_to_process))

    # ------- 6) Stitch back into container -------
    if isinstance(data, pd.DataFrame):
        out = pd.DataFrame(adtnorm_res, index=obs.index, columns=marker_to_process)
        out.index = out.index.astype(index_dtype)
        return out

    if return_location is None:
        return_location = ADT_location

    if isinstance(data, mudata.MuData):
        data.mod[return_location] = data.mod[ADT_location][:, marker_to_process].copy()
        data.mod[return_location].X = adtnorm_res.values
        return data

    # AnnData
    if not return_to_layer:
        out = pd.DataFrame(adtnorm_res, index=obs.index, columns=marker_to_process)
        out.index = out.index.astype(index_dtype)
        data.obsm[return_location] = out
        return data
    else:
        full_cols = list(data.var_names)
        out = pd.DataFrame(adtnorm_res, index=obs.index, columns=marker_to_process)

        # Backfill any not-processed markers from RAW
        missing = [c for c in full_cols if c not in out.columns]
        if missing:
            if verbose:
                print(f"{len(missing)} markers not processed; filling from raw '{ADT_location}' instead of NA.")
            if ADT_location is None:
                raw_block = pd.DataFrame(data.X, index=data.obs_names, columns=data.var_names)[missing]
            elif ADT_location in data.layers.keys():
                raw_block = pd.DataFrame(data.layers[ADT_location], index=data.obs_names, columns=data.var_names)[missing]
            elif ADT_location in data.obsm.keys():
                raw_block = pd.DataFrame(data.obsm[ADT_location], index=data.obs_names)[missing]
            else:
                raise AssertionError(f"Could not find ADT expression in '{ADT_location}' to backfill missing markers.")
            out = pd.concat([out, raw_block], axis=1)

        out = out.reindex(columns=full_cols).astype(np.float32)

        if return_location is None:
            data.X = out.values
        else:
            data.layers[return_location] = out.values
        return data


# ======================================================================================
# Helpers to extract columns, raw fallbacks
# ======================================================================================

def _extract_marker_column_from_result(res, marker: str, return_location: str):
    # AnnData
    if isinstance(res, anndata.AnnData):
        arr = res.layers[return_location] if return_location in res.layers else res.X
        j = res.var_names.get_loc(marker)
        col = arr[:, j]
        return (col.A1 if hasattr(col, "A1") else np.asarray(col)).astype(np.float32)
    # MuData
    if hasattr(res, "mod"):
        arr = res.mod[return_location].X
        j = res.mod[return_location].var_names.get_loc(marker)
        return np.asarray(arr[:, j]).astype(np.float32)
    # DataFrame
    return res[marker].to_numpy(dtype=np.float32)


def _raw_marker_column(data, marker: str, ADT_location: Optional[str] = 'protein'):
    # AnnData
    if isinstance(data, anndata.AnnData):
        if ADT_location is None:
            j = data.var_names.get_loc(marker)
            col = data.X[:, j]
            return (col.A1 if hasattr(col, "A1") else np.asarray(col)).astype(np.float32)
        elif ADT_location in data.layers:
            col = data.layers[ADT_location][:, data.var_names.get_loc(marker)]
        else:
            col = data.obsm[ADT_location][marker].values
        return (col.A1 if hasattr(col, "A1") else np.asarray(col)).astype(np.float32)
    # MuData
    if hasattr(data, "mod"):
        col = data.mod[ADT_location][:, marker].X
        return (col.A1 if hasattr(col, "A1") else np.asarray(col)).astype(np.float32)
    # DataFrame
    return data[marker].to_numpy(dtype=np.float32)


# ======================================================================================
# Landmark I/O bridges (Python CSV <-> R RDS)
# ======================================================================================

def landmarks_to_r(override_landmark, save_dir=None, study_name='ADTnormPy', append_rds=True):
    """
    Convert Python landmark dict → R ListVector; optionally save .rds files.
    If `override_landmark` is a string, treat it as a directory containing CSVs.
    """
    if isinstance(override_landmark, str):
        override_landmark = load_python_landmarks(override_landmark, study_name, append_csv=True)

    if save_dir is not None:
        if append_rds:
            save_dir = save_dir + '/RDS'
        os.makedirs(save_dir, exist_ok=True)

    base = rpy2.robjects.packages.importr('base')
    r_override_landmark = {}
    for marker in override_landmark.keys():
        with rpy2.robjects.conversion.localconverter(
            rpy2.robjects.default_converter + rpy2.robjects.pandas2ri.converter
        ):
            r_override_landmark[marker] = rpy2.robjects.ListVector(override_landmark[marker])
        for i in ['peak_landmark_list', 'valley_landmark_list']:
            r_override_landmark[marker].rx2[i] = base.data_matrix(r_override_landmark[marker].rx2[i])
        if save_dir is not None:
            base.saveRDS(r_override_landmark[marker], save_dir + f"/peak_valley_locations_{marker}_{study_name}.rds")

    return rpy2.robjects.ListVector(r_override_landmark)


def load_landmarks_r(path_or_r_obj, append_rds=True):
    ADTnormR = _load_adtnorm_R()
    return ADTnormR.load_landmarks(path_or_r_obj, append_rds)


def landmarks_to_python(res, save_dir=None, study_name='ADTnormPy', append_rds=True, append_csv=True):
    """
    Convert R landmark object or directory path → Python dict;
    optionally save CSVs for human editing.
    """
    if isinstance(res, str):
        res = load_landmarks_r(res, append_rds)

    override_landmark = {}
    for marker in res.names:
        override_landmark[marker] = {}
        for i in ['peak_landmark_list', 'valley_landmark_list']:
            info = res.rx2(marker).rx2(i)
            override_landmark[marker][i] = pd.DataFrame(np.array(info), index=info.rownames, columns=range(info.dim[1]))

    if save_dir is not None:
        save_python_landmarks(override_landmark, save_dir=save_dir, study_name=study_name, append_csv=append_csv)
    return override_landmark


def save_python_landmarks(override_landmark, save_dir, study_name='ADTnormPy', append_csv=True):
    if append_csv:
        save_dir = save_dir + '/CSV'
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for marker in override_landmark.keys():
        override_landmark[marker]['peak_landmark_list'].to_csv(
            save_dir + f"/peak_locations_{marker}_{study_name}.csv"
        )
        override_landmark[marker]['valley_landmark_list'].to_csv(
            save_dir + f"/valley_locations_{marker}_{study_name}.csv"
        )


def load_python_landmarks(load_dir, study_name='ADTnormPy', append_csv=True):
    if append_csv:
        load_dir = load_dir + '/CSV'
    res = {}
    for filename in os.listdir(load_dir):
        if filename.endswith(f'{study_name}.csv') and ('_locations_' in filename):
            stem = filename[0:-(len(study_name) + 5)]
            parts = stem.split('_')
            marker = '_'.join(parts[2:])
            if marker not in res:
                res[marker] = {}
            res[marker][f'{parts[0]}_landmark_list'] = pd.read_csv(os.path.join(load_dir, filename), index_col=0)
    return res


# ======================================================================================
# Worker: single marker (optional R capture disabled by default)
# ======================================================================================

def _run_one_marker_verbose(
    marker: str,
    data,
    sample_column: str,
    ADT_location: str,
    return_location: str,
    r_verbose_level: int = 1,
    show_params: bool = True,
    timestamp_logs: bool = True,
    capture_r: bool = False,
    **kwargs
):
    _log("Starting", marker=marker, ts=timestamp_logs)
    if show_params:
        params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        _log(
            f"Params: sample_column={sample_column}, ADT_location={ADT_location}, "
            f"return_location={return_location}" + (f" | extra: {params_str}" if params_str else ""),
            marker=marker, ts=timestamp_logs
        )

    tmpdir = tempfile.mkdtemp(prefix=f"adtnorm_{marker}_")
    r_out = os.path.join(tmpdir, "R_stdout.txt")
    r_err = os.path.join(tmpdir, "R_stderr.txt")

    try:
        d = data.copy()

        def _call():
            return adtnorm(
                data=d,
                sample_column=sample_column,
                marker_to_process=[marker],
                ADT_location=ADT_location,
                return_location=return_location,
                customize_landmark=False,
                verbose=r_verbose_level,
                **kwargs
            )

        if capture_r:
            with _r_capture_context(r_out, r_err):
                res = _call()
            _print_r_file(r_out, marker)
            _print_r_file(r_err, marker)
        else:
            res = _call()

        if res is None:
            _log("WARNING: ADTnorm returned None, using RAW.", marker=marker)
            raw = _raw_marker_column(data, marker, ADT_location)
            return marker, raw

        col = _extract_marker_column_from_result(res, marker, return_location)
        if not np.isfinite(col).all():
            _log("NaNs/inf detected in normalized values, replacing with RAW.", marker=marker)
            raw = _raw_marker_column(data, marker, ADT_location)
            bad = ~np.isfinite(col)
            col[bad] = raw[bad]

        _log(f"Done (n_cells={col.size})", marker=marker, ts=timestamp_logs)
        return marker, col.astype(np.float32)

    except Exception as e:
        _log(f"ERROR: {type(e).__name__}: {e}", marker=marker, ts=timestamp_logs)
        try:
            if capture_r:
                _print_r_file(r_out, marker)
                _print_r_file(r_err, marker)
        except Exception:
            pass
        _log("Falling back to RAW for this marker.", marker=marker)
        raw = _raw_marker_column(data, marker, ADT_location)
        return marker, raw.astype(np.float32)

    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


# ======================================================================================
# Public parallel API with chunking
# ======================================================================================

def adtnorm_parallel_markers(data,
                             sample_column: str = "sample",
                             ADT_location: str = "protein",
                             return_location: str = "ADTnorm",
                             markers: Optional[List[str]] = None,
                             n_jobs: int = 4,
                             chunk_size: int = 4,
                             verbose: bool = True,
                             r_verbose_level: int = 1,
                             show_params: bool = True,
                             timestamp_logs: bool = True,
                             **kwargs):
    """
    Parallel ADTnorm runner with chunking to reduce R overhead.

    Parameters
    ----------
    data : AnnData | MuData | pd.DataFrame
    sample_column : str
    ADT_location : str
        For AnnData: layer key (or None for .X, or obsm key for DataFrame input)
        For MuData: modality key
    return_location : str
        Destination layer/modality for normalized output.
    markers : list[str] | None
        Default: all markers. For testing, pass a subset.
    n_jobs : int
        Number of parallel worker processes (loky backend).
    chunk_size : int
        Number of markers to process per R call, per worker.
        Use 4–10 for large panels to amortize R startup.
    r_verbose_level : int
        Forwarded to `adtnorm(... verbose=...)`. 1 is usually enough.
    show_params, timestamp_logs : bool
        Controls Python-side logging.

    Returns
    -------
    data (modified in-place for AnnData/MuData) or a new DataFrame (if DataFrame input).
    """
    # Resolve marker names from container
    if isinstance(data, anndata.AnnData):
        all_markers = list(data.var_names)
    elif hasattr(data, "mod"):  # MuData
        all_markers = list(data.mod[ADT_location].var_names)
    else:  # DataFrame
        all_markers = list(data.columns)

    if markers is None:
        markers = all_markers
    else:
        markers = [m for m in markers if m in all_markers]

    if verbose:
        _log(f"Parallel ADTnorm starting: {len(markers)} markers | n_jobs={n_jobs} | chunk_size={chunk_size}")

    # --- CHUNKED path: one R call per chunk per worker ---
    chunks = [markers[i:i + chunk_size] for i in range(0, len(markers), chunk_size)]

    def _runner_chunk(markers_chunk: List[str]):
        out = {}
        try:
            d = data.copy()
            # Single R call for the whole chunk
            res = adtnorm(
                data=d,
                sample_column=sample_column,
                marker_to_process=markers_chunk,
                ADT_location=ADT_location,
                return_location=return_location,
                customize_landmark=False,
                verbose=r_verbose_level,
                **kwargs
            )

            # Extract each marker from the returned object
            for m in markers_chunk:
                try:
                    col = _extract_marker_column_from_result(res, m, return_location)
                    if not np.isfinite(col).all():
                        raw = _raw_marker_column(data, m, ADT_location)
                        bad = ~np.isfinite(col)
                        col[bad] = raw[bad]
                    out[m] = col.astype(np.float32)
                except Exception:
                    # Per-marker fallback from chunk failure
                    out[m] = _raw_marker_column(data, m, ADT_location).astype(np.float32)
            return out

        except Exception:
            # If the whole chunk call failed, backfill RAW for the whole chunk
            for m in markers_chunk:
                out[m] = _raw_marker_column(data, m, ADT_location).astype(np.float32)
            return out

    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_runner_chunk)(c) for c in chunks)
    col_map: Dict[str, np.ndarray] = {}
    for dct in results:
        col_map.update(dct)

    # --- Stitch back in original order, backfilling gaps from RAW, final NaN guard ---
    if isinstance(data, anndata.AnnData):
        n, P = data.n_obs, len(all_markers)
        M = np.empty((n, P), dtype=np.float32); M[:] = np.nan

        for j, m in enumerate(all_markers):
            if m in col_map:
                M[:, j] = col_map[m]
            else:
                if verbose:
                    _log(f"Note: marker {m} was not processed; backfilling from RAW.")
                M[:, j] = _raw_marker_column(data, m, ADT_location)

        # Per-cell NaN → RAW
        if ADT_location is None:
            raw_full = data.X
        elif ADT_location in data.layers:
            raw_full = data.layers[ADT_location]
        else:
            raw_full = data.obsm[ADT_location]
        raw_np = raw_full if isinstance(raw_full, np.ndarray) else (
            raw_full.values if hasattr(raw_full, "values") else np.asarray(raw_full)
        )
        bad = ~np.isfinite(M)
        if bad.any():
            if verbose:
                _log(f"Final guard: {int(bad.sum())} NaN/inf values → RAW fallback")
            M[bad] = raw_np[bad]

        data.layers[return_location] = M
        if verbose:
            _log(f"Done. Wrote layer '{return_location}' with shape {M.shape}")
        return data

    elif hasattr(data, "mod"):  # MuData
        n = data.mod[ADT_location].n_obs
        P = len(all_markers)
        M = np.empty((n, P), dtype=np.float32); M[:] = np.nan
        for j, m in enumerate(all_markers):
            M[:, j] = col_map.get(m, _raw_marker_column(data, m, ADT_location))

        raw = data.mod[ADT_location].X
        raw_np = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
        bad = ~np.isfinite(M)
        if bad.any():
            if verbose:
                _log(f"Final guard: {int(bad.sum())} NaN/inf values → RAW fallback")
            M[bad] = raw_np[bad]

        data.mod[return_location] = data.mod[ADT_location][:, all_markers].copy()
        data.mod[return_location].X = M
        if verbose:
            _log(f"Done. Wrote modality '{return_location}' with shape {M.shape}")
        return data

    else:  # DataFrame input/output
        df = pd.DataFrame(index=data.index, columns=all_markers, dtype=np.float32)
        for m in all_markers:
            df[m] = col_map.get(m, data[m].astype(np.float32).values)
        if verbose:
            _log(f"Done. Returning DataFrame with shape {df.shape}")
        return df
