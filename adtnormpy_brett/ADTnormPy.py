# ADTnormPy.py
# =============================================================================
# ADTnorm Python bridge (single-process, chunked execution, robust NaN guards)
# -----------------------------------------------------------------------------
# Highlights
# - Single R session per Python process (no per-marker restarts).
# - Chunked execution: call ADTnorm once per group of markers to amortize cost.
# - Strong safety: any NaN/Inf in normalized values are replaced with RAW values.
# - If ADTnorm skips a marker, values are backfilled from RAW.
# - Public API mirrors the original:
#     * load_adtnorm_R(), install_adtnorm_R()
#     * adtnorm(...)
#     * adtnorm_parallel_markers(...)  ← now runs *serially* but chunked
#     * landmarks_to_r(), landmarks_to_python(), save_python_landmarks(), load_python_landmarks()
# - Supports pandas.DataFrame, anndata.AnnData, and mudata.MuData.
# =============================================================================

import os
import sys
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


# =============================================================================
# R session handle (cached per Python process)
# =============================================================================

_ADTNORM_R = None

def load_adtnorm_R(verbose: int = 1):
    """
    Load the R package 'ADTnorm' exactly once per Python process.

    Parameters
    ----------
    verbose : int
        >1 prints R sessionInfo() when the package is first loaded.

    Returns
    -------
    ADTnorm R package handle.
    """
    global _ADTNORM_R
    if _ADTNORM_R is not None:
        return _ADTNORM_R

    try:
        ADTnormR = rpy2.robjects.packages.importr('ADTnorm')
    except rpy2.robjects.packages.PackageNotInstalledError:
        install_adtnorm_R()
        ADTnormR = rpy2.robjects.packages.importr('ADTnorm')

    _ADTNORM_R = ADTnormR
    if verbose > 1:
        print(rpy2.robjects.r('sessionInfo()'))
    return _ADTNORM_R


def install_adtnorm_R():
    """
    Install ADTnorm into the active R library via devtools::install_github.
    Only called automatically if ADTnorm is not found.

    This function may take a while the first time (downloads sources & deps).
    """
    _ = input(
        f"Install ADTnorm into R at: {os.environ.get('R_HOME','<unknown R_HOME>')} ? "
        f"Type anything to continue..."
    )
    try:
        devtools = rpy2.robjects.packages.importr('devtools')
    except rpy2.robjects.packages.PackageNotInstalledError:
        utils = rpy2.robjects.packages.importr('utils')
        print("Installing 'devtools' before installing ADTnorm...")
        utils.install_packages('devtools', quiet=True, quick=True, upgrade_dependencies=True, keep_source=False)
        devtools = rpy2.robjects.packages.importr('devtools')

    print("Installing ADTnorm and dependencies (this may take a while)...")
    devtools.install_github("yezhengSTAT/ADTnorm", build_vignettes=False, quiet=True)


# =============================================================================
# Small logging helper
# =============================================================================

def _log(msg: str):
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


# =============================================================================
# Python ↔ R argument conversion & core ADTnorm call
# =============================================================================

def _process_kwargs(kwargs: Dict) -> Dict:
    """
    Convert Python kwargs into rpy2-friendly objects and provide defaults.

    Rules
    -----
    - None            → R NULL
    - list/tuple[int] → IntVector
    - list/tuple[float] → FloatVector
    - list/tuple[str] → StrVector
    - scalars (str/bool/float/int) pass-through

    Defaults
    --------
    save_outpath='ADTnorm', study_name='ADTnormPy'
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
                raise NotImplementedError(f"Rpy2 conversion for {k}: element type {type(v[0])}")
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
    Call ADTnorm::ADTnorm() (R) on a pandas frame + obs and return a pandas DataFrame.

    Any R failure raises RuntimeError to the caller, which then can apply fallbacks.
    """
    ADTnormR = load_adtnorm_R(verbose)
    kwargs = _process_kwargs(kwargs)

    # Optional: override peak/valley landmarks
    if override_landmark is not None:
        if isinstance(override_landmark, str):
            # Try .rds via ADTnorm helper; otherwise load CSV bundle
            try:
                _log('Attempting to load landmark overrides from .rds ...')
                res = load_landmarks_r(override_landmark, append_rds=False)
                assert len(res) > 0
                _log(f"Found overrides for: {list(res.names)}")
                override_landmark = res
            except Exception:
                _log('No .rds; attempting to load overrides from CSV directory ...')
                override_landmark = load_python_landmarks(override_landmark, study_name='ADTnormPy', append_csv=True)
                _log(f"Found overrides for: {list(override_landmark.keys())}")
        if isinstance(override_landmark, dict):
            override_landmark = landmarks_to_r(override_landmark)
        kwargs['override_landmark'] = override_landmark

    with rpy2.robjects.conversion.localconverter(
        rpy2.robjects.default_converter + rpy2.robjects.pandas2ri.converter
    ):
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
            raise RuntimeError('R Runtime Error inside ADTnorm')

    return cell_x_adtnorm


# =============================================================================
# Helpers to extract columns and to fetch RAW columns
# =============================================================================

def _extract_marker_column_from_result(res, marker: str, return_location: str) -> np.ndarray:
    """
    Extract a single marker vector from the object returned by `adtnorm(...)`.
    - If `adtnorm` was called on AnnData/MuData, `res` is the container.
    - If `adtnorm` was called on a DataFrame, `res` is a DataFrame.
    """
    # AnnData
    if isinstance(res, anndata.AnnData):
        arr = res.layers[return_location] if return_location in res.layers else res.X
        j = res.var_names.get_loc(marker)
        col = arr[:, j]
        return (col.A1 if hasattr(col, "A1") else np.asarray(col)).astype(np.float32)

    # MuData
    if hasattr(res, "mod"):  # duck-typing for MuData
        arr = res.mod[return_location].X
        j = res.mod[return_location].var_names.get_loc(marker)
        return np.asarray(arr[:, j]).astype(np.float32)

    # DataFrame
    return res[marker].to_numpy(dtype=np.float32)


def _raw_marker_column(data, marker: str, ADT_location: Optional[str] = 'protein') -> np.ndarray:
    """
    Pull the RAW (pre-normalized) values for a single marker from the input container.
    """
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


# =============================================================================
# Public single-call API (serial)
# =============================================================================

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
    Run ADTnorm on (optionally many) markers in a single call.

    Parameters
    ----------
    data : DataFrame | AnnData | MuData
        ADT matrix (cells x markers). AnnData: use `ADT_location` to choose X/layer/obsm.
    sample_column : str
        Column in `obs` indicating sample identity per cell.
    marker_to_process : None | str | list[str]
        If None, all columns/var_names are processed. Otherwise restrict to these markers.
    obs : DataFrame | None
        Required when `data` is a DataFrame (must match rows). Ignored otherwise (taken from container).
    batch_column : str | None
        Optional column in `obs` for batches (if None, ADTnorm treats batch=sample).
    ADT_location : str | None
        Location of protein expression in AnnData/MuData. Use None to reference AnnData.X directly.
    return_location : str | None
        Destination to store normalized output (AnnData layer, MuData modality, or DataFrame).
        If None, overwrites the source location.
    customize_landmark : bool
        If True, invoke ADTnorm’s interactive landmark tuning (serial-only).
    save_landmark : bool
        If True, ADTnorm saves landmark summary files.
    override_landmark : str | dict | None
        Either a directory with CSV overrides, an .rds path, or a Python dict in the expected format.
    verbose : int
        0 = minimal, 1 = standard, 2 = R sessionInfo() on first import.
    **kwargs :
        Passed through to ADTnorm::ADTnorm. e.g., trimodal_marker, exclude_zeroes, quantile_clip, etc.

    Returns
    -------
    A DataFrame with normalized values (if input was DataFrame),
    or the original AnnData/MuData with `return_location` filled.
    """

    # ------- 1) Materialize ADT_data (pandas) and obs -------
    if isinstance(data, pd.DataFrame):
        assert isinstance(obs, pd.DataFrame) and len(data) == len(obs) and all(data.index == obs.index), \
            "If ADT expression is a DataFrame, provide an obs DataFrame with matching indices."
        ADT_data = data
        obs = obs.copy()
        return_to_layer = False
    elif isinstance(data, mudata.MuData):
        ADT_data = pd.DataFrame(
            data.mod[ADT_location].X,
            columns=data.mod[ADT_location].var_names,
            index=data.mod[ADT_location].obs_names
        )
        if obs is None:
            obs = data.mod[ADT_location].obs.copy()
        return_to_layer = True
    elif isinstance(data, anndata.AnnData):
        return_to_layer = True
        if ADT_location is None:
            ADT_data = pd.DataFrame(data.X, columns=data.var_names, index=data.obs_names)
        elif ADT_location in data.layers:
            ADT_data = pd.DataFrame(data.layers[ADT_location], columns=data.var_names, index=data.obs_names)
        elif ADT_location in data.obsm:
            ADT_data = data.obsm[ADT_location]  # DataFrame-like
            return_to_layer = False
        else:
            raise AssertionError(f"Could not find ADT expression in '{ADT_location}'")
        if obs is None:
            obs = data.obs.copy()
    else:
        raise AssertionError("Provide AnnData, MuData, or DataFrame (+obs)")

    # ------- 2) Sample/batch columns for ADTnorm -------
    assert sample_column in obs.columns, f"Could not find '{sample_column}' in obs"
    obs = obs.copy()
    obs['sample'] = obs[sample_column]
    if batch_column is None:
        # ADTnorm interprets missing 'batch' as batch=sample, so we drop any stale 'batch' numeric/text
        obs.drop(['batch'], axis=1, errors='ignore', inplace=True)
    else:
        obs['batch'] = obs[batch_column]

    # ------- 3) Marker selection -------
    if marker_to_process is None:
        marker_to_process = ADT_data.columns
    elif isinstance(marker_to_process, str):
        marker_to_process = [marker_to_process]
    marker_to_process = list(marker_to_process)

    index_dtype = type(ADT_data.index[0]) if len(ADT_data.index) else str

    # ------- 4) Call ADTnorm in one shot -------
    adtnorm_res = _adtnorm_core(
        ADT_data=ADT_data,
        obs=obs,
        marker_to_process=marker_to_process,
        customize_landmark=customize_landmark,
        save_landmark=save_landmark,
        verbose=verbose,
        override_landmark=override_landmark,
        **kwargs
    )
    adtnorm_res.index = ADT_data.index

    # ------- 5) Optional ADT name cleaning via R helper -------
    if kwargs.get('clean_adt_name', False):
        ADTnormR = load_adtnorm_R(verbose)
        with rpy2.robjects.conversion.localconverter(
            rpy2.robjects.default_converter + rpy2.robjects.pandas2ri.converter
        ):
            marker_to_process = list(ADTnormR.clean_adt_name(marker_to_process))

    # ------- 6) Stitch back with strong NaN/Inf and "missing marker" guards -------
    if isinstance(data, pd.DataFrame):
        out = pd.DataFrame(adtnorm_res, index=obs.index, columns=marker_to_process)
        out.index = out.index.astype(index_dtype)

        # Per-cell NaN/Inf guard → RAW
        out_np = out.values
        bad = ~np.isfinite(out_np)
        if bad.any():
            raw_np = ADT_data[marker_to_process].to_numpy(copy=False)
            out_np[bad] = raw_np[bad]
        return out

    # Containers
    if return_location is None:
        return_location = ADT_location

    # MuData
    if isinstance(data, mudata.MuData):
        data.mod[return_location] = data.mod[ADT_location][:, marker_to_process].copy()
        data.mod[return_location].X = adtnorm_res.values
        return data

    # AnnData
    if not return_to_layer:
        # Write into obsm if the source was obsm
        out = pd.DataFrame(adtnorm_res, index=obs.index, columns=marker_to_process)
        out.index = out.index.astype(index_dtype)
        data.obsm[return_location] = out
        return data

    # Layer-based writeback
    full_cols = list(data.var_names)
    out = pd.DataFrame(adtnorm_res, index=obs.index, columns=marker_to_process)

    # Backfill any markers ADTnorm did not process
    missing = [c for c in full_cols if c not in out.columns]
    if missing:
        if verbose:
            _log(f"{len(missing)} markers not processed; filling from raw '{ADT_location}' instead of NA.")
        if ADT_location is None:
            raw_block = pd.DataFrame(data.X, index=data.obs_names, columns=data.var_names)[missing]
        elif ADT_location in data.layers:
            raw_block = pd.DataFrame(data.layers[ADT_location], index=data.obs_names, columns=data.var_names)[missing]
        elif ADT_location in data.obsm:
            raw_block = pd.DataFrame(data.obsm[ADT_location], index=data.obs_names)[missing]
        else:
            raise AssertionError(f"Could not find ADT expression in '{ADT_location}' to backfill missing markers.")
        out = pd.concat([out, raw_block], axis=1)

    # Align & cast
    out = out.reindex(columns=full_cols).astype(np.float32)

    # Per-cell NaN/Inf guard → RAW
    if ADT_location is None:
        raw_full = data.X
    elif ADT_location in data.layers:
        raw_full = data.layers[ADT_location]
    else:
        raw_full = data.obsm[ADT_location]
    raw_np = raw_full if isinstance(raw_full, np.ndarray) else (
        raw_full.values if hasattr(raw_full, "values") else np.asarray(raw_full)
    )
    out_np = out.values
    bad = ~np.isfinite(out_np)
    if bad.any():
        if verbose:
            _log(f"Final guard: {int(bad.sum())} NaN/inf values → RAW fallback")
        out_np[bad] = raw_np[bad]

    if return_location is None:
        data.X = out_np
    else:
        data.layers[return_location] = out_np
    return data


# =============================================================================
# Chunked multi-marker API (serial; keeps one R session)
# =============================================================================

def adtnorm_parallel_markers(data,
                             sample_column: str = "sample",
                             ADT_location: str = "protein",
                             return_location: str = "ADTnorm",
                             markers: Optional[List[str]] = None,
                             n_jobs: int = 1,        # kept for API compatibility; ignored
                             chunk_size: int = 8,     # typical sweet spot 4–16
                             verbose: bool = True,
                             r_verbose_level: int = 1,
                             show_params: bool = True,
                             timestamp_logs: bool = True,
                             **kwargs):
    """
    Run ADTnorm in chunks (serial) to avoid per-marker R restarts and avoid
    multiprocessing pickling/sink issues while still reducing R-call overhead.

    Returns
    -------
    The same container with `return_location` filled (AnnData/MuData) or a
    new DataFrame (if input was DataFrame).
    """
    # Resolve markers from the input container
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
        _log(f"Parallel ADTnorm (chunked serial) starting: {len(markers)} markers | chunk_size={chunk_size}")

    # Chunk the panel
    chunks = [markers[i:i + chunk_size] for i in range(0, len(markers), chunk_size)]

    # Collect per-marker columns into a map
    col_map: Dict[str, np.ndarray] = {}

    for chunk in chunks:
        try:
            if show_params and timestamp_logs:
                _log(f"Processing chunk of {len(chunk)} markers: {chunk[:3]}{'...' if len(chunk) > 3 else ''}")

            # Single call to adtnorm for the chunk
            res = adtnorm(
                data=data.copy(),  # cheap view for AnnData; isolated for safety
                sample_column=sample_column,
                marker_to_process=chunk,
                ADT_location=ADT_location,
                return_location=return_location,
                customize_landmark=False,
                verbose=r_verbose_level,
                **kwargs
            )

            # Extract each marker (with NaN/Inf → RAW safety)
            for m in chunk:
                try:
                    col = _extract_marker_column_from_result(res, m, return_location)
                    if not np.isfinite(col).all():
                        raw = _raw_marker_column(data, m, ADT_location)
                        col[~np.isfinite(col)] = raw[~np.isfinite(col)]
                    col_map[m] = col.astype(np.float32)
                except Exception:
                    col_map[m] = _raw_marker_column(data, m, ADT_location).astype(np.float32)

        except Exception as e:
            # Whole-chunk failure → RAW for each marker in the chunk
            if verbose:
                _log(f"Chunk failed ({type(e).__name__}): {e}. Backfilling RAW for {len(chunk)} markers.")
            for m in chunk:
                col_map[m] = _raw_marker_column(data, m, ADT_location).astype(np.float32)

    # Stitch columns back in original order + final NaN guard
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

        # Per-cell NaN/Inf → RAW
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
        if bad.any() and verbose:
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
        if bad.any() and verbose:
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


# =============================================================================
# Landmark I/O bridges (Python CSV <-> R RDS)
# =============================================================================

def landmarks_to_r(override_landmark,
                   save_dir: Optional[str] = None,
                   study_name: str = 'ADTnormPy',
                   append_rds: bool = True):
    """
    Convert Python landmark-overrides to an R ListVector; optionally save .rds files.

    `override_landmark` can be:
      - a dict {marker: {'peak_landmark_list': df, 'valley_landmark_list': df}}
      - a path to a directory containing the CSVs saved by save_python_landmarks()

    Returns
    -------
    rpy2.robjects.ListVector
    """
    if isinstance(override_landmark, str):
        override_landmark = load_python_landmarks(override_landmark, study_name=study_name, append_csv=True)

    if save_dir is not None:
        if append_rds:
            save_dir = os.path.join(save_dir, 'RDS')
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
            base.saveRDS(r_override_landmark[marker],
                         os.path.join(save_dir, f"peak_valley_locations_{marker}_{study_name}.rds"))

    return rpy2.robjects.ListVector(r_override_landmark)


def load_landmarks_r(path_or_r_obj, append_rds: bool = True):
    """
    Load landmark R objects using ADTnormR helper (path or existing R object).
    """
    ADTnormR = load_adtnorm_R()
    return ADTnormR.load_landmarks(path_or_r_obj, append_rds)


def landmarks_to_python(res,
                        save_dir: Optional[str] = None,
                        study_name: str = 'ADTnormPy',
                        append_rds: bool = True,
                        append_csv: bool = True):
    """
    Convert an R landmark object (or a path to one) into a Python dict and
    optionally save CSVs for easy editing.
    """
    if isinstance(res, str):
        res = load_landmarks_r(res, append_rds=append_rds)

    override_landmark: Dict[str, Dict[str, pd.DataFrame]] = {}
    for marker in res.names:
        override_landmark[marker] = {}
        for i in ['peak_landmark_list', 'valley_landmark_list']:
            info = res.rx2(marker).rx2(i)
            override_landmark[marker][i] = pd.DataFrame(np.array(info),
                                                        index=info.rownames,
                                                        columns=range(info.dim[1]))
    if save_dir is not None:
        save_python_landmarks(override_landmark, save_dir=save_dir, study_name=study_name, append_csv=append_csv)
    return override_landmark


def save_python_landmarks(override_landmark: Dict[str, Dict[str, pd.DataFrame]],
                          save_dir: str,
                          study_name: str = 'ADTnormPy',
                          append_csv: bool = True):
    """
    Save a landmark dict as CSVs ({save_dir}/CSV/*_{study_name}.csv).
    """
    if append_csv:
        save_dir = os.path.join(save_dir, 'CSV')
    os.makedirs(save_dir, exist_ok=True)

    for marker in override_landmark.keys():
        override_landmark[marker]['peak_landmark_list'].to_csv(
            os.path.join(save_dir, f"peak_locations_{marker}_{study_name}.csv")
        )
        override_landmark[marker]['valley_landmark_list'].to_csv(
            os.path.join(save_dir, f"valley_locations_{marker}_{study_name}.csv")
        )


def load_python_landmarks(load_dir: str,
                          study_name: str = 'ADTnormPy',
                          append_csv: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load landmark CSVs previously saved by save_python_landmarks(...).
    """
    if append_csv:
        load_dir = os.path.join(load_dir, 'CSV')

    res: Dict[str, Dict[str, pd.DataFrame]] = {}
    for filename in os.listdir(load_dir):
        if filename.endswith(f'{study_name}.csv') and ('_locations_' in filename):
            stem = filename[:- (len(study_name) + 5)]
            parts = stem.split('_')
            marker = '_'.join(parts[2:])
            if marker not in res:
                res[marker] = {}
            key = f'{parts[0]}_landmark_list'
            res[marker][key] = pd.read_csv(os.path.join(load_dir, filename), index_col=0)
    return res
