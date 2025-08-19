import pandas as pd
import numpy as np
import anndata
import shutil
try:
    import mudata
except ImportError:
    class EmptyModule: # Create a fake module if not installed
        def __init__(self):
            self.MuData = type(None)
    mudata = EmptyModule()
    
import rpy2.robjects
import rpy2.robjects.pandas2ri
import rpy2.robjects.packages
import os,sys, tempfile
from joblib import Parallel, delayed
from typing import Union, Optional, Sequence, Any, Mapping, List, Tuple, Callable, List, Set, Iterable, Dict
def adtnorm(data: Union[pd.DataFrame,anndata.AnnData,mudata.MuData],
            sample_column: str = 'sample',
            marker_to_process: Optional[Union[str,List[str]]] = None,
            obs: Optional[pd.DataFrame] = None,
            batch_column: Optional[str] = None,
            ADT_location: Optional[str] = 'protein',
            return_location: Optional[str] = 'ADTnorm',
            customize_landmark: bool = False, 
            save_landmark: bool = True,
            override_landmark: Optional[Union[str,Dict[str,Dict[str,pd.DataFrame]]]] = None,
            verbose: int = 1,
            **kwargs) -> Union[pd.DataFrame,anndata.AnnData,mudata.MuData]:
    """
    Run ADTnorm through Python, converting pandas.DataFrame, anndata.AnnData or mudata.MuData objects into the proper format, and returning landmark registered ADT expression in the same object. 
    ADTnorm results can optionally be stored as a separate layer (for AnnData objects) or modality (for MuData objects). 
        
    Note: This package requires a functional ADTnorm to have been installed in R. 
    
    This may be achieved using `python -c "import adtnormpy;adtnormpy.install_adtnorm_R()"` or by installing via instructions in the README.md.
    
    Parameters
    ----------
    data
        Object containing ADT expression stored in one of: a pandas.DataFrame (with a row for each cell and column for each ADT marker), 
                                                           an anndata.AnnData object, stored in the .X, a layer, or the .obsm, or 
                                                           a mudata.MuData object, stored in a .mod
                                                           Note: ADTnorm does not support sparse matrices as inputs
    sample_column
        Column name corresponding to the sample for which to perform landmark registration. 
    marker_to_process
        ADT marker(s) to perform landmark registration on. Use None to specify performing landmark registration on all markers.
    obs
        Optional, must be provided when data is a pandas.DataFrame. pandas.DataFrame corresponding to cell metadata
    batch_column
        Optional column name corresponding to the batches (which can be used to plot the histograms of many batches within the same sample. If unset, it will be treated as the same as the sample column.
    ADT_location
        When an anndata.AnnData object or mudata.MuData object are provided as data, what key to use to extract protein information. Note, use None to reference AnnData.X.
    return_location
        When an anndata.AnnData object or mudata.MuData object are provided as data, after landmark registration is performed where to store batch-corrected expression. If None, will overwrite the original ADT location.
    customize_landmark
        Whether to open a popup for each marker to tune landmark detection. Depending on your system this may not pop up, but instead provide a link. We recommend using this function after initial rounds of ADTnorm 
        normalizing with a few attempts at tuning parameters. It is better to narrow down to a few ADT markers that need manual tuning and provide the list to marker_to_process, as this will trigger an interactive 
        function for every marker processed. 
    override_landmark
        Override the peak and valley locations if prior information is available or the user wants to manually adjust the peak and valley locations for certain markers. This is much faster to use , and can be easily rerun
        and edited. Input can be formatted as a dictionary of markers, with a dictionary of pd.DataFrames, one for peak locations (first) and one for valley locations (second). Alternatively, can be provided as a file path.
        with rows in each corresponding to each sample to override landmark detection of. 
    verbose
        The function verbosity: 0 for minimal R and Python outputs, 1 for normal, and 2 for extended verbosity.
    **kwargs
        Keyword arguments which are passed to ADTnorm::ADTnorm() in R. A list of keywords can be found [here](https://yezhengstat.github.io/ADTnorm/reference/ADTnorm.html). Note clean_adt_names is not supported when data is an AnnData or MuData object.

    Returns
    -------
    Landmark-registered ADT expression, in the same format as data was provided.  
    
    """
    # Format conversion and information
    if isinstance(data, pd.DataFrame):
        assert isinstance(obs, pd.DataFrame) and len(data) == len(obs) and all(data.index == obs.index), 'If ADT expression is provided as a pd.DataFrame, please provide an obs DataFrame of metadata with matching indicies.'
        ADT_data = data
        obs = obs.copy()
    elif isinstance(data, mudata.MuData):
        ADT_data = pd.DataFrame(data.mod[ADT_location].X,columns=data.mod[ADT_location].var_names,index=data.mod[ADT_location].obs_names)
        if obs is None:
            obs = data.mod[ADT_location].obs.copy()
    elif isinstance(data, anndata.AnnData):
        return_to_layer = True
        if ADT_location is None:
            ADT_data = pd.DataFrame(data.X,columns=data.var_names,index=data.obs_names)
        elif ADT_location in data.layers.keys():
            ADT_data = pd.DataFrame(data.layers[ADT_location],columns=data.var_names,index=data.obs_names)
        elif ADT_location in data.obsm.keys():
            ADT_data = data.obsm[ADT_location]
            return_to_layer = False
        else:
            assert False, f"Could not find ADT expression in '{ADT_location}'"
        if obs is None:
            obs = data.obs.copy()
    else:
        assert False, 'Please specify data as an AnnData object or a MuData object, or specify data as a pd.DataFrame and obs as pd.DataFrame'
    
    # Identification and assignment of sample and batch columns
    assert sample_column in obs.columns, f"Could not find '{sample_column}' in the obs"
    obs['sample'] = obs[sample_column]
    if batch_column is None:
        obs.drop(['batch'],axis=1,errors='ignore',inplace=True)
    else:
        obs['batch'] = obs[batch_column] 
    
    # Select markers to return
    if marker_to_process is None:
        marker_to_process = ADT_data.columns
    elif isinstance(marker_to_process,str):
        marker_to_process = [marker_to_process]
    
    index_dtype = type(ADT_data.index[0])
    adtnorm_res = _adtnorm(ADT_data, obs, marker_to_process, customize_landmark, save_landmark, verbose,override_landmark, **kwargs)
    adtnorm_res.index = ADT_data.index
    
    if 'clean_adt_name' in kwargs and kwargs['clean_adt_name']:
        ADTnorm = load_adtnorm_R(verbose)
        with rpy2.robjects.conversion.localconverter(rpy2.robjects.default_converter + rpy2.robjects.pandas2ri.converter):
            marker_to_process = ADTnorm.clean_adt_name(marker_to_process)
        
        
    if isinstance(data, pd.DataFrame):
        data = pd.DataFrame(adtnorm_res,index=obs.index,columns=marker_to_process)
        data.index = data.index.astype(index_dtype)
    else:
        if return_location is None: 
            return_location = ADT_location
        if isinstance(data, mudata.MuData):
            data.mod[return_location] = data.mod[ADT_location][:,marker_to_process].copy() # Check that this subsets correctly.
            data.mod[return_location].X = adtnorm_res.values
        if isinstance(data, anndata.AnnData):
            if not return_to_layer:
                adtnorm_res = pd.DataFrame(adtnorm_res,index=obs.index,columns=marker_to_process)
                adtnorm_res.index = adtnorm_res.index.astype(index_dtype)
                data.obsm[return_location] = adtnorm_res
            else:
                full_cols = list(data.var_names)
                adtnorm_res = pd.DataFrame(adtnorm_res, index=obs.index, columns=marker_to_process)
                # Fallback policy for markers we didn’t process
                missing = [c for c in full_cols if c not in adtnorm_res.columns]
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
                    adtnorm_res = pd.concat([adtnorm_res, raw_block], axis=1)

                # Final align + type
                adtnorm_res = adtnorm_res.reindex(columns=full_cols).astype(np.float32)

                if return_location is None:
                    data.X = adtnorm_res.values
                else:
                    data.layers[return_location] = adtnorm_res.values
    return data
def _adtnorm(ADT_data, obs,marker_to_process,customize_landmark, save_landmark, verbose, override_landmark, **kwargs):
    # Loading the R function
    ADTnormR = load_adtnorm_R(verbose)
    
    # Converting to R objects for running
    kwargs = _process_kwargs(kwargs) 

    # Provide landmark overrides
    if not override_landmark is None:
        if type(override_landmark) is str:
            try:
                print('Attempting to load override_landmark from .rds')
                res = load_landmarks_r(override_landmark,False)
                assert len(res)>0
                print(f'Success, found overrides for: {list(res.names)}')
                override_landmark = res
            except:
                print('Failed. Attempting to load override_landmark from .csv')
                override_landmark = load_python_landmarks(override_landmark,'ADTnormPy',False)
                print(f'Success, found overrides for: {override_landmark.keys()}')
        if type(override_landmark) is dict:
            override_landmark = landmarks_to_r(override_landmark)
        kwargs['override_landmark'] = override_landmark   
        
    with rpy2.robjects.conversion.localconverter(rpy2.robjects.default_converter + rpy2.robjects.pandas2ri.converter):
        try:
            if customize_landmark:
                base = rpy2.robjects.packages.importr('base')
                if list(base.getOption('browser')) == ['']:
                    base.options(rpy2.robjects.ListVector(dict(browser='firefox')))
            cell_x_adtnorm = ADTnormR.ADTnorm(
                cell_x_adt = ADT_data, 
                cell_x_feature = obs,
                marker_to_process = marker_to_process,
                customize_landmark = customize_landmark,
                save_landmark = save_landmark,
                verbose = bool(verbose-1),
                **kwargs)
        except Exception as e:
            tb = rpy2.robjects.r("traceback(max.lines=1)")
            assert False, 'Ran into R Runtime Error'
    return cell_x_adtnorm

def _process_kwargs(kwargs):
    default_kwargs = dict(save_outpath = 'ADTnorm', study_name = 'ADTnormPy')
    # TODO figure out how to handle named lists
    # positive_peak = list(ADT = "CD3", sample = "buus_2021_T"),

    for i in default_kwargs.keys():
        if not i in kwargs.keys():
            kwargs[i] = default_kwargs[i]
    for i in kwargs.keys():
        if type(kwargs[i]) is str or type(kwargs[i]) is bool or type(kwargs[i]) is float or type(kwargs[i]) is int:
            pass
        elif type(kwargs[i]) is tuple or type(kwargs[i]) is list:
            if type(kwargs[i][0]) is int:
                kwargs[i] = rpy2.robjects.vectors.IntVector(kwargs[i]) 
            elif type(kwargs[i][0]) is float:
                kwargs[i] = rpy2.robjects.vectors.FloatVector(kwargs[i])
            elif type(kwargs[i][0]) is str:
                kwargs[i] = rpy2.robjects.vectors.StrVector(kwargs[i])
            else:
                assert False, f'Rpy2 for {i}:{kwargs[i]} has not been implemented ({type(kwargs[i])} of {type(kwargs[i][0])})'
        elif kwargs[i] is None:
            kwargs[i] = rpy2.rinterface.NULL
        else:
            assert False, f'Rpy2 for {i}:{kwargs[i]} has not been implemented ({type(kwargs[i])}'
            
    return kwargs

def load_adtnorm_R(verbose=1):
    '''Use this to load ADTnorm library in R, returns the package via rpy2.'''
    try:
        ADTnormR = rpy2.robjects.packages.importr('ADTnorm')
    except rpy2.robjects.packages.PackageNotInstalledError:
        install_adtnorm_R()
        ADTnormR = rpy2.robjects.packages.importr('ADTnorm')
    if verbose > 1:
        print(rpy2.robjects.r('sessionInfo()'))
    return ADTnormR

def install_adtnorm_R():
    '''Use this to install ADTnorm library in R, requires devtools to be installed as well.'''
    res = input(f"Are you sure you would like to install ADTnorm into R at: {os.environ['R_HOME']} ? \
                  Type anything to continue...")
    try:
        devtools = rpy2.robjects.packages.importr('devtools')
    except rpy2.robjects.packages.PackageNotInstalledError:
        utils = rpy2.robjects.packages.importr('utils')
        print("Installing devtools prior to installing ADTnorm. This may take a while...")
        utils.install_packages('devtools',quiet=True,quick=True,upgrade_dependencies=True,keep_source=False)            
        devtools = rpy2.robjects.packages.importr('devtools')
    print("Installing ADTnorm and dependencies. This may take a while...")
    devtools.install_github("yezhengSTAT/ADTnorm",build_vignettes=False,quiet=True)
    return

def landmarks_to_r(override_landmark, save_dir=None, study_name='ADTnormPy', append_rds=True):
    '''
    Convert override_landmark from Python format to R. Can provide a str (to a directory path) or a dictonary of dictionaries of pd.DataFrames. 
    Provide a save_dir and study_name to save them as .rds files. Use append_rds to add "/RDS" to the provided directory path.
    
    '''
    if type(override_landmark) is str: # load .csv files
        override_landmark = load_python_landmarks(override_landmark,study_name,append_rds)
    
    if not save_dir is None:
        if append_rds:
            save_dir = save_dir+'/RDS'
        os.makedirs(save_dir,exist_ok=True)
        
    base = rpy2.robjects.packages.importr('base')
    r_override_landmark = dict()
    for marker in override_landmark.keys():
        with rpy2.robjects.conversion.localconverter(rpy2.robjects.default_converter+rpy2.robjects.pandas2ri.converter):
            r_override_landmark[marker] = rpy2.robjects.ListVector(override_landmark[marker])
        for i in ['peak_landmark_list','valley_landmark_list']:
            r_override_landmark[marker].rx2[i] = base.data_matrix(r_override_landmark[marker].rx2[i])
        if not save_dir is None:
            base.saveRDS(r_override_landmark[marker],save_dir+f"/peak_valley_locations_{marker}_{study_name}.rds")
    r_override_landmark = rpy2.robjects.ListVector(r_override_landmark)
    
    return r_override_landmark

def load_landmarks_r(res,append_rds=True):
    ADTnormR = load_adtnorm_R()
    res = ADTnormR.load_landmarks(res,append_rds)
    return res
    
def landmarks_to_python(res, save_dir=None, study_name='ADTnormPy', append_rds=True, append_csv=True):
    '''
    Convert override_landmark from R format to Python. Can provide a str (to a directory path) or the result of ADTnormR::load_landmarks() in rpy2.
    Provide a save_dir and study_name to save them as .csv files. Use append_csv to add "/CSV" to the provided directory path.
    '''
    if type(res) is str:
        res = load_landmarks_r(res,append_rds)
        
    override_landmark = dict()
    for marker in res.names:
        override_landmark[marker] = dict()
        for i in ['peak_landmark_list','valley_landmark_list']:
            info = res.rx2(marker).rx2(i)
            override_landmark[marker][i] = pd.DataFrame(np.array(info),index=info.rownames,columns=range(info.dim[1]))
    # Saves .csv files
    if not save_dir is None:
        save_python_landmarks(override_landmark, save_dir=save_dir, study_name=study_name, append_csv=append_csv)
    return override_landmark

def save_python_landmarks(override_landmark, save_dir, study_name='ADTnormPy', append_csv=True):
    '''
    Save Python landmark-overrides as .csv files. Provide a save_dir and study_name. Use append_csv to add "/CSV" to the provided directory path.
    '''
    if append_csv:
        save_dir = save_dir+'/CSV'
    if not save_dir is None:
        os.makedirs(save_dir,exist_ok=True)
        
    for marker in override_landmark.keys():
        override_landmark[marker]['peak_landmark_list'].to_csv(save_dir+f"/peak_locations_{marker}_{study_name}.csv")
        override_landmark[marker]['valley_landmark_list'].to_csv(save_dir+f"/valley_locations_{marker}_{study_name}.csv")
    return

def load_python_landmarks(load_dir, study_name='ADTnormPy', append_csv=True):
    '''
    Load Python landmark-overrides from .csv files. Provide a save_dir and study_name. Use append_csv to add "/CSV" to the provided directory path.
    '''    
    if append_csv:
        load_dir = load_dir+'/CSV'
    res = dict()
    for filename in os.listdir(load_dir):
        if filename.endswith(f'{study_name}.csv') and ('_locations_' in filename):
            i = filename[0:-(len(study_name)+5)]
            file_items = i.split('_')
            marker = '_'.join(file_items[2:])
            if not marker in res:
                res[marker] = dict()
            res[marker][f'{file_items[0]}_landmark_list'] = pd.read_csv(load_dir+'/'+filename,index_col=0)

    return res

# ---------- tiny logging helpers ----------
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

# ---------- capture R stdout/stderr via sink() ----------
# We guard this so it only runs in the worker processes.
def _r_capture_context(stdout_path: str, stderr_path: str):
    import contextlib
    import rpy2.robjects as ro

    @contextlib.contextmanager
    def _ctx():
        file_fun = ro.r["file"]
        sink_fun = ro.r["sink"]
        close_fun = ro.r["close"]
        sink_number = ro.r["sink.number"]

        # 1) PRE-DRAIN any existing sinks in this R session (avoids "sink stack is full")
        try:
            while int(sink_number(type="output")[0]) > 0:
                sink_fun(None, type="output")
            while int(sink_number(type="message")[0]) > 0:
                sink_fun(None, type="message")
        except Exception:
            # ignore—older R versions may behave slightly differently
            pass

        # 2) OPEN real connections (fixes "'file' must be NULL..." errors)
        out_con = file_fun(stdout_path, open="wt")
        err_con = file_fun(stderr_path, open="wt")

        # 3) SINK both streams to those connections
        sink_fun(out_con, type="output")
        sink_fun(err_con, type="message")
        try:
            yield
        finally:
            # 4) ALWAYS unwind all sinks (even if ADTnorm internally nested more sink() calls)
            try:
                while int(sink_number(type="output")[0]) > 0:
                    sink_fun(None, type="output")
                while int(sink_number(type="message")[0]) > 0:
                    sink_fun(None, type="message")
            finally:
                # 5) CLOSE connections
                try: close_fun(out_con)
                except Exception: pass
                try: close_fun(err_con)
                except Exception: pass

    return _ctx()

# ---------- helpers to extract columns & raw fallbacks ----------
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

# ---------- single-marker worker (super verbose) ----------
def _run_one_marker_verbose(marker: str,
                            data,
                            sample_column: str,
                            ADT_location: str,
                            return_location: str,
                            r_verbose_level: int = 2,
                            show_params: bool = True,
                            timestamp_logs: bool = True,
                            **kwargs):
    from copy import deepcopy
    import traceback
    import shutil

    _log("Starting", marker=marker, ts=timestamp_logs)
    if show_params:
        params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        _log(f"Params: sample_column={sample_column}, ADT_location={ADT_location}, return_location={return_location}"
             + (f" | extra: {params_str}" if params_str else ""), marker=marker, ts=timestamp_logs)

    tmpdir = tempfile.mkdtemp(prefix=f"adtnorm_{marker}_")
    r_out = os.path.join(tmpdir, "R_stdout.txt")
    r_err = os.path.join(tmpdir, "R_stderr.txt")

    try:
        d = data.copy()

        with _r_capture_context(r_out, r_err):
            res = adtnorm(
                data=d,
                sample_column=sample_column,
                marker_to_process=[marker],
                ADT_location=ADT_location,
                return_location=return_location,
                customize_landmark=False,
                verbose=r_verbose_level,
                **kwargs
            )

        _print_r_file(r_out, marker)
        _print_r_file(r_err, marker)

        col = _extract_marker_column_from_result(res, marker, return_location)
        if not np.isfinite(col).all():
            _log("NaNs detected in normalized values, replacing with raw.", marker=marker)
            raw = _raw_marker_column(data, marker, ADT_location)
            bad = ~np.isfinite(col)
            col[bad] = raw[bad]
        _log(f"Done (n_cells={col.size})", marker=marker, ts=timestamp_logs)
        return marker, col

    except Exception as e:
        _log(f"ERROR: {type(e).__name__}: {e}", marker=marker, ts=timestamp_logs)
        _print_r_file(r_out, marker)
        _print_r_file(r_err, marker)
        _log("Falling back to RAW for this marker.", marker=marker)
        raw = _raw_marker_column(data, marker, ADT_location)
        return marker, raw

    finally:
        # always remove the temp directory
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

# ---------- public API: parallel per-marker (or small chunks) ----------
def adtnorm_parallel_markers(data,
                             sample_column: str = "sample",
                             ADT_location: str = "protein",
                             return_location: str = "ADTnorm",
                             markers: Optional[List[str]] = None,
                             n_jobs: int = 4,
                             chunk_size: int = 1,
                             verbose: bool = True,
                             r_verbose_level: int = 2,
                             show_params: bool = True,
                             timestamp_logs: bool = True,
                             **kwargs):
    """
    Super-verbose parallel ADTnorm runner.

    Parameters
    ----------
    data : AnnData | MuData | pd.DataFrame
    sample_column : str
    ADT_location : str
        For AnnData: layer key (or None for .X, or obsm key for DataFrame input)
        For MuData: modality key
    return_location : str
        Destination layer/modality name for normalized output.
    markers : list[str] | None
        Default: all markers. Subset for testing.
    n_jobs : int
        Number of parallel processes (loky backend).
    chunk_size : int
        >1 to amortize R startup by processing small groups per worker.
    verbose : bool
        Controls Python-side chatter (prefixes); R prints are controlled by r_verbose_level.
    r_verbose_level : int
        Passed through to your `adtnorm(... verbose=...)` (2 → original-level R prints).
    show_params : bool
        Print the kwargs being passed for reproducibility.
    timestamp_logs : bool
        Prefix logs with HH:MM:SS times.

    Returns
    -------
    data (with `return_location` filled) or a DataFrame (if input was DataFrame).
    """
    import anndata

    # Resolve all marker names from the input container
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

    # ---- run in parallel ----
    if chunk_size > 1:
        # process small batches per worker to reduce R startup overhead
        chunks = [markers[i:i+chunk_size] for i in range(0, len(markers), chunk_size)]

        def _runner_chunk(chunk):
            out = {}
            # run each marker in the chunk with full verbosity & capture
            for m in chunk:
                k, col = _run_one_marker_verbose(
                    m, data, sample_column, ADT_location, return_location,
                    r_verbose_level=r_verbose_level,
                    show_params=show_params,
                    timestamp_logs=timestamp_logs,
                    **kwargs
                )
                out[k] = col
            return out

        results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_runner_chunk)(c) for c in chunks)
        col_map = {}
        for dct in results:
            col_map.update(dct)
    else:
        # one marker per process
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_one_marker_verbose)(
                m, data, sample_column, ADT_location, return_location,
                r_verbose_level=r_verbose_level,
                show_params=show_params,
                timestamp_logs=timestamp_logs,
                **kwargs
            ) for m in markers
        )
        col_map = {m: col for m, col in results}

    # ---- stitch back in original order; backfill gaps from raw; final NaN guard ----
    if isinstance(data, anndata.AnnData):
        n, P = data.n_obs, len(all_markers)
        M = np.empty((n, P), dtype=np.float32); M[:] = np.nan

        for j, m in enumerate(all_markers):
            if m in col_map:
                M[:, j] = col_map[m]
            else:
                if verbose:
                    _log(f"Note: marker {m} was not processed; backfilling from raw.")
                M[:, j] = _raw_marker_column(data, m, ADT_location)

        # per-cell NaN → raw
        if ADT_location is None:
            raw_full = data.X
        elif ADT_location in data.layers:
            raw_full = data.layers[ADT_location]
        else:
            raw_full = data.obsm[ADT_location]
        raw_np = raw_full if isinstance(raw_full, np.ndarray) else (raw_full.values if hasattr(raw_full, "values") else np.asarray(raw_full))
        bad = ~np.isfinite(M)
        if bad.any():
            if verbose:
                _log(f"Final guard: {int(bad.sum())} NaN/inf values → raw fallback")
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
                _log(f"Final guard: {int(bad.sum())} NaN/inf values → raw fallback")
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
