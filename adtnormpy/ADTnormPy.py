import pandas as pd
import numpy as np
import anndata
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
import os

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
from joblib import Parallel, delayed

def _log(msg, prefix=None, enable=True):
    if not enable:
        return
    if prefix:
        sys.stdout.write(f"[{prefix}] {msg}\n")
    else:
        sys.stdout.write(f"{msg}\n")
    sys.stdout.flush()

def _extract_raw_column(data, marker, ADT_location):
    """Return raw values for one marker as float32 1D array."""
    if isinstance(data, anndata.AnnData):
        if ADT_location is None:
            j = data.var_names.get_loc(marker)
            col = data.X[:, j]
            return (col.A1 if hasattr(col, "A1") else np.asarray(col)).astype(np.float32)
        elif ADT_location in data.layers:
            arr = data.layers[ADT_location]
            j = data.var_names.get_loc(marker)
            col = arr[:, j]
            return (col.A1 if hasattr(col, "A1") else np.asarray(col)).astype(np.float32)
        else:  # obsm
            return pd.Series(data.obsm[ADT_location][marker].values).to_numpy(dtype=np.float32)

    # MuData
    if hasattr(data, "mod"):
        mod = data.mod[ADT_location]
        j = mod.var_names.get_loc(marker)
        col = mod.X[:, j]
        return (col.A1 if hasattr(col, "A1") else np.asarray(col)).astype(np.float32)

    # DataFrame
    return data[marker].to_numpy(dtype=np.float32)

def _extract_result_column(res, marker, return_location):
    """Pull the single marker column from the adtnorm() result, regardless of container type."""
    if isinstance(res, anndata.AnnData):
        arr = res.layers[return_location] if return_location in res.layers else res.X
        j = res.var_names.get_loc(marker)
        col = arr[:, j]
        return (col.A1 if hasattr(col, "A1") else np.asarray(col)).astype(np.float32)

    if hasattr(res, "mod"):  # MuData
        mod = res.mod[return_location]
        j = mod.var_names.get_loc(marker)
        col = mod.X[:, j]
        return (col.A1 if hasattr(col, "A1") else np.asarray(col)).astype(np.float32)

    # DataFrame
    return res[marker].to_numpy(dtype=np.float32)

def _run_one_marker(marker, data, sample_column, ADT_location, return_location, kwargs, verbose=True):
    """Worker: run adtnorm() on a single marker with verbose logging, fallback to raw on failure."""
    _log("starting ADTnorm", prefix=marker, enable=verbose)
    t0 = time.time()
    try:
        # import inside worker so each process gets its own rpy2/R
        res = adtnorm(
            data=data.copy(),
            sample_column=sample_column,
            marker_to_process=[marker],
            ADT_location=ADT_location,
            return_location=return_location,
            customize_landmark=False,          # interactive off in parallel
            verbose=True,                      # make R chatty
            **kwargs
        )
        col = _extract_result_column(res, marker, return_location)
        dt = time.time() - t0
        _log(f"finished in {dt:.2f}s", prefix=marker, enable=verbose)
        # sanitize: replace any NaN/Inf with raw
        if not np.isfinite(col).all():
            _log("non-finite values detected; falling back to raw for those cells", prefix=marker, enable=verbose)
            raw = _extract_raw_column(data, marker, ADT_location)
            bad = ~np.isfinite(col)
            col[bad] = raw[bad]
        return marker, col
    except Exception as e:
        dt = time.time() - t0
        _log(f"FAILED in {dt:.2f}s: {repr(e)}", prefix=marker, enable=verbose)
        _log(traceback.format_exc().rstrip(), prefix=marker, enable=verbose)
        # fallback: raw column
        raw = _extract_raw_column(data, marker, ADT_location)
        return marker, raw

def _run_chunk(chunk, data, sample_column, ADT_location, return_location, kwargs, verbose=True):
    """Worker: run adtnorm() on a small batch of markers to amortize R startup."""
    tag = f"chunk({len(chunk)}):{','.join(chunk[:3])}{'...' if len(chunk)>3 else ''}"
    _log("starting ADTnorm", prefix=tag, enable=verbose)
    t0 = time.time()
    out = {}
    try:
        res = adtnorm(
            data=data.copy(),
            sample_column=sample_column,
            marker_to_process=chunk,
            ADT_location=ADT_location,
            return_location=return_location,
            customize_landmark=False,
            verbose=True,      # R verbose
            **kwargs
        )
        for m in chunk:
            try:
                col = _extract_result_column(res, m, return_location)
                if not np.isfinite(col).all():
                    raw = _extract_raw_column(data, m, ADT_location)
                    bad = ~np.isfinite(col)
                    col[bad] = raw[bad]
                out[m] = col.astype(np.float32, copy=False)
            except Exception:
                _log(f"marker {m} failed inside chunk; using raw", prefix=tag, enable=verbose)
                out[m] = _extract_raw_column(data, m, ADT_location)
        dt = time.time() - t0
        _log(f"finished in {dt:.2f}s", prefix=tag, enable=verbose)
        return out
    except Exception as e:
        dt = time.time() - t0
        _log(f"FAILED in {dt:.2f}s: {repr(e)}", prefix=tag, enable=verbose)
        _log(traceback.format_exc().rstrip(), prefix=tag, enable=verbose)
        # fallback: raw for whole chunk
        for m in chunk:
            out[m] = _extract_raw_column(data, m, ADT_location)
        return out

def adtnorm_parallel_markers(
    data,
    sample_column="sample",
    ADT_location="protein",
    return_location="ADTnorm",
    markers=None,
    n_jobs=4,
    chunk_size=1,
    verbose=True,
    **kwargs
):
    """
    Parallel per-marker (or small chunk) ADTnorm with verbose logging.

    - Each worker runs in a separate process (safe for rpy2/R).
    - ADTnorm R prints are enabled (verbose=True) and prefixed by marker/chunk.
    - Any NaN/Inf from ADTnorm is replaced cell-by-cell with the raw input values.
    - Unprocessed markers are backfilled from the raw input (no holes).
    """
    # Build the complete marker list for the container
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

    # Parallel execution
    if chunk_size > 1:
        chunks = [markers[i:i+chunk_size] for i in range(0, len(markers), chunk_size)]
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_chunk)(c, data, sample_column, ADT_location, return_location, kwargs, verbose)
            for c in chunks
        )
        col_map = {}
        for dct in results:
            col_map.update(dct)
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_one_marker)(m, data, sample_column, ADT_location, return_location, kwargs, verbose)
            for m in markers
        )
        col_map = {m: col for m, col in results}

    # Stitch back in the original order, backfilling any gaps from raw
    if isinstance(data, anndata.AnnData):
        n = data.n_obs
        p = len(all_markers)
        M = np.empty((n, p), dtype=np.float32); M[:] = np.nan
        for j, m in enumerate(all_markers):
            if m in col_map:
                M[:, j] = col_map[m]
            else:
                # not processed → raw
                M[:, j] = _extract_raw_column(data, m, ADT_location)

        # final cell-by-cell NaN→raw guard
        if ADT_location is None:
            raw_full = data.X
        elif ADT_location in data.layers:
            raw_full = data.layers[ADT_location]
        else:
            raw_full = data.obsm[ADT_location].values  # DataFrame->ndarray
        raw_np = raw_full if isinstance(raw_full, np.ndarray) else np.asarray(raw_full)
        bad = ~np.isfinite(M)
        if bad.any():
            M[bad] = raw_np[bad]

        data.layers[return_location] = M
        _log(f"wrote layer '{return_location}' with shape {M.shape}", enable=verbose)
        return data

    if hasattr(data, "mod"):  # MuData
        mod_in = data.mod[ADT_location]
        n = mod_in.n_obs
        p = len(all_markers)
        M = np.empty((n, p), dtype=np.float32); M[:] = np.nan
        for j, m in enumerate(all_markers):
            if m in col_map:
                M[:, j] = col_map[m]
            else:
                M[:, j] = _extract_raw_column(data, m, ADT_location)

        # final NaN→raw
        raw = mod_in.X
        raw_np = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
        bad = ~np.isfinite(M)
        if bad.any():
            M[bad] = raw_np[bad]

        # create/overwrite modality for return
        data.mod[return_location] = mod_in[:, all_markers].copy()
        data.mod[return_location].X = M
        _log(f"created modality '{return_location}' with shape {M.shape}", enable=verbose)
        return data

    # DataFrame
    df = pd.DataFrame(index=data.index, columns=all_markers, dtype=np.float32)
    for m in all_markers:
        df[m] = col_map.get(m, data[m].astype(np.float32).values)
    _log(f"built DataFrame result with shape {df.shape}", enable=verbose)
    return df


