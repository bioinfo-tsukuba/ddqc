#!/usr/bin/env python3

from typing import Union
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from ddqc_utils import mad


def ddqc_scanpy(adata, 
                cluster_key: str = "louvain",
                method: str = "mad", threshold: float = 2.0, 
                threshold_counts: Union[int, None] = 0,
                threshold_genes: Union[int, None] = 0, 
                threshold_mito: Union[float, None] = 0,
                threshold_ribo: Union[float, None] = 0,
                mito_prefix: str = "MT-", 
                ribo_prefix: str = "^RP[SL][[:digit:]]|^RPLP[[:digit:]]|^RPSA",
                n_genes_lower_bound: int = 0, 
                percent_mito_upper_bound: float = 10.0,
                return_df_qc: bool = False, 
                display_plots: bool = True):
    """
    Perform DDQC on scanpy AnnData object using existing clusters.
    
    Parameters:
        adata: scanpy AnnData object with existing clustering
        cluster_key (str): key in adata.obs containing cluster labels (default: "louvain")
        method (str): statistic method - "mad" or "outlier" (default: "mad")
        threshold (float): parameter for the selected method (default: 2.0)
        threshold_counts, threshold_genes, threshold_mito, threshold_ribo: 
            specific thresholds for each metric (default: 0 = use method threshold, None = skip)
        mito_prefix (str): gene prefix for mitochondrial genes (default: "MT-")
        ribo_prefix (str): regex pattern for ribosomal genes
        n_genes_lower_bound (int): lower bound for n_genes cluster threshold (default: 0)
        percent_mito_upper_bound (float): upper bound for percent_mito cluster threshold (default: 10.0)
        return_df_qc (bool): whether to return QC dataframe (default: False)
        display_plots (bool): whether to display QC plots (default: True)
    
    Returns:
        AnnData object with 'passed_qc' added to .obs, 
        optionally returns QC dataframe if return_df_qc=True
    """
    # Make a copy to avoid modifying the original
    adata_copy = adata.copy()
    
    # Calculate basic QC metrics if not present or rename to match DDQC expected names
    if 'n_genes_by_counts' not in adata_copy.obs:
        adata_copy.var['mt'] = adata_copy.var_names.str.startswith(mito_prefix)
        sc.pp.calculate_qc_metrics(adata_copy, percent_top=None, log1p=False, inplace=True)
    
    # Always ensure we have the expected column names
    if 'n_counts' not in adata_copy.obs and 'total_counts' in adata_copy.obs:
        adata_copy.obs['n_counts'] = adata_copy.obs['total_counts']
    if 'n_genes' not in adata_copy.obs and 'n_genes_by_counts' in adata_copy.obs:
        adata_copy.obs['n_genes'] = adata_copy.obs['n_genes_by_counts']
    if 'percent_mito' not in adata_copy.obs and 'pct_counts_mt' in adata_copy.obs:
        adata_copy.obs['percent_mito'] = adata_copy.obs['pct_counts_mt']
    
    # Handle ribosomal gene percentage - use calculated values if available, otherwise calculate
    if 'percent_ribo' not in adata_copy.obs:
        if 'pct_counts_ribo' in adata_copy.obs:
            adata_copy.obs['percent_ribo'] = adata_copy.obs['pct_counts_ribo']
        else:
            # Fallback: calculate ribosomal gene percentage
            ribo_genes = adata_copy.var_names.str.match(ribo_prefix, case=False)
            if ribo_genes.any():
                if issparse(adata_copy.X):
                    ribo_counts = np.array(adata_copy.X[:, ribo_genes].sum(axis=1)).flatten()
                else:
                    ribo_counts = adata_copy.X[:, ribo_genes].sum(axis=1)
                adata_copy.obs['percent_ribo'] = (ribo_counts / np.maximum(adata_copy.obs['n_counts'].values, 1.0)) * 100
            else:
                adata_copy.obs['percent_ribo'] = 0.0
    
    # Use existing clusters
    if cluster_key not in adata_copy.obs:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")
    
    adata_copy.obs['cluster_labels'] = adata_copy.obs[cluster_key].astype('category')
    
    # Perform DDQC filtering using the adapted logic
    passed_qc, df_qc = _perform_ddqc_scanpy(adata_copy, method, threshold,
                                          threshold_counts, threshold_genes, 
                                          threshold_mito, threshold_ribo,
                                          n_genes_lower_bound, percent_mito_upper_bound)
    
    # Add results back to original adata
    adata.obs['passed_qc'] = passed_qc
    
    # Display plots if requested
    if display_plots:
        # Simple plotting without dependency on old plotting module
        print("Note: Basic plotting disabled to avoid Pegasus dependencies. Use ddqc_post.enhanced_ddqc_with_plots() for plotting.")
    
    if return_df_qc:
        return adata, df_qc
    else:
        return adata


def _perform_ddqc_scanpy(adata, method: str, param: float,
                        threshold_counts: Union[int, None],
                        threshold_genes: Union[int, None], 
                        threshold_mito: Union[float, None],
                        threshold_ribo: Union[float, None],
                        n_genes_lower_bound: int,
                        percent_mito_upper_bound: float):
    """
    Adapted DDQC filtering logic for scanpy AnnData objects.
    """
    n_cells = adata.n_obs
    qc_pass = np.ones(n_cells, dtype=bool)
    
    # Create QC dataframe
    df_qc = pd.DataFrame(index=adata.obs.index)
    df_qc['cluster_labels'] = adata.obs['cluster_labels']
    
    # Apply filtering for each metric
    metrics_to_filter = []
    
    if threshold_counts is not None:
        if threshold_counts == 0:
            metrics_to_filter.append(('n_counts', param, True, False, -np.inf, np.inf))
        else:
            metrics_to_filter.append(('n_counts', threshold_counts, True, False, -np.inf, np.inf))
    
    if threshold_genes is not None:
        if threshold_genes == 0:
            metrics_to_filter.append(('n_genes', param, True, False, n_genes_lower_bound, np.inf))
        else:
            metrics_to_filter.append(('n_genes', threshold_genes, True, False, n_genes_lower_bound, np.inf))
    
    if threshold_mito is not None:
        if threshold_mito == 0:
            metrics_to_filter.append(('percent_mito', param, False, True, -np.inf, percent_mito_upper_bound))
        else:
            metrics_to_filter.append(('percent_mito', threshold_mito, False, True, -np.inf, percent_mito_upper_bound))
    
    if threshold_ribo is not None:
        if threshold_ribo == 0:
            metrics_to_filter.append(('percent_ribo', param, False, True, -np.inf, np.inf))
        else:
            metrics_to_filter.append(('percent_ribo', threshold_ribo, False, True, -np.inf, np.inf))
    
    # Apply filtering for each metric
    for metric_name, threshold_val, do_lower_co, do_upper_co, lower_bound, upper_bound in metrics_to_filter:
        metric_pass = _metric_filter_scanpy(adata, method, threshold_val, metric_name, 
                                          do_lower_co, do_upper_co, 
                                          lower_bound, upper_bound, df_qc)
        qc_pass &= metric_pass
    
    df_qc['passed_qc'] = qc_pass
    return qc_pass, df_qc


def _metric_filter_scanpy(adata, method: str, param: float, metric_name: str, 
                         do_lower_co: bool = False, do_upper_co: bool = False,
                         lower_bound: float = -np.inf, upper_bound: float = np.inf,
                         df_qc: pd.DataFrame = None):
    """
    Adapted metric filtering for scanpy AnnData objects.
    """
    qc_pass = np.zeros(adata.n_obs, dtype=bool)
    
    if df_qc is not None:
        df_qc[f"{metric_name}_lower_co"] = None
        df_qc[f"{metric_name}_upper_co"] = None
    
    for cl in adata.obs["cluster_labels"].cat.categories:
        idx = adata.obs["cluster_labels"] == cl
        values = adata.obs.loc[idx, metric_name]
        
        if method == "mad":
            median_v = np.median(values)
            mad_v = mad(values)
            lower_co = median_v - param * mad_v
            upper_co = median_v + param * mad_v
        else:  # outlier method
            q75, q25 = np.percentile(values, [75, 25])
            lower_co = q25 - 1.5 * (q75 - q25)
            upper_co = q75 + 1.5 * (q75 - q25)
        
        lower_co = max(lower_co, lower_bound)
        upper_co = min(upper_co, upper_bound)
        
        qc_pass_cl = np.ones(values.size, dtype=bool)
        if df_qc is not None:
            df_qc.loc[idx, f"{metric_name}"] = values
        if do_lower_co:
            qc_pass_cl &= (values >= lower_co)
            if df_qc is not None:
                df_qc.loc[idx, f"{metric_name}_lower_co"] = lower_co
        if do_upper_co:
            qc_pass_cl &= (values <= upper_co)
            if df_qc is not None:
                df_qc.loc[idx, f"{metric_name}_upper_co"] = upper_co
        if df_qc is not None:
            df_qc.loc[idx, f"{metric_name}_passed_qc"] = qc_pass_cl
        
        qc_pass[idx] = qc_pass_cl
    
    return qc_pass