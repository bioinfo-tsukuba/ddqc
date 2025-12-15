from typing import Union
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from pegasusio import MultimodalData
import pegasus as pg

from ddqc.filtering import perform_ddqc
from ddqc.plotting import filtering_facet_plot, boxplot_sorted, calculate_filtering_stats
from ddqc.utils import cluster_data, calculate_percent_ribo, mad


def ddqc_metrics(data: MultimodalData,
                 res: float = 1.3, clustering_method: str = "louvain", n_components: int = 50, k: int = 20,
                 method: str = "mad", threshold: float = 2.0, threshold_counts: Union[int, None] = 0,
                 threshold_genes: Union[int, None] = 0, threshold_mito: Union[float, None] = 0,
                 threshold_ribo: Union[float, None] = 0, basic_n_counts: int = 0,
                 basic_n_genes: int = 100, basic_percent_mito: float = 80.0,
                 mito_prefix: str = "MT-", ribo_prefix: str = "^RP[SL][[:digit:]]|^RPLP[[:digit:]]|^RPSA",
                 n_genes_lower_bound: int = 200, percent_mito_upper_bound: float = 10.0, random_state: int = 29,
                 return_df_qc: bool = False, display_plots: bool = True) -> Union[None, pd.DataFrame]:
    """
    Parameters:
        data (MultimodalData): Pegasus object.
        res (float): clustering resolution (default is 1.3).
        clustering_method (str): clustering method that will be used by ddqc clustering. Supported options are:
            "louvain", "leiden", "spectral_louvain", and "spectral_leiden" (default is "louvain").
        n_components (int): number of PCA components (default is 50).
        k (int): k to be used by neighbors Pegasus function (default is 20).
        method (str): statistic on which the threshold would be calculated. Supported options are "mad" and "outlier"
            (default is "mad").
        threshold (float): parameter for the selected method (default is 2).
            Note that "outlier" method doesn't requre parameter and will ignore this option.
        threshold_counts (int, None): setting for applying ddqc based on number of counts. (Default is 0)
            - If set to 0, will perform ddqc on number of counts using the "threshold" parameter provided earlier.
            - If set to a number other than 0, will  overwrite "threshold" parameter for number of counts.
            - If set to None, won't perform ddqc on number of counts.
        threshold_genes (int, None): Same as above, but for number of genes.
        threshold_mito (float, None): Same as above, but for percent of mitochondrial transcripts.
        threshold_ribo (float, None): Same as above, but for percent of ribosomal transcripts.
        basic_n_counts (int): parameter for the initial QC n_counts filtering (default is 0).
        basic_n_genes (int): parameter for the initial QC n_genes filtering (default is 100).
        basic_percent_mito (float): parameter for the initial QC percent_mito filtering (default is 80.0).
        mito_prefix (str): gene prefix used to calculate percent_mito in a cell (default is "MT-").
        ribo_prefix (str): gene regular expression used to calculate percent_ribo in a cell
            (default is "^RP[SL][[:digit:]]|^RPLP[[:digit:]]|^RPSA").
        n_genes_lower_bound (int): bound for lower n_genes cluster-level threshold (default is 200).
        percent_mito_upper_bound (float): bound for upper percent_mito cluster-level threshold (default is 10).
        random_state (int): random seed for clustering results reproducibility (default is 29)
        return_df_qc (bool): whether to return a dataframe with the information about on what metric and what threshold
            the cell was removed for each removed cell. (default is False)
        display_plots (bool): whether to show plots that would show filtering statistics (default is True).
    Returns:
        (None, pandas.DataFrame). DataFrame with cluster labels and thresholds for each metric is returned
            if return_df_qc was True.
    """
    assert isinstance(data, MultimodalData)

    # initial qc
    pg.qc_metrics(data, mito_prefix=mito_prefix, min_umis=basic_n_counts, max_umis=10 ** 10, min_genes=basic_n_genes,
                  max_genes=10 ** 10,
                  percent_mito=basic_percent_mito)  # default PG filtering with custom cutoffs
    calculate_percent_ribo(data, ribo_prefix)  # calculate percent ribo
    pg.filter_data(data)  # filtering based on the parameters from qc_metrics

    data_copy = data.copy()
    cluster_data(data_copy, basic_n_counts, basic_n_genes, basic_percent_mito, mito_prefix, ribo_prefix,
                 resolution=res, clustering_method=clustering_method,
                 n_components=n_components, k=k, random_state=random_state)
    passed_qc, df_qc, _ = perform_ddqc(data_copy, method, threshold,
                                       threshold_counts, threshold_genes, threshold_mito, threshold_ribo,
                                       n_genes_lower_bound, percent_mito_upper_bound)

    if display_plots:
        boxplot_sorted(df_qc, "n_genes", "cluster_labels", hline_x=np.log2(200), log=True)
        plt.show()
        boxplot_sorted(df_qc, "percent_mito", "cluster_labels", hline_x=10)
        plt.show()
        if ((threshold_counts == 0 or threshold_counts is None)
                and (threshold_genes == 0 or threshold_genes is None)
                and (threshold_mito == 0 or threshold_mito is None)
                and (threshold_ribo == 0 or threshold_ribo is None)):
            if method == "mad":
                fs = calculate_filtering_stats(data_copy, threshold, n_genes_lower_bound, percent_mito_upper_bound)
                filtering_facet_plot(fs, threshold, pct=False)
                plt.show()
                filtering_facet_plot(fs, threshold, pct=True)
                plt.show()

    # reverse_to_raw_matrix(data.current_data(), obs_copy, var_copy, uns_copy)
    data.obs["passed_qc"] = passed_qc
    df_qc["passed_qc"] = data.obs["passed_qc"]

    if return_df_qc:
        return df_qc


def ddqc_scanpy(adata, 
                cluster_key: str = "louvain",
                method: str = "mad", threshold: float = 2.0, 
                threshold_counts: Union[int, None] = 0,
                threshold_genes: Union[int, None] = 0, 
                threshold_mito: Union[float, None] = 0,
                threshold_ribo: Union[float, None] = 0,
                mito_prefix: str = "MT-", 
                ribo_prefix: str = "^RP[SL][[:digit:]]|^RPLP[[:digit:]]|^RPSA",
                n_genes_lower_bound: int = 200, 
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
        n_genes_lower_bound (int): lower bound for n_genes cluster threshold (default: 200)
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
        boxplot_sorted(df_qc, "n_genes", "cluster_labels", hline_x=np.log2(200), log=True)
        plt.show()
        boxplot_sorted(df_qc, "percent_mito", "cluster_labels", hline_x=10)
        plt.show()
    
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



def detect_species_and_load_genes(gene_names, csv_path="../tools/ddqc/qc_gene_list.csv"):
    """
    Detect species by finding maximum exact matches with gene symbols in CSV
    and return species-specific mitochondrial and ribosomal gene lists.
    
    Parameters:
        gene_names: array-like of gene symbols from the dataset
        csv_path: path to the QC gene list CSV file
        
    Returns:
        tuple: (detected_species, mito_genes, ribo_genes)
    """
    # Read the CSV file
    qc_genes = pd.read_csv(csv_path)
    
    # Convert gene_names to set for faster lookup
    gene_set = set(gene_names)
    
    # Count matches for each species
    species_matches = {}
    for species in qc_genes['Species'].unique():
        species_genes = set(qc_genes[qc_genes['Species'] == species]['Gene_symbol'])
        matches = len(gene_set.intersection(species_genes))
        species_matches[species] = matches
        print(f"Species {species}: {matches} gene matches")
    
    # Find species with maximum matches
    detected_species = max(species_matches, key=species_matches.get)
    print(f"Detected species: {detected_species} with {species_matches[detected_species]} matches")
    
    # Get species-specific gene lists
    species_data = qc_genes[qc_genes['Species'] == detected_species]
    mito_genes = set(species_data[species_data['Group'] == 'mitochondrial']['Gene_symbol'])
    ribo_genes = set(species_data[species_data['Group'] == 'ribosomal']['Gene_symbol'])
    
    # Filter to only genes present in the dataset
    mito_genes_present = [gene for gene in gene_names if gene in mito_genes]
    ribo_genes_present = [gene for gene in gene_names if gene in ribo_genes]
    
    # print(f"Mitochondrial genes found in dataset: {len(mito_genes_present)}")
    # print(f"Ribosomal genes found in dataset: {len(ribo_genes_present)}")
    
    return detected_species, mito_genes_present, ribo_genes_present


def three_way_boxplot(df, metric, cluster_col, ax, cluster_order, log_scale=False, title=""):
    """
    Create three-way DDQC comparison boxplots.
    Shows all cells vs passed cells vs not-passed cells for each cluster.
    """
    # Prepare three datasets
    df_all = df[[metric, cluster_col, 'passed_qc']].copy()  # All cells
    df_pass = df[df['passed_qc'] == True][[metric, cluster_col, 'passed_qc']].copy()  # Passed cells
    df_not_pass = df[df['passed_qc'] == False][[metric, cluster_col, 'passed_qc']].copy()  # Not-passed cells
    
    if log_scale:
        df_all[f'{metric}_plot'] = np.log2(df_all[metric])
        df_pass[f'{metric}_plot'] = np.log2(df_pass[metric])
        df_not_pass[f'{metric}_plot'] = np.log2(df_not_pass[metric])
        ylabel = f'log2({metric})'
    else:
        df_all[f'{metric}_plot'] = df_all[metric]
        df_pass[f'{metric}_plot'] = df_pass[metric]
        df_not_pass[f'{metric}_plot'] = df_not_pass[metric]
        ylabel = metric
    
    # Add labels for three-way comparison
    df_all['ddqc_status'] = 'All'
    df_pass['ddqc_status'] = 'Pass'
    df_not_pass['ddqc_status'] = 'Not-passed'
    
    # Combine all three datasets
    df_combined = pd.concat([df_all, df_pass, df_not_pass], ignore_index=True)
    
    # Create boxplot with three-way comparison
    sns.boxplot(data=df_combined, x=cluster_col, y=f'{metric}_plot', hue='ddqc_status', 
               order=cluster_order, ax=ax, palette=['lightgray', 'lightblue', 'lightcoral'])
    
    # Add cell count annotations
    for i, cluster in enumerate(cluster_order):
        cluster_all = df_all[df_all[cluster_col] == cluster]
        cluster_pass = df_pass[df_pass[cluster_col] == cluster]
        cluster_not_pass = df_not_pass[df_not_pass[cluster_col] == cluster]
        
        n_all = len(cluster_all)
        n_pass = len(cluster_pass)
        n_not_pass = len(cluster_not_pass)
        
        if n_all > 0:  # Only annotate if cluster exists
            # Add text annotation at the top showing passed/total
            y_max = cluster_all[f'{metric}_plot'].max()
            ax.text(i, y_max, f'{n_pass}/{n_all}', ha='center', va='bottom', 
                   fontsize=8, weight='bold', color='darkblue')
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Cluster')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    
    # Move legend to avoid overlap
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


def ddqc_with_enhanced_plots(adata, cluster_key="louvain", method="mad", threshold=2.0, 
                           threshold_counts=0, threshold_genes=0, threshold_mito=0, threshold_ribo=0,
                           plot_output_dir="../plots/"):
    """
    Perform DDQC and create enhanced combined plots with cluster-specific thresholds.
    """
    # Ensure output directory exists
    os.makedirs(plot_output_dir, exist_ok=True)
    
    # First run DDQC without plots to get the QC dataframe
    adata_result, df_qc = ddqc_scanpy(
        adata, 
        cluster_key=cluster_key,
        method=method,
        threshold=threshold,
        threshold_counts=threshold_counts,
        threshold_genes=threshold_genes,
        threshold_mito=threshold_mito,
        threshold_ribo=threshold_ribo,
        display_plots=False,
        return_df_qc=True
    )
    
    # Determine which metrics to plot
    metrics_to_plot = []
    if 'n_genes' in df_qc.columns:
        metrics_to_plot.append(('n_genes', True, 'N_genes per cluster (DDQC)'))
    if 'percent_mito' in df_qc.columns:
        metrics_to_plot.append(('percent_mito', False, 'Percent mitochondrial genes per cluster (DDQC)'))
    if 'percent_ribo' in df_qc.columns:
        metrics_to_plot.append(('percent_ribo', False, 'Percent ribosomal genes per cluster (DDQC)'))
    
    # Create combined plot
    n_plots = len(metrics_to_plot)
    if n_plots == 0:
        print("No metrics available for plotting")
        return adata_result
    
    # Determine consistent cluster order (natural/numerical order)
    cluster_order = sorted(df_qc['cluster_labels'].unique(), key=lambda x: int(x) if x.isdigit() else x)
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 6 * n_plots))
    if n_plots == 1:
        axes = [axes]  # Make it iterable
    
    for i, (metric, log_scale, title) in enumerate(metrics_to_plot):
        three_way_boxplot(df_qc, metric, 'cluster_labels', axes[i], cluster_order, log_scale, title)
    
    # Add overall title and filtering summary
    n_total_cells = len(df_qc)
    n_passed_cells = (df_qc['passed_qc'] == True).sum()
    n_failed_cells = n_total_cells - n_passed_cells
    
    fig.suptitle(f'DDQC Three-Way Comparison - {experimental_id}\n'
                f'All: {n_total_cells}, Pass: {n_passed_cells} ({n_passed_cells/n_total_cells*100:.1f}%), '
                f'Not-passed: {n_failed_cells} ({n_failed_cells/n_total_cells*100:.1f}%)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    
    # Save the combined plot
    output_path = os.path.join(plot_output_dir, f"{experimental_id}_ddqc_three_way.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"DDQC three-way comparison plot saved to: {output_path}")
    
    return adata_result