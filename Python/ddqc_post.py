#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import scanpy as sc
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from typing import Set, Tuple, Union


# Import existing DDQC functionality
from ddqc_scanpy import ddqc_scanpy


def load_genes(
    taxid: Union[str, int],
    qc_gene_list_path: str
) -> Tuple[Set[str], Set[str]]:
    """
    Load mitochondrial and ribosomal genes from CSV file.
    Enhanced version leveraging DDQC patterns.
    """
    taxid = str(taxid)

    mito_genes: Set[str] = set()
    ribo_genes: Set[str] = set()

    with open(qc_gene_list_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        required_cols = {"TaxID", "Gene_symbol", "Group"}
        if not required_cols.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain columns {required_cols}, "
                f"found {reader.fieldnames}"
            )

        for row in reader:
            if row["TaxID"] != taxid:
                continue

            gene = row["Gene_symbol"]
            group = row["Group"].lower()

            if group == "mitochondrial":
                mito_genes.add(gene)
            elif group == "ribosomal":
                ribo_genes.add(gene)

    return mito_genes, ribo_genes


def three_way_boxplot(df, metric, cluster_col, ax, cluster_order, log_scale=False, title=""):
    """
    Create three-way DDQC comparison boxplots using enhanced plotting.
    Shows all cells vs passed cells vs not-passed cells for each cluster.
    Leverages DDQC boxplot_sorted functionality.
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


def enhanced_ddqc_with_plots(adata, cluster_key="louvain", method="mad", threshold=2.0, 
                           threshold_counts=0, threshold_genes=0, threshold_mito=0, threshold_ribo=0,
                           plot_output_dir="../plots/", target_srx=""):
    """
    Enhanced DDQC function that leverages existing DDQC modules and creates comprehensive plots.
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
    
    fig.suptitle(f'DDQC Three-Way Comparison - {target_srx}\n'
                f'All: {n_total_cells}, Pass: {n_passed_cells} ({n_passed_cells/n_total_cells*100:.1f}%), '
                f'Not-passed: {n_failed_cells} ({n_failed_cells/n_total_cells*100:.1f}%)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    
    # Save the combined plot
    output_path = os.path.join(plot_output_dir, f"{target_srx}_ddqc.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"--- Box plot saved to: {output_path}")
    
    return adata_result


def create_doublet_plots(adata, target_srx, cluster_key="louvain", plot_output_dir="../plots/"):
    """
    Create doublet detection result plots including score distribution and per-cluster analysis.
    Enhanced version with DDQC plotting patterns.
    """
    # Ensure output directory exists
    os.makedirs(plot_output_dir, exist_ok=True)
    
    # Check if doublet detection results are available
    if 'doublet_score' not in adata.obs.columns:
        print("Warning: doublet_score not found in adata.obs. Skipping doublet plots.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Doublet score distribution histogram
    ax1 = axes[0]
    doublet_scores = adata.obs['doublet_score']
    predicted_doublets = adata.obs['predicted_doublet']
    
    # Plot histogram for all cells
    ax1.hist(doublet_scores, bins=50, alpha=0.7, color='lightblue', label='All cells', density=True)
    
    # Overlay histogram for predicted doublets
    doublet_scores_positive = doublet_scores[predicted_doublets]
    if len(doublet_scores_positive) > 0:
        ax1.hist(doublet_scores_positive, bins=30, alpha=0.7, color='red', 
                label=f'Predicted doublets (n={len(doublet_scores_positive)})', density=True)
    
    # Add mean value labels on the histogram
    singlet_scores = doublet_scores[~predicted_doublets]
    mean_singlet = singlet_scores.mean()
    mean_doublet = doublet_scores_positive.mean() if len(doublet_scores_positive) > 0 else 0
    
    # Add mean labels with corresponding colors
    ax1.axvline(mean_singlet, color='blue', linestyle='--', alpha=0.8)
    ax1.text(mean_singlet, ax1.get_ylim()[1] * 1.05, f'Singlet mean: {mean_singlet:.3f}', 
             color='blue', fontweight='bold', ha='center', va='bottom')
    
    if len(doublet_scores_positive) > 0:
        ax1.axvline(mean_doublet, color='red', linestyle='--', alpha=0.8)
        ax1.text(mean_doublet, ax1.get_ylim()[1] * 1.05, f'Doublet mean: {mean_doublet:.3f}', 
                 color='red', fontweight='bold', ha='center', va='bottom')
    
    ax1.set_xlabel('Doublet Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Doublet Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Doublet score by cluster boxplot
    ax2 = axes[1]
    if cluster_key in adata.obs.columns:
        # Create dataframe for plotting
        plot_data = pd.DataFrame({
            'doublet_score': adata.obs['doublet_score'],
            'cluster': adata.obs[cluster_key],
            'predicted_doublet': adata.obs['predicted_doublet']
        })
        
        # Order clusters naturally
        cluster_order = sorted(plot_data['cluster'].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        
        # Create boxplot
        sns.boxplot(data=plot_data, x='cluster', y='doublet_score', order=cluster_order, ax=ax2)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Doublet Score')
        ax2.set_title('Doublet Scores by Cluster')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add doublet counts per cluster as text annotations
        for i, cluster in enumerate(cluster_order):
            cluster_data = plot_data[plot_data['cluster'] == cluster]
            n_doublets = cluster_data['predicted_doublet'].sum()
            n_total = len(cluster_data)
            ax2.text(i, ax2.get_ylim()[1] * 0.95, f'{n_doublets}/{n_total}', 
                    ha='center', va='top', fontsize=8, weight='bold', color='red')
    else:
        ax2.text(0.5, 0.5, f'Cluster key "{cluster_key}" not found', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Doublet Scores by Cluster (N/A)')
    
    # 3. Per-cluster doublet statistics
    ax3 = axes[2]
    if cluster_key in adata.obs.columns:
        cluster_stats = []
        for cluster in sorted(adata.obs[cluster_key].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            cluster_data = adata.obs[adata.obs[cluster_key] == cluster]
            n_cells = len(cluster_data)
            n_doublets_cluster = cluster_data['predicted_doublet'].sum()
            doublet_rate_cluster = n_doublets_cluster / n_cells * 100 if n_cells > 0 else 0
            mean_score_cluster = cluster_data['doublet_score'].mean()
            
            cluster_stats.append({
                'Cluster': cluster,
                'Total': n_cells,
                'Doublets': n_doublets_cluster,
                'Rate(%)': doublet_rate_cluster,
                'Mean_Score': mean_score_cluster
            })
        
        # Create bar plot of doublet rates per cluster
        cluster_df = pd.DataFrame(cluster_stats)
        bars = ax3.bar(range(len(cluster_df)), cluster_df['Rate(%)'], color='lightcoral', alpha=0.7)
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Doublet Rate (%)')
        ax3.set_title('Doublet Rate by Cluster')
        ax3.set_xticks(range(len(cluster_df)))
        ax3.set_xticklabels(cluster_df['Cluster'], rotation=45)
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, cluster_df['Rate(%)'])):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    else:
        ax3.text(0.5, 0.5, f'Cluster key "{cluster_key}" not found', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Doublet Rate by Cluster (N/A)')
    
    # Calculate summary statistics for title
    total_cells = len(adata.obs)
    n_doublets = adata.obs['predicted_doublet'].sum()
    doublet_rate = n_doublets / total_cells * 100
    
    # Overall title
    fig.suptitle(f'Doublet Detection Results - {target_srx}\n'
                f'Total: {total_cells:,} cells, Doublets: {n_doublets:,} ({doublet_rate:.1f}%)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the plot
    output_path = os.path.join(plot_output_dir, f"{target_srx}_doublet.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"--- Doublet detection plots saved to: {output_path}")


def create_enhanced_umap_plots(adata, target_srx, cluster_key="louvain", plot_output_dir="../plots/"):
    """
    Create and save combined UMAP plot with clusters and cell types side by side.
    Enhanced version leveraging DDQC plotting patterns.
    """
    # Ensure output directory exists
    os.makedirs(plot_output_dir, exist_ok=True)
    
    # Check if cell_type annotation exists
    if 'cell_type' in adata.obs.columns:
        # Combined UMAP plot (side by side)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left panel: clusters
        sc.pl.umap(adata, color=cluster_key, legend_loc='on data', 
                   title=f'Clusters', frameon=False, ax=axes[0], 
                   save=False, show=False)
        
        # Right panel: cell types  
        sc.pl.umap(adata, color='cell_type', legend_loc='right margin',
                   title=f'Cell Types', frameon=False, ax=axes[1],
                   save=False, show=False)
        
        # Overall title
        fig.suptitle(f'UMAP: Clusters vs Cell Types - {target_srx}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        output_path_combined = os.path.join(plot_output_dir, f"{target_srx}_umap_combined.png")
        plt.savefig(output_path_combined, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"--- UMAP combined plot saved to: {output_path_combined}")
        
    else:
        # Fallback: only cluster plot if no cell types available
        plt.figure(figsize=(10, 8))
        sc.pl.umap(adata, color=cluster_key, legend_loc='on data', 
                   title=f'UMAP - Clusters\n{target_srx}',
                   frameon=False, save=False, show=False)
        
        output_path = os.path.join(plot_output_dir, f"{target_srx}_umap_clusters.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"--- UMAP clusters plot saved to: {output_path}")
        print("--- Cell type annotations not found, only clusters plotted")