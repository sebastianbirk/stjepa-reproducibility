import argparse
import math
import pickle
import tracemalloc
from collections import defaultdict
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import scanpy as sc


def main(args):
    data_directory_path = Path(args.data_directory_path)

    # Read dataset names
    datasets_df = pd.read_csv(
        args.cohort_dataset_names_file_path,
        index_col=0,
        header=None)
    print(datasets_df)

    tracemalloc.start()

    if not args.compute_median_stats:
        print("Computing norm factors based on individual batches.")
        running_n_cells = 0
        batchwise_norm_factor_dict = {}

        for dataset_name, _ in datasets_df.iterrows():
            print('Processing: ', dataset_name)
            
            dataset_directory_path = data_directory_path / dataset_name

            for file in dataset_directory_path.glob('*.h5ad'):
                adata = sc.read_h5ad(file)
                running_n_cells += len(adata)

                # Compute batchwise nonzero means and means per gene
                X_dense = adata.X.toarray()
                nonzero_mask = X_dense > 0
                nonzero_counts = np.sum(nonzero_mask, axis=0)
                nonzero_sums = np.sum(X_dense * nonzero_mask, axis=0)
                nonzero_means = np.true_divide(nonzero_sums, nonzero_counts)
                means = X_dense.mean(0)

                # Perform gene corrected read depth normalization
                X = adata.X / adata.X.sum(axis=1).reshape(-1, 1) * (
                    args.basis_target_read_depth + adata.X.shape[1] *
                    args.target_read_depth_per_gene)

                # Compute batchwise gene corrected read depth normalized nonzero
                # means and means per gene
                X_dense_gcrd = X.toarray()
                nonzero_mask_gcrd = X_dense_gcrd > 0
                nonzero_counts_gcrd = np.sum(nonzero_mask_gcrd, axis=0)
                nonzero_sums_gcrd = np.sum(X_dense_gcrd * nonzero_mask_gcrd, axis=0)
                nonzero_means_gcrd = np.true_divide(
                    nonzero_sums_gcrd, nonzero_counts_gcrd)
                means_gcrd = X_dense_gcrd.mean(0)

                # Store batchwise norm factors
                for idx, gene_id in enumerate(adata.var['ensembl_id'].values):
                    if math.isnan(nonzero_means_gcrd[idx]): # math.isnan(nonzero_means[idx]) is equivalent
                        nonzero_means[idx] = 0.0
                        nonzero_means_gcrd[idx] = 0.0
                        n_cells = 0
                    else:
                        n_cells = len(adata)
                        n_nonzero_cells = nonzero_counts[idx]
                    if f"{gene_id}_n_cells" in batchwise_norm_factor_dict.keys():
                        batchwise_norm_factor_dict[f"{gene_id}_nonzero_means"].append(nonzero_means[idx])
                        batchwise_norm_factor_dict[f"{gene_id}_means"].append(means[idx])
                        batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_nonzero_means"].append(nonzero_means_gcrd[idx])
                        batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_means"].append(means_gcrd[idx])
                        batchwise_norm_factor_dict[f"{gene_id}_n_cells"].append(n_cells)
                        batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"].append(n_nonzero_cells)
                    else:
                        batchwise_norm_factor_dict[f"{gene_id}_nonzero_means"] = [nonzero_means[idx]]
                        batchwise_norm_factor_dict[f"{gene_id}_means"] = [means[idx]]
                        batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_nonzero_means"] = [nonzero_means_gcrd[idx]]
                        batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_means"] = [means_gcrd[idx]]
                        batchwise_norm_factor_dict[f"{gene_id}_n_cells"] = [n_cells]
                        batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"] = [n_nonzero_cells]

                print(f"Running number of cells: {running_n_cells}.")

                current, peak = tracemalloc.get_traced_memory()
                print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB.")

                # Write to disk
                with open(f'{args.results_directory_path}/{args.cohort_name}/{args.cohort_name}_batchwise_norm_factors.pkl', 'wb') as file:
                    pickle.dump(batchwise_norm_factor_dict, file)

        # Compute overall nonzero means and means per gene and store cell numbers
        norm_factor_dict = {}
        gene_ids = list(set([key.split('_')[0] for key in batchwise_norm_factor_dict.keys()]))
        
        for gene_id in gene_ids:
            norm_factor_dict[gene_id] = {}
            norm_factor_dict[gene_id]["nonzero_mean"] = np.dot(
                np.array(batchwise_norm_factor_dict[f"{gene_id}_nonzero_means"]),
                np.array(batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"])
            ) / np.array(batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"]).sum()
            norm_factor_dict[gene_id]["mean"] = np.dot(
                np.array(batchwise_norm_factor_dict[f"{gene_id}_means"]),
                np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"])
            ) / np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"]).sum()
            norm_factor_dict[gene_id]["gene_corrected_read_depth_nonzero_mean"] = np.dot(
                np.array(batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_nonzero_means"]),
                np.array(batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"])
            ) / np.array(batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"]).sum()
            norm_factor_dict[gene_id]["gene_corrected_read_depth_mean"] = np.dot(
                np.array(batchwise_norm_factor_dict[f"{gene_id}_gene_corrected_read_depth_means"]),
                np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"])
            ) / np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"]).sum()
            norm_factor_dict[gene_id]["n_cells"] = np.array(batchwise_norm_factor_dict[f"{gene_id}_n_cells"]).sum()
            norm_factor_dict[gene_id]["n_nonzero_cells"] = np.array(batchwise_norm_factor_dict[f"{gene_id}_n_nonzero_cells"]).sum()

        norm_factor_df = pd.DataFrame.from_dict(norm_factor_dict, orient='index').reset_index()
        norm_factor_df = norm_factor_df.rename(columns={'index': 'gene_id'})

        # Write to disk
        print("Saving norm factor DataFrame.")
        norm_factor_df.to_csv(
            f'{args.results_directory_path}/{args.cohort_name}/{args.cohort_name}_norm_factors.csv',
            index=False)

    if args.compute_median_stats:
        print("Computing norm factors based on concatenated corpus.")
        running_n_cells = 0
        gene_count_dict = defaultdict(lambda: np.empty((0,)))
        gene_gcrd_count_dict = defaultdict(lambda: np.empty((0,)))

        for dataset_name, _ in datasets_df.iterrows():
            print('Processing: ', dataset_name)
            
            dataset_directory_path = data_directory_path / dataset_name

            for file in dataset_directory_path.glob('*.h5ad'):
                adata = sc.read_h5ad(file)
                adata.X = adata.X.toarray()
                running_n_cells += len(adata)
                for i, gene in enumerate(adata.var['ensembl_id']):
                    values = adata.X[:, i]
                    gene_count_dict[gene] = np.concatenate(
                        [gene_count_dict[gene], values])

                # Perform gene corrected read depth normalization
                X = adata.X / adata.X.sum(axis=1).reshape(-1, 1) * (
                    args.basis_target_read_depth + adata.X.shape[1] *
                    args.target_read_depth_per_gene)

                for i, gene in enumerate(adata.var['ensembl_id']):
                    gcrd_values = X[:, i]
                    gene_gcrd_count_dict[gene] = np.concatenate(
                        [gene_gcrd_count_dict[gene], gcrd_values])

                print(f"Running number of cells: {running_n_cells}.")
        
                current, peak = tracemalloc.get_traced_memory()
                print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB.")

        # Write to disk
        print("Saving gene count dictionary.")
        with open(f'{args.results_directory_path}/{args.cohort_name}/{args.cohort_name}_gene_counts.pkl', 'wb') as file:
            pickle.dump(dict(gene_count_dict), file)
        with open(f'{args.results_directory_path}/{args.cohort_name}/{args.cohort_name}_gene_gcrd_counts.pkl', 'wb') as file:
            pickle.dump(dict(gene_gcrd_count_dict), file)

        norm_factor_dict = {}

        for gene in gene_count_dict.keys():
            norm_factor_dict[gene] = {}
            median = np.median(gene_count_dict[gene])
            mean = np.mean(gene_count_dict[gene])
            nz_values = gene_count_dict[gene][gene_count_dict[gene] != 0]
            if nz_values.size > 0:
                nz_median = np.median(nz_values)
                nz_mean = np.mean(nz_values)
            else:
                nz_median = np.nan
                nz_mean = np.nan

            norm_factor_dict[gene]['median'] = median
            norm_factor_dict[gene]['mean'] = mean
            norm_factor_dict[gene]['nonzero_median'] = nz_median
            norm_factor_dict[gene]['nonzero_mean'] = nz_mean

        for gene in gene_gcrd_count_dict.keys():
            median = np.median(gene_gcrd_count_dict[gene])
            mean = np.mean(gene_gcrd_count_dict[gene])
            nz_values = gene_gcrd_count_dict[gene][gene_gcrd_count_dict[gene] != 0]
            if nz_values.size > 0:
                nz_median = np.median(nz_values)
                nz_mean = np.mean(nz_values)
            else:
                nz_median = np.nan
                nz_mean = np.nan

            norm_factor_dict[gene]['gene_corrected_read_depth_median'] = median
            norm_factor_dict[gene]['gene_corrected_read_depth_mean'] = mean
            norm_factor_dict[gene]['gene_corrected_read_depth_nonzero_median'] = nz_median
            norm_factor_dict[gene]['gene_corrected_read_depth_nonzero_mean'] = nz_mean
            norm_factor_dict[gene]['n_cells'] = int(len(gene_gcrd_count_dict[gene]))
            norm_factor_dict[gene]['n_nonzero_cells'] = int(len(nz_values))

        norm_factor_df = pd.DataFrame(norm_factor_dict).T
        norm_factor_df.reset_index(inplace=True)
        norm_factor_df.rename(columns={'index': 'gene_id'}, inplace=True)

        # Write to disk
        print("Saving norm factor DataFrame.")
        norm_factor_df.to_csv(
            f'{args.results_directory_path}/{args.cohort_name}/{args.cohort_name}_norm_factors.csv',
            index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Processing command line arguments...')
    parser.add_argument('--data_directory_path',
                        type=str,
                        default="/nfs/team361/sb75/DATASETS/silver",
                        help='Data directory path.')
    parser.add_argument('--results_directory_path',
                        type=str,
                        default="/nfs/team361/sb75/sttokbench/artifacts/data",
                        help='Results directory path.')
    parser.add_argument('--cohort_name',
                        type=str,
                        default="mst_corpus_10m",
                        help='Name of the cohort.')
    parser.add_argument('--cohort_dataset_names_file_path',
                        type=str,
                        default="/nfs/team361/sb75/sttokbench/analysis/tokenization/dataset_list.txt",
                        help='Path to the file containing datasets of the cohort.')
    parser.add_argument("--compute_median_stats",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Whether to compute median stats (default: True).")                        
    parser.add_argument("--basis_target_read_depth",
                        type=float,
                        default=-752.364435146444,
                        help="Read depth independent of number of probed genes.")
    parser.add_argument("--target_read_depth_per_gene",
                        type=float,
                        default=0.8921817095038856,
                        help="Additional read depth increment per probed gene.")
    args = parser.parse_args()
    
    main(args)