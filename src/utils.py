import os

import anndata as ad
import scanpy as sc


def collect_adata_from_folder(load_folder_path: str,
                              cell_ids: list[str] | None = None,
                              dataset_ids: list[str] | None = None,
                              obs_cols: list[str] | None = None,
                              uns_cols: list[str] | None = None,
                              include_gene_panel_size: bool = True,
                              ) -> ad.AnnData:
    """
    Loop through folder, read all `.h5ad` files and concatenate them
    into one AnnData object.

    Parameters
    --------
    load_folder_path:
        Directory which is searched for AnnData objects.
    cell_ids:
        IDs of cells which are included in the final object. If `None`,
        all cells are included.
    dataset_ids:
        IDs of datasets which are included in the final object. If
        `None`, all subdirectories are included.
    obs_cols:
        Columns in `adata.obs` that should be retained in the final
        object.
    uns_cols:
        Keys in `adata.uns` that should be retained in the final
        object.

    Returns
    --------
    adata:
        The concatenated AnnData object.
    """
    adata_list = []

    # Walk through the load folder path and read files
    for subdir, _, files in os.walk(load_folder_path):
        print(subdir)
        print(subdir.split('/')[-1].split('-')[0])
        if dataset_ids:
            if not any(dataset_id in subdir.split('/')[-1].split('-')[0] for dataset_id in dataset_ids):
                continue
        print(f'Loading AnnData objects from {subdir}.')
        for file in files:
            if file.endswith('.h5ad'):
                file_path = os.path.join(subdir, file)
                adata = sc.read_h5ad(file_path)
                if cell_ids:
                    adata = adata[adata.obs['cell_id'].isin(cell_ids)]
                if len(adata) == 0:
                    del adata
                    continue
                if obs_cols is None:
                    adata.obs = adata.obs[['cell_id']]
                else:
                    adata.obs = adata.obs[['cell_id'] + [
                        col for col in obs_cols if col in adata.obs.columns]]
                if uns_cols:
                    for col in uns_cols:
                        if col in adata.uns:
                            adata.obs[col] = adata.uns[col]
                if include_gene_panel_size:
                    adata.obs['gene_panel_size'] = len(adata.var_names)
                adata_list.append(adata)        

    concatenated_adata = ad.concat(adata_list, join='outer', index_unique=None)
    
    return concatenated_adata