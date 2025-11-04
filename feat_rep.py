import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import config as c


def load_and_merge_data(feature_csv: str, metadata_csv: str) -> pd.DataFrame:
    """Load feature and metadata CSVs, remove duplicates, and merge on 'eid'."""
    # Load data
    df_features = pd.read_csv(feature_csv)
    df_metadata = pd.read_csv(metadata_csv)
    
    # Remove duplicates
    print(f"Duplicates in features: {df_features['eid'].duplicated().sum()}")
    print(f"Duplicates in metadata: {df_metadata['eid'].duplicated().sum()}")
    
    df_features = df_features.drop_duplicates(subset='eid', keep='first')
    df_metadata = df_metadata.drop_duplicates(subset='eid', keep='first')
    
    # Merge
    df = pd.merge(df_features, df_metadata, on='eid', how='inner')
    print(f"Merged dataframe shape: {df.shape}")
    
    return df


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Extract numeric feature columns (columns with digit names)."""
    feature_cols = [col for col in df.columns if col.isdigit()]
    X = df[feature_cols].values
    print(f"Feature matrix shape: {X.shape}")
    return X


def compute_embedding(
    X: np.ndarray, 
    method: str = 'umap',
    n_neighbors: int = 10, 
    min_dist: float = 1.0,
    perplexity: int = 30,
    n_components: int = 2,
    random_state: int = 3
) -> np.ndarray:
    """
    Compute dimensionality reduction embedding from feature matrix.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        method: Reduction method - 'umap', 'tsne', or 'pca'
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        perplexity: Perplexity parameter for t-SNE
        n_components: Number of output dimensions (default: 2)
        random_state: Random seed for reproducibility
        
    Returns:
        Embedding coordinates (n_samples, n_components)
    """
    method = method.lower()
    
    if method == 'umap':
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric='euclidean',
            random_state=random_state
        )
        embedding = reducer.fit_transform(X)
        print(f"✅ UMAP embedding shape: {embedding.shape}")
        
    elif method == 'tsne':
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000
        )
        embedding = reducer.fit_transform(X)
        print(f"✅ t-SNE embedding shape: {embedding.shape}")
        
    elif method == 'pca':
        reducer = PCA(
            n_components=n_components,
            random_state=random_state
        )
        embedding = reducer.fit_transform(X)
        explained_var = reducer.explained_variance_ratio_
        print(f"✅ PCA embedding shape: {embedding.shape}")
        print(f"   Explained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}")
        
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'umap', 'tsne', 'pca'")
    
    return embedding


def plot_embedding_with_metadata(
    embedding: np.ndarray,
    metadata: np.ndarray,
    metadata_name: str,
    save_path: str,
    palette: dict = None,
    point_size: int = 500,
    alpha: float = 0.5,
    figsize: tuple = (10, 5),
    show_legend: bool = True,
    discrete: bool = True,
    bins: list = None,
    legend_fontsize: int = 30,
    legend_title_fontsize: int = 35,
    colorbar_fontsize: int = 10,
    colorbar_labelsize: int = 30,
    frameon: bool = True,
):
    """
    Plot UMAP embedding with metadata overlay.
    
    Args:
        embedding: UMAP coordinates (n_samples, 2)
        metadata: Metadata values to color by
        metadata_name: Name of metadata for labels
        save_path: Path to save the figure
        palette: Color palette for discrete variables
        point_size: Size of scatter points
        alpha: Transparency of points
        figsize: Figure size tuple
        show_legend: Whether to show legend
        discrete: Whether metadata is discrete or continuous
        bins: Bins for discretizing continuous metadata
        legend_fontsize: Font size for legend text
        legend_title_fontsize: Font size for legend title
        colorbar_fontsize: Font size for colorbar ticks
        colorbar_labelsize: Font size for colorbar label
        frameon: Whether to show frame around legend
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not discrete:
        # Continuous variable: use colorbar
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=metadata.astype(float),
            cmap='viridis_r',
            s=point_size,
            alpha=alpha,
            linewidth=0
        )
        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)

    else:
        # Discrete variable: use seaborn with legend
        if bins:
            metadata_binned = pd.cut(metadata.astype(float), bins=bins).astype(str)
        else:
            metadata_binned = metadata

        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=metadata_binned,
            palette=palette,
            s=point_size,
            alpha=alpha,
            linewidth=0,
            legend=show_legend,
            ax=ax
        )

        if show_legend:
            ax.legend(
                fontsize=legend_fontsize,
                title_fontsize=legend_title_fontsize,
                loc='upper right',
                frameon=frameon
            )

    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {save_path}")


def get_metadata_columns(df: pd.DataFrame) -> list:
    """
    Automatically detect metadata columns (non-feature columns).
    
    Args:
        df: Merged dataframe with features and metadata
        
    Returns:
        List of metadata column names
    """
    # Exclude feature columns (digit names), eid, and embedding coordinates
    exclude_patterns = ['eid', 'EEG_ID', 
                       'UMAP-1', 'UMAP-2', 
                       'TSNE-1', 'TSNE-2',
                       'PCA-1', 'PCA-2']
    
    metadata_cols = [
        col for col in df.columns 
        if not col.isdigit() and col not in exclude_patterns
    ]
    
    return metadata_cols


def infer_column_type(series: pd.Series) -> tuple:
    """
    Infer if a column is discrete or continuous and suggest a palette.
    
    Args:
        series: Pandas series to analyze
        
    Returns:
        Tuple of (is_discrete, palette)
    """
    # Remove NaN values for analysis
    series_clean = series.dropna()
    
    if len(series_clean) == 0:
        return True, None
    
    # Check if string/object type -> discrete
    if series_clean.dtype == 'object':
        unique_vals = series_clean.unique()
        n_unique = len(unique_vals)
        
        # Generate palette for categorical variables
        if n_unique <= 10:
            # Use a predefined color scheme
            colors = ['#4682B4', '#DC143C', '#32CD32', '#FF8C00', 
                     '#9370DB', '#FFD700', '#FF69B4', '#00CED1', 
                     '#8B4513', '#2E8B57']
            palette = {val: colors[i % len(colors)] for i, val in enumerate(unique_vals)}
            return True, palette
        else:
            return True, None
    
    # Check if numeric
    elif pd.api.types.is_numeric_dtype(series_clean):
        unique_vals = series_clean.unique()
        n_unique = len(unique_vals)
        
        # If few unique values, treat as discrete
        if n_unique <= 10:
            colors = ['#4682B4', '#DC143C', '#32CD32', '#FF8C00', 
                     '#9370DB', '#FFD700', '#FF69B4', '#00CED1', 
                     '#8B4513', '#2E8B57']
            palette = {val: colors[i % len(colors)] for i, val in enumerate(sorted(unique_vals))}
            return True, palette
        else:
            # Many unique values -> continuous
            return False, None
    
    return True, None


def main():
    """Main execution function."""
    # Configuration
    feature_csv = os.path.join(c.FEATURES_EXT_DIR, '_features.csv')
    metadata_csv = c.DATA_PATH
    viz_dir = c.VIZ_DIR
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get reduction method from config (default to UMAP if not specified)
    reduction_method = getattr(c, 'REDUCTION_METHOD', 'umap').lower()
    print(f"\n🔬 Using reduction method: {reduction_method.upper()}")
    
    # Load and prepare data
    df = load_and_merge_data(feature_csv, metadata_csv)
    
    # Extract features and compute embedding
    X = extract_features(df)
    embedding = compute_embedding(
        X, 
        method=reduction_method,
        n_neighbors=getattr(c, 'N_NEIGHBORS', 10),
        min_dist=getattr(c, 'MIN_DIST', 1.0),
        perplexity=getattr(c, 'PERPLEXITY', 30),
        random_state=getattr(c, 'RANDOM_STATE', 3)
    )
    
    # Add embedding to dataframe with method-specific column names
    method_prefix = reduction_method.upper()
    df[f'{method_prefix}-1'] = embedding[:, 0]
    df[f'{method_prefix}-2'] = embedding[:, 1]
    
    # Automatically detect all metadata columns
    metadata_columns = get_metadata_columns(df)
    print(f"\nDetected metadata columns: {metadata_columns}")
    
    # Plot each metadata column
    for label in metadata_columns:
        # Skip columns with all NaN values
        if df[label].isna().all():
            print(f"⚠️  Skipping '{label}' - all values are NaN")
            continue
        
        # Save with method prefix to distinguish different reduction methods
        viz_path = os.path.join(viz_dir, f'{label}.png')
        label_array = df[label].values
        
        # Automatically infer column type and palette
        discrete, palette = infer_column_type(df[label])
        
        print(f"\nPlotting '{label}' - Discrete: {discrete}, Unique values: {df[label].nunique()}")
        
        # Plot and save
        plot_embedding_with_metadata(
            embedding=embedding,
            metadata=label_array,
            metadata_name=label.upper(),
            save_path=viz_path,
            palette=palette,
            discrete=discrete,
            point_size=c.POINT_SIZE,
            alpha=c.TRANSPARENCY,
            legend_fontsize= c.FONTSIZE_MIN,
            legend_title_fontsize= c.FONTSIZE_MAX,
            colorbar_fontsize=c.FONTSIZE_MIN,
            colorbar_labelsize=c.FONTSIZE_MIN,
            frameon=True
        )


if __name__ == "__main__":
    main()