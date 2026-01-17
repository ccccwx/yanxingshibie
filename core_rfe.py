
import pandas as pd
import numpy as np

class RobustFeatureEngineering:
    """
    Implementation of the RFE framework proposed in the paper.
    Key Components:
    1. Meta-Information Tensors (Explicitly capturing missing patterns)
    2. High-dimensional Windowed Features (Efficient O(N) complexity aggregation)
    """
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        # Standard well-log curves used in the study
        self.feature_cols = ['GR', 'RHOB', 'NPHI', 'DTC', 'RDEP', 'SP', 'CALI'] 
        
    def _augment_features_window(self, X, N_neig):
        """
        Performs window-based feature augmentation with O(N) time complexity.
        Instead of nested loops, we use vectorized padding and striding.
        """
        N_row = X.shape[0]
        N_feat = X.shape[1]
        
        # Zero padding for boundary conditions
        X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))
        
        # Efficient concatenation
        X_feat = np.zeros((N_row, (2*N_neig+1)*N_feat))
        for r in np.arange(N_row) + N_neig:
            # Flatten window into a single feature vector
            X_feat[r-N_neig] = X[r-N_neig : r+N_neig+1].flatten()
            
        return X_feat

    def build_meta_information(self, df):
        """
        Constructs Meta-Information Tensors to quantify data quality.
        This allows the model to learn from 'missingness' patterns.
        """
        # 1. Binary Mask for Missing Values (using -999 sentinel)
        missing_mask = (df[self.feature_cols] == -999).astype(int)
        
        # 2. Aggregated Statistics (Sample-wise quality)
        meta_features = pd.DataFrame()
        meta_features['meta_missing_count'] = missing_mask.sum(axis=1)
        meta_features['meta_missing_ratio'] = missing_mask.mean(axis=1)
        
        # 3. Rolling Quality Metrics (Spatial continuity of quality)
        # Uses pandas rolling() for optimized linear-time execution
        for col in self.feature_cols:
            # Calculate local missing density (trend)
            meta_features[f'meta_{col}_trend'] = missing_mask[col].rolling(
                window=self.window_size, center=True).mean().fillna(0)
            
        return meta_features

    def transform(self, df):
        """
        Main pipeline to transform raw logs into RFE features.
        """
        print(f"Starting RFE Transformation with window size {self.window_size}...")
        
        # Step 1: Meta-Information Extraction
        meta_df = self.build_meta_information(df)
        
        # Step 2: Windowed Feature Augmentation
        # (Note: Data normalization is assumed to be done prior to this step)
        X_numeric = df[self.feature_cols].values
        
        # N_neig=1 implies a window size of 3 (center + 1 left + 1 right)
        # Adjust N_neig based on experimental requirements
        X_windowed = self._augment_features_window(X_numeric, N_neig=1) 
        
        # Step 3: Feature Concatenation (Physical + Meta features)
        X_final = np.hstack([X_windowed, meta_df.values])
        
        print(f"Feature Engineering Complete. Output shape: {X_final.shape}")
        return X_final

# Usage Example:
# rfe = RobustFeatureEngineering(window_size=5)
# X_train_processed = rfe.transform(train_df)
