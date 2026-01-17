import pandas as pd
import numpy as np

class RobustFeatureEngineering:
    """
    Implementation of the RFE framework:
    1. Meta-Information Tensors (Missing patterns)
    2. High-dimensional Windowed Features (O(N) complexity)
    """
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.feature_cols = ['GR', 'RHOB', 'NPHI', 'DTC', 'RDEP', 'SP', 'CALI'] # Adjust based on data
        
    def _augment_features_window(self, X, N_neig):
        """
        Efficient linear-time window aggregation (O(N))
        """
        N_row = X.shape[0]
        N_feat = X.shape[1]
        
        # Zero padding
        X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))
        
        # Vectorized concatenation (avoiding deep loops)
        X_feat = np.zeros((N_row, (2*N_neig+1)*N_feat))
        for r in np.arange(N_row) + N_neig:
            # Flatten window into a single feature vector
            X_feat[r-N_neig] = X[r-N_neig : r+N_neig+1].flatten()
            
        return X_feat

    def build_meta_information(self, df):
        """
        Construct Meta-Information Tensors to capture data quality.
        """
        # 1. Binary Mask for Missing Values (using -999 sentinel)
        missing_mask = (df[self.feature_cols] == -999).astype(int)
        
        # 2. Aggregated Statistics
        meta_features = pd.DataFrame()
        meta_features['missing_count'] = missing_mask.sum(axis=1)
        meta_features['missing_ratio'] = missing_mask.mean(axis=1)
        
        # 3. Rolling Quality Metrics (Spatial continuity of quality)
        # O(N) complexity using pandas rolling
        for col in self.feature_cols:
            # Calculate local missing density
            meta_features[f'{col}_missing_trend'] = missing_mask[col].rolling(
                window=self.window_size, center=True).mean().fillna(0)
            
        return meta_features

    def transform(self, df):
        """
        Main pipeline to transform raw logs into RFE features.
        """
        print("Starting RFE Transformation...")
        
        # 1. Meta-Information Extraction
        meta_df = self.build_meta_information(df)
        
        # 2. Windowed Feature Augmentation
        # Normalize data first (standard practice)
        X_numeric = df[self.feature_cols].values
        # (Add scaler logic here if needed, e.g., StandardScaler)
        
        X_windowed = self._augment_features_window(X_numeric, N_neig=1) 
        
        # 3. Concatenate
        X_final = np.hstack([X_windowed, meta_df.values])
        
        print(f"Feature Engineering Complete. Output shape: {X_final.shape}")
        return X_final

# Usage Example:
# rfe = RobustFeatureEngineering(window_size=5)
# X_train_processed = rfe.transform(train_df)
