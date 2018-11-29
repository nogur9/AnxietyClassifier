from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
class RemoveCorrelationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, correlation_threshold=0.7):
        self.correlation_threshold = correlation_threshold

    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        df_in = pd.DataFrame(X)
        df_corr = df_in.corr(method='pearson', min_periods=1)
        # if use_pca:
        #     df_not_correlated = ~(np.tril(np.ones([len(df_corr)] * 2, dtype=bool)).abs() > self.correlation_threshold).any()
        #
        #     df_correlated = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == False].index
        #     pca = PCA(n_components=n)
        #     corr_out =

        df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)] * 2, dtype=bool))).abs() > self.correlation_threshold).any()

        un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
        df_out = df_in[un_corr_idx]
        return df_out.values

# cutoff high correlation
