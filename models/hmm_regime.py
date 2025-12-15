# models/hmm_regime.py
import numpy as np
from hmmlearn.hmm import GaussianHMM


class RegimeHMM:
    def __init__(self, n_states=2, covariance_type="full", n_iter=200, random_state=42):
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        self.fitted = False

    def fit(self, returns_df):
        """
        Fit HMM to a multivariate return series (DataFrame: dates x assets).
        """
        X = returns_df.values
        self.model.fit(X)
        self.fitted = True

    def predict_states(self, returns_df):
        """
        Return most likely state sequence for given returns.
        """
        if not self.fitted:
            raise RuntimeError("HMM not fitted yet.")
        X = returns_df.values
        states = self.model.predict(X)
        return states

    def state_probabilities(self, returns_df):
        """
        Return state posterior probabilities for each date.
        """
        if not self.fitted:
            raise RuntimeError("HMM not fitted yet.")
        X = returns_df.values
        logprob, posteriors = self.model.score_samples(X)
        return posteriors
