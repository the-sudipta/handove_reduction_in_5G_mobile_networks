"""Paper phase two: K-Means + PCA + Decision Tree feature extractor.

  * K-Means (K = 5) groups the cells by signal strength, quality and load
    -> `pci_cluster` of the serving cell;
  * PCA (3 components) compresses [RSRP, CINR, RSRQ, distance, speed, cluster];
  * a depth-5 Decision Tree on the PCA output predicts p_HO, the probability
    that handing over to the candidate is beneficial.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ..history import STATE_COLUMNS

TREE_INPUTS = ["rsrp", "cinr", "rsrq", "distance", "speed"]


class EnsembleFeatures:
    def __init__(self, clusters=5, components=3, tree_depth=5, seed=0):
        self.kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=seed)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=components, random_state=seed)
        self.tree = DecisionTreeClassifier(max_depth=tree_depth, random_state=seed)

    def fit(self, events, cell_stats):
        self.cell_cluster = self.kmeans.fit_predict(StandardScaler().fit_transform(cell_stats))
        x = self._tree_inputs(events[TREE_INPUTS].to_numpy(), events["serving"].to_numpy(), fit=True)
        self.tree.fit(x, events["label"].to_numpy())
        return self

    def transform(self, raw: np.ndarray, serving: np.ndarray):
        """raw: rows of STATE_COLUMNS. Returns (pci_cluster, p_ho)."""
        x = self._tree_inputs(raw[:, :len(TREE_INPUTS)], serving)
        return self.cell_cluster[serving], self.tree.predict_proba(x)[:, 1]

    def _tree_inputs(self, measurements, serving, fit=False):
        x = np.column_stack([measurements, self.cell_cluster[serving]])
        if fit:
            return self.pca.fit_transform(self.scaler.fit_transform(x))
        return self.pca.transform(self.scaler.transform(x))


assert STATE_COLUMNS[:len(TREE_INPUTS)] == TREE_INPUTS   # raw rows start with the tree inputs
