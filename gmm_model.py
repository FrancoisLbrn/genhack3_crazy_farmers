import pandas as pd
import numpy as np
from joblib import load
import ot

yields_espsilon = pd.read_csv('CSVs/yields_subset.csv').iloc[:, 2:].to_numpy()
noise = np.load('data/noise.npy')[:, :4]
loaded_model = load('gmm.joblib')


weights = loaded_model.weights_
means = loaded_model.means_
covariances = loaded_model.covariances_

simul = np.zeros((4,10_000))
for j in range(10000):
    component_idx = np.random.choice(np.arange(len(weights)), p=weights)
    S = np.linalg.cholesky(covariances[component_idx])
    simul[:, j] = S @ noise[j, :] + means[component_idx]

simul = (simul * (simul > 0)).T

print(ot.sliced.sliced_wasserstein_distance(yields_espsilon, simul, seed=0))