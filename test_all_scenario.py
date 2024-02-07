import numpy as np
import ot
import pandas as pd
from main import simulate

SWDs = []
for i in range(1, 10):
    SCENARIO_NUMBER = i  # <- PICK A SCENARIO NUMBER BETWEEN 1 AND 9
    noise = np.load("data/noise.npy")
    # create a vector of zeros with shape (n_samples, 9)
    scenario = np.zeros((noise.shape[0], 9))
    scenario[:, SCENARIO_NUMBER-1] = 1
    output = simulate(noise, scenario)
    yields_df = pd.read_csv(f'CSVs/scenarios/scenario{i}.csv').iloc[:, -6:-2]
    swd = ot.sliced.sliced_wasserstein_distance(
        output, yields_df.to_numpy(), n_projections=1000)
    SWDs.append(swd)

for i, e in enumerate(SWDs):
    print(f'scenario {i+1} : {e}')

size_scenario = [464, 1290, 1678, 534, 1254, 1082, 1007, 1690, 1001]

s = 0
for i in range(len(size_scenario)):
    poids = (1 - size_scenario[i]/10_000)
    s += poids * SWDs[i]
    print(f'poids {i+1} : {poids}')
print('Weighted SWD :', s)
