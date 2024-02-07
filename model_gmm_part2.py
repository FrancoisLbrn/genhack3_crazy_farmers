#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# Z,x |-> G_\theta(Z,x)
############################################################################

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>

import numpy as np
from joblib import load


def generative_model(noise, scenario):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim=4)
        input noise of the generative model
    scenario: ndarray with shape (n_samples, n_scenarios=9)
        input categorical variable of the conditional generative model
    """
    # See below an example
    # ---------------------
    latent_dims = 4
    # choose the appropriate latent dimension of your model
    latent_variable = noise[:, :latent_dims]

    # Loading the model
    scen = np.argmax(scenario[0]) + 1
    # OLD
    # model = load(f'parameters/gmm_part2/gmm_by_scenario/model_{scen}.joblib')
    # BEST
    model = load(f'parameters/gmm_part2/tuned_best/model_{scen}.joblib')

    # Getting the parameters
    weights = model.weights_
    means = model.means_
    covariances = model.covariances_

    # Simulating
    simul = np.zeros((4, noise.shape[0]))
    for j in range(noise.shape[0]):
        component_idx = np.random.choice(np.arange(len(weights)), p=weights)
        S = np.linalg.cholesky(covariances[component_idx])
        simul[:, j] = S @ latent_variable[j] + means[component_idx]

    simul = np.where(simul < 0, 0, simul)
    simul = np.where(simul > 15.75, 15.75, simul)

    return simul.T
    # return model(latent_variable, scenario) # G(Z, x)
