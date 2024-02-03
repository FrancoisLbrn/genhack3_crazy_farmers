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
# Z |-> G_\theta(Z)
############################################################################
import numpy as np
from joblib import load

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim=4)
        input noise of the generative model
    """
    # See below an example
    # ---------------------
    latent_dims = 4
    latent_variable = noise[:, :latent_dims]  # choose the appropriate latent dimension of your model

    # load your parameters or your model
    # <!> be sure that they are stored in the parameters/ directory <!>
    loaded_model = load('parameters/gmm.joblib')

    weights = loaded_model.weights_
    means = loaded_model.means_
    covariances = loaded_model.covariances_

    simul = np.zeros((4,noise.shape[0]))
    for j in range(noise.shape[0]):
        component_idx = np.random.choice(np.arange(len(weights)), p=weights)
        S = np.linalg.cholesky(covariances[component_idx])
        simul[:, j] = S @ latent_variable[j, :].T + means[component_idx]

    simul = (simul * (simul > 0)).T

    return simul # G(Z)




