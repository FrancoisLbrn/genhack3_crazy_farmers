##############################################
# DO NOT MODIFY THIS FILE
##############################################

from model import generative_model
import numpy as np
import logging

##
import numpy as np
import random
from os.path import join

from utils import prepare_dirs, save_config
from config import get_config
from trainer import Trainer

import torch
##


logging.basicConfig(filename="check.log", level=logging.DEBUG, 
                    format="%(asctime)s:%(levelname)s: %(message)s", 
                    filemode='w')

def main(config):
    
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # ensure directories are setup
    trial_dir = join(config.logs_folder, config.dataset, 'trial'+str(config.trial_num))
    prepare_dirs(trial_dir, config.flush)
    if config.is_train:
        try:
            save_config(trial_dir, config)
        except ValueError:
            print(
                "[!] file already exist. Either change the trial number,",
                "or delete the json file and rerun.",
                sep=' ',
            )
    trainer = Trainer(trial_dir, config)

    if config.is_train: 
        trainer.train()
    else:
        trainer.test()




def simulate(noise):
    """
    Simulation of your Generative Model

    Parameters
    ----------
    noise : ndarray
        input of the generative model

    Returns
    -------
    output: array (noise.shape[0], 4)
        Generated yield containing respectively the 4 stations (49, 80, 40, 63)
    """

    try:
        output = generative_model(noise)
        message = "Successful simulation" 
        assert output.shape == (noise.shape[0], 4), "Shape error, it must be (noise.shape[0], 4). Please verify the shape of the output."
        
        # write the output
        np.save("output.npy", output)

    except Exception as e:
        message = e
                
    finally:
        logging.debug(message)

    return output

    
if __name__ == "__main__":
    noise = np.load("data/noise.npy")

    config, unparsed = get_config()
    main(config)

    simulate(noise)
    
    
