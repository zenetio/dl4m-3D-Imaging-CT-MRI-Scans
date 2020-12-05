"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import random
import numpy as np

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"../../section1/out/TrainingSet"
        self.n_epochs = 12
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "../out"

if __name__ == "__main__":
    # Get configuration

    # Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()
    # where am i?
    x = os.getcwd()
    # Load data
    print("Loading data...")

    #data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)
    data = LoadHippocampusData(c.root_dir, x_shape = c.patch_size, y_shape = c.patch_size)

    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = range(len(data))
    all_idx = list(keys)
    random.shuffle(all_idx)

    #train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 
    # lets split our dataset as 60, 20, 20
    train, validate, test = np.split(all_idx, [int(.6 * len(all_idx)), int(.8 * len(all_idx))])

    split = dict()

    # create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    split = {"train": train, "val": validate, "test": test}

    # Set up and run experiment
    
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    # del dataset 

    # summary
    exp.print_summary()
    
    # run training
    exp.run()
    #exp.load_model_parameters(path='..\\out\\model.pth')

    # prep and run testing

    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))
