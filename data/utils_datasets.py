from datasets import load_dataset
import ipdb
import pandas as pd 
import shared.constants as constants

def read_data(_task):
    dataset = "super_glue"
    data = load_dataset(dataset, _task)
    return data

def get_data(_task):
    ds = read_data(_task)
    train = pd.DataFrame(ds["train"]).sample(frac=1, random_state=1)
    data = dict(
        train=train[:250], 
        validation=pd.DataFrame(ds["validation"])[:56], 
        test=pd.DataFrame(ds["test"])
    )
    return data
# copa | cb | rte
