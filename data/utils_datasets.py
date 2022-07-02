from datasets import load_dataset

def get_data(_dataset_name): 
    dataset = "super_glue"
    data = load_dataset(dataset, _dataset_name)
    return data

# copa | cb | rte