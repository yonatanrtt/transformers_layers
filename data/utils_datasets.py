from datasets import load_dataset

def get_data(_dataset_name): 
    dataset = "super_glue"
    data = load_dataset(dataset, "")
    return data

# copa | cb | rte