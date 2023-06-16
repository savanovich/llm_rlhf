import os

import pandas as pd
from datasets import Dataset


def push_dataset(name, data_dict, access_token=os.environ.get('HUGGING_FACE_TOKEN'), **kwargs):
    df = pd.DataFrame(data_dict)
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(name, private=True, token=access_token, **kwargs)
