import os

import pandas as pd
from datasets import Dataset


def push_dataset(name, data_dict, access_token=os.environ.get('HUGGING_FACE_TOKEN'), **kwargs):
    sft_df = pd.DataFrame(data_dict)
    sft_dataset = Dataset.from_pandas(sft_df)
    sft_dataset.push_to_hub(name, private=True, token=access_token, **kwargs)
