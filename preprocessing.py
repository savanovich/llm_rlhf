import argparse
import logging
import sys
import time
from collections import defaultdict

import pandas as pd
from joblib import Parallel, delayed

file_handler = logging.FileHandler(filename='logs/preprocessing.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

logger = logging.getLogger('preprocessing')

INPUT_FILE = "data/QueryResults.csv"


def process_question(df_group):
    data_sft = []
    data_rlhf = []

    current_score = -10e8
    prev_score = -10e8
    prev_rows = defaultdict(list)

    for row_index, row in df_group.iterrows():
        data_sft.append({
            'link': row['Post Link'],
            'qid': row['QuestionId'],
            'question': row['QuestionBody'],
            'response_j': row['AnswerBody'],
            'answer_j_score': row['AnswerScore'],
        })

        if current_score != row['AnswerScore']:
            prev_score = current_score
            current_score = row['AnswerScore']
        if len(prev_rows[prev_score]) and prev_score != row['AnswerScore']:
            for prev_row in prev_rows[prev_score]:
                rlhf_d = {
                    'link': row['Post Link'],
                    'qid': row['QuestionId'],

                    'answer_j': row['AnswerBody'],
                    'answer_j_score': row['AnswerScore'],

                    'answer_k': prev_row['AnswerBody'],
                    'answer_k_score': prev_row['AnswerScore'],
                }
                data_rlhf.append(rlhf_d)

        prev_rows[row['AnswerScore']].append(row)

    # print(row['QuestionId'])

    return data_sft, data_rlhf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Raw data preprocessing')
    parser.add_argument('--branch', dest='branch', type=str, default='main', help='Dataset version for HuggingFace')
    args = parser.parse_args()

    df = pd.read_csv(INPUT_FILE)
    # print(df.columns)
    # print(df.describe())
    df_grouped = df.sort_values(['QuestionId', 'AnswerScore'], ascending=True).groupby('QuestionId')
    # print(df_grouped[['QuestionId', 'AnswerId', 'AnswerScore']].head(10))

    logger.info('Data preprocessing...')
    data_sft = []
    data_rlhf = []
    start = time.time()

    # Parallel
    out = Parallel(n_jobs=24, backend='multiprocessing')(
        delayed(process_question)(df_group) for group_name, df_group in df_grouped
    )
    for out_i in out:
        data_sft.extend(out_i[0])
        data_rlhf.extend(out_i[1])

    # Sequential
    # for group_name, df_group in df_grouped:
    #     out = process_question(df_group)
    #     data_sft.extend(out[0])
    #     data_rlhf.extend(out[1])

    logger.info('Done: %0.5f ms', time.time() - start)

    logger.info('Data SFT %d samples', len(data_sft))
    logger.info('Data RLHF %d samples', len(data_rlhf))

    # logger.info('Uploading sft dataset...')
    # push_dataset("asavanovich/sft_dataset", data_sft, branch=args.branch)
    # logger.info('Done')
    #
    # logger.info('Uploading rlhf dataset...')
    # push_dataset("asavanovich/rlhf_dataset", data_rlhf, branch=args.branch)
    # logger.info('Done')
