import argparse
import json
import time
from collections import defaultdict
from math import log2

import pandas as pd
from joblib import Parallel, delayed
from textblob import TextBlob

from finetuning.training import augment
from logger_config import *
from utils.hugging_face import push_dataset
from utils.markdown import html2md, apply_transformation_for_non_code

logger = logging.getLogger('preprocessing')


def process_record(df_group):
    data_sft = []
    data_rlhf = []

    current_score = -10e8
    prev_score = -10e8
    prev_rows = defaultdict(list)

    for row_index, row in df_group.iterrows():
        row['QuestionBody'] = html2md(row['QuestionBody'])
        row['AnswerBody'] = html2md(row['AnswerBody'])
        row['AnswerScore'] = log2(1 + row['AnswerScore']) if row['AnswerScore'] > 0 else -1

        row['QuestionBody'] = apply_transformation_for_non_code(row['QuestionBody'], functions=[
            # spelling correction
            lambda s: str(TextBlob(s).correct())
        ])

        print('='*10, 'ORIGINAL')
        print(row['AnswerBody'])
        row['AnswerBody'] = augment(row['AnswerBody'])
        print('='*10, 'AUGMENTED')
        print(row['AnswerBody'])
        print('='*10, 'PREPROCESSING')
        row['AnswerBody'] = apply_transformation_for_non_code(row['AnswerBody'], functions=[
            # spelling correction
            lambda s: str(TextBlob(s).correct())
        ])
        print(row['AnswerBody'])
        print('='*100)

        data_sft.append({
            'question_id': row['QuestionId'],
            'question': row['QuestionBody'],

            'answer_id': row['AnswerId'],
            'answer': row['AnswerBody'],
            'answer_score': row['AnswerScore'],
        })

        if current_score != row['AnswerScore']:
            prev_score = current_score
            current_score = row['AnswerScore']
        if len(prev_rows[prev_score]) and prev_score != row['AnswerScore']:
            for prev_row in prev_rows[prev_score]:
                rlhf_d = {
                    'answer_j_id': row['AnswerId'],
                    'answer_j': row['AnswerBody'],
                    'answer_j_score': row['AnswerScore'],

                    'answer_k_id': row['AnswerId'],
                    'answer_k': prev_row['AnswerBody'],
                    'answer_k_score': prev_row['AnswerScore'],
                }
                data_rlhf.append(rlhf_d)

        prev_rows[row['AnswerScore']].append(row)

    return data_sft, data_rlhf


def export_for_labelling(data: list[dict], path: str, id_key: str = 'answer_id'):
    data_json = []
    for d in data:
        data_json.append({
            'id': d[id_key],
            'data': d
        })

    with open(path, 'w') as f:
        json.dump(data_json, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Raw data preprocessing')
    parser.add_argument('--branch', type=str, default='main', help='Dataset version for HuggingFace')
    parser.add_argument('--data-file', dest='data_file', type=str, default='data/QueryResults.csv', help='Raw data file')
    parser.add_argument('--split', type=str, default='train', help='Dataset split type')
    parser.add_argument('--n_jobs', type=int, default=1, help='Run N parallel jobs')
    parser.add_argument('--export-for-labeling',  dest='export_for_labeling', action='store_true', help='Export data for labelling')
    parser.add_argument('--push-huggingface', dest='push_huggingface', action='store_true', help='Push to HuggingFace')
    args = parser.parse_args()

    df = pd.read_csv(args.data_file)[:10]
    logger.debug('Imported columns: %s', df.columns)
    # logger.debug(df.describe())
    df_grouped = df.sort_values(['QuestionId', 'AnswerScore'], ascending=True).groupby('QuestionId')
    # logger.debug(df_grouped[['QuestionId', 'AnswerId', 'AnswerScore']].head(10))

    logger.info('Data preprocessing...')
    data_sft = []
    data_rlhf = []
    start = time.time()

    if args.n_jobs > 1:
        # Parallel
        out = Parallel(n_jobs=24, backend='multiprocessing')(
            delayed(process_record)(df_group) for group_name, df_group in df_grouped
        )
        for out_i in out:
            data_sft.extend(out_i[0])
            data_rlhf.extend(out_i[1])
    else:
        # Sequential
        for group_name, df_group in df_grouped:
            out = process_record(df_group)
            data_sft.extend(out[0])
            data_rlhf.extend(out[1])

    logger.info('Done: %0.5fs', time.time() - start)

    logger.info('Data SFT %d samples', len(data_sft))
    logger.info('Data RLHF %d samples', len(data_rlhf))

    if args.export_for_labeling:
        logger.info('Export for dataset labelling...')
        export_for_labelling(data_sft, 'data/sft_for_labelling.json')
        export_for_labelling(data_rlhf, 'data/rlhf_for_labelling.json', id_key='answer_j_id')
        logger.info('Done')

    if args.push_huggingface:
        logger.info('Uploading sft dataset...')
        push_dataset("asavanovich/sft_dataset", data_sft, branch=args.branch, split=args.split)
        logger.info('Done')

        logger.info('Uploading reward dataset...')
        push_dataset("asavanovich/reward_dataset", data_rlhf[:len(data_rlhf)//2], branch=args.branch, split=args.split)
        logger.info('Done')

        logger.info('Uploading rlhf dataset...')
        push_dataset("asavanovich/rlhf_dataset", data_rlhf[len(data_rlhf)//2:], branch=args.branch, split=args.split)
        logger.info('Done')
