import argparse
import logging
import re
import sys
import time
from collections import defaultdict
from math import log2

import pandas as pd
from joblib import Parallel, delayed
from markdownify import markdownify as md

from utils.hugging_face import push_dataset

file_handler = logging.FileHandler(filename='logs/preprocessing.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

logger = logging.getLogger('preprocessing')

# REG_CODE_BLOCK = re.compile(r'^(([ \t]*`{3,4})([^\n]*)([\s\S]+?)(^[ \t]*\2))')
REG_CODE_BLOCK = re.compile(r'^`')

def lang_callback(el):
    lang = el['class'][0] if el.has_attr('class') else None

    if not lang is None:
        lang = lang.split("-")[-1]
    return lang


def html2md(text):
    text = md(text, code_language_callback=lang_callback)
    text = re.sub(r"\n\s*\n", "\n\n", text).strip()
    return text.encode('utf-8', 'replace').decode()
    # return text

def process_question(df_group):
    data_sft = []
    data_rlhf = []

    current_score = -10e8
    prev_score = -10e8
    prev_rows = defaultdict(list)

    for row_index, row in df_group.iterrows():
        row['AnswerBody'] = html2md(row['AnswerBody'])
        row['QuestionBody'] = html2md(row['QuestionBody'])
        row['AnswerScore'] = log2(1 + row['AnswerScore']) if row['AnswerScore'] > 0 else -1

        if "```" in row['AnswerBody']:
            print(re.findall(REG_CODE_BLOCK, row['AnswerBody']))
        # if "```" in row['QuestionBody']:
        #     print(re.findall(REG_CODE_BLOCK, row['QuestionBody']))

        data_sft.append({
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
    parser.add_argument('--branch', type=str, default='main', help='Dataset version for HuggingFace')
    parser.add_argument('--data-file', dest='data_file', type=str, default='data/QueryResults.csv', help='Raw data file')
    parser.add_argument('--split', type=str, default='train', help='Dataset split type')
    args = parser.parse_args()

    df = pd.read_csv(args.data_file)
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
    # push_dataset("asavanovich/sft_dataset", data_sft, branch=args.branch, split=args.split)
    # logger.info('Done')
    #
    # logger.info('Uploading reward dataset...')
    # push_dataset("asavanovich/reward_dataset", data_rlhf[:len(data_rlhf)//2], branch=args.branch, split=args.split)
    # logger.info('Done')
    #
    # logger.info('Uploading rlhf dataset...')
    # push_dataset("asavanovich/rlhf_dataset", data_rlhf[len(data_rlhf)//2:], branch=args.branch, split=args.split)
    # logger.info('Done')
