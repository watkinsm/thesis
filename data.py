import csv
import delegator
import json
import os
import re
import jsonlines

import pandas as pd
import numpy as np
import seaborn as sn

from collections import Counter


class Data():
    @staticmethod
    def clean_atis_2_data(filename, mode, output='./data/atis'):
        output_file = output.rstrip('/') + f'/atis_{mode}.tsv'

        data = [f'{re.sub(r"^(BOS)|(EOS)$", "", row[0])}\tatis_{row[1].rsplit("atis_")[1].strip("#")}'
        # data = [f'{re.sub(r"^(BOS)|(EOS)$", "", row[0])}\tatis_{row[1].rsplit("atis_")[1:]}'
                for row in
                    csv.reader(
                        open(filename),
                        delimiter='\t')]

        if not os.path.exists(output):
            os.makedirs(output)
        with open(output_file, 'wt') as f:
            f.write('\n'.join(data))

        print(f'Processed {len(data)} lines')
        
        return output_file

    @staticmethod
    def clean_snips_data(directory, mode, extra='', output='./data/snips', name='snips'):
        if not os.path.isdir(directory):
            raise Exception('Provided path is not a directory')

        output_file = output.rstrip('/') + f'/{name}_{mode}.tsv'
        directory = directory.rstrip('/')

        cmd = f'cd {directory} && find . {extra}'
        files = delegator.run(cmd).out.strip('\n').split('\n')

        data = []
        for f in files:
            k, v = list(
                json.loads(
                    open(f'{directory}/{f}').read()).items())[0]

            data.extend(
                [f'{"".join(text)}\t{k}'
                    for text in
                        [[part['text']
                            for part in query['data']]
                            for query in v]])

        if not os.path.exists(output):
            os.makedirs(output)
        with open(output_file, 'wt') as f:
            f.write('\n'.join(data))

        print(f'Processed {len(data)} lines')

        return output_file

    @staticmethod
    def clean_almawave_sl2_data(directory, mode, extra='', output='./data/Almawave_SLU_1.0',
                                name='aw_slu'):
        if not os.path.isdir(directory):
            raise Exception('Provided path is not a directory')

        output_file = output.rstrip('/') + f'/{name}_{mode}.tsv'
        directory = directory.rstrip('/')

        cmd = f'cd {directory} && find . |grep {mode}'
        files = delegator.run(cmd).out.strip('\n').split('\n')

        data = []
        for fn in files:
            with jsonlines.open(f'{directory}/{fn}') as v:
                for query in v:
                    intent = query.get('intent')
                    sentence = query.get('sentence')
                    data.append(f'{sentence}\t{intent}')

        if not os.path.exists(output):
            os.makedirs(output)
        with open(output_file, 'wt') as f:
            f.write('\n'.join(data))

        print(f'Processed {len(data)} lines')

        return output_file


def make_intent_dist_barplot(filename, dataset):
    df = pd.read_csv(filename, delimiter='\t', header=None,
                     names=['sentence', 'intent']).dropna(how='any')

    cnt = Counter(df.intent)
    keys = np.array(list(cnt.keys()))
    vals = np.array(list(cnt.values())).astype(float)

    chart = sn.barplot(x=keys, y=vals, palette='muted')
    chart.set_ylabel('count')
    chart.set_xlabel('intent')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    # chart.set_title(f'Distribution of {dataset} intents')

    return chart


if __name__ == '__main__':
    atis_2_cleaned_dev = Data.clean_atis_2_data('./data/atis-2/atis-2.dev.w-intent.tsv', mode='dev')
    # snips_cleaned_dev = Data.clean_snips_data(
    #     './data/nlu-benchmark/2017-06-custom-intent-engines/',
    #     mode='validate',
    #     extra='|grep validate_')
    # snips_cleaned_train = Data.clean_snips_data(
    #     './data/nlu-benchmark/2017-06-custom-intent-engines/',
    #     mode='train',
    #     extra='|grep "train_" |grep -v "full"')

    atis_2_cleaned_train = Data.clean_atis_2_data('data/atis-2/atis-2.train.w-intent.tsv',
                                                  mode='train')

    # df_atis = pd.read_csv(atis_2_cleaned_dev, delimiter='\t', header=None, names=['sentence', 'intent'])
    # df_snips = pd.read_csv(snips_cleaned_train, delimiter='\t', header=None, names=['sentence', 'intent'])

    # print(df_atis.head())
    # print(df_snips.head())

    # almawave_slu_test_cleaned = Data.clean_almawave_sl2_data(
    #     'data/Almawave_SLU_1.0/',
    #     mode='test',
    #     name='aw_slu',
    #     output='data/aw_slu')

    # almawave_slu_train_cleaned = Data.clean_almawave_sl2_data(
    #     'data/Almawave_SLU_1.0/',
    #     mode='train',
    #     name='aw_slu',
    #     output='data/aw_slu')
    #
    # df_almwave_slu = pd.read_csv(almawave_slu_train_cleaned, delimiter='\t', header=None,
    #                              names=['sentence', 'intent'])


