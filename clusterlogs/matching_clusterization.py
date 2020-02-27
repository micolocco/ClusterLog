import pandas as pd
import difflib
import editdistance
from .tokenization import Tokens
import nltk
from itertools import groupby


class SClustering:

    def __init__(self, groups, tokens, accuracy):
        self.groups = groups
        self.tokens = tokens
        self.accuracy = accuracy


    def matching_clusterization(self):
        """
        Clusterization messages using sequence matching
        :param df:
        :param accuracy:
        :return:
        """
        result = []
        self.reclustering(self.groups.copy(deep=True), result)
        self.result = pd.DataFrame(result)
        print('Postprocessed with {} clusters'.format(self.result.shape[0]))
        return self.result.sort_values(by=['cluster_size'], ascending=False)



    def reclustering(self, df, result):
        """
        Clusterization of the groups:
        - take the 1st message (pattern) and compare if with others
        - take all messages, which are similar with the 1st with more than 80% and
        join them into the new separate cluster
        - remove these messages from the initial group
        - repeat these steps while group has messages
        :param df:
        :param result:
        :param accuracy:
        :return:
        """
        top_sequence = df['sequence'].describe().top
        df['ratio'] = self.levenshtein_similarity(top_sequence, df['sequence'].values)
        filtered = df[(df['ratio'] >= self.accuracy)]
        tokenized_pattern = self.matcher(filtered['tokenized_pattern'].values)
        indices = [item for sublist in filtered['indices'].values for item in sublist]
        result.append({'pattern': Tokens.detokenize_row(Tokens.TOKENIZER, tokenized_pattern),
                       'indices': indices,
                       'cluster_size': len(indices),
                       'sequence':tokenized_pattern})
        df.drop(filtered.index, axis=0, inplace=True)
        while df.shape[0] > 0:
            self.reclustering(df, result)



    def matcher(self, lines):
        if len(lines) > 1:
            fdist = nltk.FreqDist([i for l in lines for i in l])
            #x = [token for token in lines[0] if (fdist[token] / len(lines) >= 1)]
            x = [token if (fdist[token]/len(lines) >= 1) else '｟*｠' for token in lines[0]]
            return [i[0] for i in groupby(x)]
        else:
            return lines[0]



    def sequence_matcher(self, sequences):
        if len(sequences) > 1:
            pattern = sequences[0]
            for i in range(1, len(sequences)):
                matches = difflib.SequenceMatcher(None, pattern, sequences[i])
                m = [pattern[m.a:m.a + m.size] for m
                     in matches.get_matching_blocks() if m.size > 0]
                pattern = [val for sublist in m for val in sublist]
            return Tokens.detokenize_row(Tokens.TOKENIZER, pattern)
        else:
            return Tokens.detokenize_row(Tokens.TOKENIZER, sequences[0])



    def levenshtein_similarity(self, top, rows):
        """
        :param rows:
        :return:
        """
        if len(rows) > 1:
            return (
                [(1 - editdistance.eval(top, rows[i]) / max(len(top), len(rows[i]))) for i in
                 range(0, len(rows))])
        else:
            return 1
