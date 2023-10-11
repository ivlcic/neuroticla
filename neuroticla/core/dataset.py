import torch
import logging
import pandas as pd

from typing import List, Union

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding
from tokenizers.tokenizers import Encoding

from .labels import Labeler, MultiLabeler

logger = logging.getLogger('core.dataset')


class ClassifyDataset(Dataset):
    def __init__(self, labeler: Labeler, tokenizer: PreTrainedTokenizer, max_seq_len: int,
                 label_field: Union[str, List[str], None] = 'label', text_field: Union[str, List[str]] = 'text'):
        self._labeler = labeler
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._label_field = label_field
        self._text_field = text_field

    def __getitem__(self, index):
        return None


class TokenClassifyDataset(ClassifyDataset):

    def align_labels(self, encoded: Encoding, labels: List[str]):
        word_ids = encoded.word_ids
        label_ids = []
        max_idx = len(labels)
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx < 0 or word_idx >= max_idx:
                label_ids.append(-100)
            else:
                label_ids.append(self._labeler.label2id(labels[word_idx]))
        return label_ids

    def __init__(self, labeler: Labeler, tokenizer: PreTrainedTokenizer, data: pd.DataFrame, max_seq_len: int,
                 label_field: Union[str, List[str]] = 'label', text_field: Union[str, List[str]] = 'text'):
        """Encodes the text data and labels
        """
        super().__init__(labeler, tokenizer, max_seq_len, label_field, text_field)
        if not isinstance(label_field, str):
            raise ValueError('Only single field for label is supported')
        if not isinstance(text_field, str):
            raise ValueError('Only single field for text is supported')

        ds_labels = [self._labeler.filter_replace(line).split() for line in data[label_field].values.tolist()]

        # check if labels in the dataset are also in labeler
        true_labels = self._labeler.kept_labels()
        unique_labels = set()
        for lb in ds_labels:
            [unique_labels.add(i) for i in lb if i not in unique_labels]
        if unique_labels != true_labels:
            logger.warning("Unexpected label [%s] in [%s] in dataset!", unique_labels, true_labels)
            # exit(1)

        # encode the text
        texts = data[text_field].values.tolist()
        self.encodings: BatchEncoding = tokenizer(
            texts, padding='max_length', max_length=max_seq_len, truncation=True, return_tensors="pt"
        )
        # encode the labels
        self.labels = []
        for i, e in enumerate(self.encodings.encodings):
            self.labels.append(self.align_labels(e, ds_labels[i]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        self.encodings = {
          'input_ids':[dataset_size, [encoded ids vector]]
          'attention_mask':[dataset_size, [attention mask - padded tokens]]
        }
        so we need to take only the "idx" one
        item = {
          'input_ids':[encoded ids vector for idx]
          'attention_mask':[attention mask - padded tokens for idx]
        }
        """
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


class SeqClassifyDataset(ClassifyDataset):

    def __init__(self, labeler: Labeler, tokenizer: PreTrainedTokenizer, data: pd.DataFrame, max_seq_len: int,
                 label_field: Union[str, List[str]] = 'label', text_fields: Union[str, List[str]] = 'text'):
        """Encodes the text data and labels
                """
        super().__init__(labeler, tokenizer, max_seq_len, label_field, text_fields)

        # encode the text
        if isinstance(text_fields, str):
            text_fields = [text_fields]

        texts = []
        for i, row in data.iterrows():
            concatenated_string = '\n'.join([str(row[col]) for col in text_fields])
            texts.append(concatenated_string)

        self.encodings: BatchEncoding = tokenizer(
            texts,
            padding='max_length',
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt"
        )
        # encode the labels
        # single label field
        self.labels = []
        if isinstance(label_field, str) or len(label_field) == 1:
            lf_name = label_field
            if not isinstance(label_field, str):
                lf_name = label_field[0]
            for label in data[lf_name].values.tolist():
                self.labels.append(label)
        elif isinstance(labeler, MultiLabeler):
            for index, row in data.iterrows():
                label_list = [l for l in label_field if row[l] == 1]
                encoded = labeler.encode(label_list)
                self.labels.append(encoded)
        else:
            ValueError('MultiLabeler should be used with more than one label field!')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        self.encodings = {
          'input_ids':[dataset_size, [encoded ids vector]]
          'attention_mask':[dataset_size, [attention mask - padded tokens]]
        }
        so we need to take only the "idx" one
        item = {
          'input_ids':[encoded ids vector for idx]
          'attention_mask':[attention mask - padded tokens for idx]
        }
        """
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


class SeqEvalDataset(ClassifyDataset):

    def __init__(self, labeler: Labeler, tokenizer: PreTrainedTokenizer, data: pd.DataFrame, max_seq_len: int,
                 text_fields: Union[str, List[str]] = 'text'):
        """Encodes the text data and labels
                """
        super().__init__(labeler, tokenizer, max_seq_len, None, text_fields)

        # encode the text
        if isinstance(text_fields, str):
            text_fields = [text_fields]

        texts = []
        for i, row in data.iterrows():
            concatenated_string = '\n'.join([str(row[col]) for col in text_fields])
            texts.append(concatenated_string)
        self._len = len(texts)
        self.encodings: BatchEncoding = tokenizer(
            texts,
            padding='max_length',
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """
        self.encodings = {
          'input_ids':[dataset_size, [encoded ids vector]]
          'attention_mask':[dataset_size, [attention mask - padded tokens]]
        }
        so we need to take only the "idx" one
        item = {
          'input_ids':[encoded ids vector for idx]
          'attention_mask':[attention mask - padded tokens for idx]
        }
        """
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        return item
