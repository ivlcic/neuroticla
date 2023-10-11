import os
import shutil
import logging
import numpy as np
import pandas as pd
import torch
import evaluate

from typing import List, Dict, Union, Callable, Any

from torch.nn.modules.module import T
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, \
    PreTrainedModel, PreTrainedTokenizer, TrainingArguments, Trainer

from .eval import ClassificationMetrics, MultilabelMetrics
from .labels import Labeler, MultiLabeler, BinaryLabeler
from .dataset import ClassifyDataset

logger = logging.getLogger('core.transformers')


class ModelContainer(torch.nn.Module):

    model_name_map = {
        'mcbert': 'bert-base-multilingual-cased',
        'xlmrb': 'xlm-roberta-base',
        'xlmrl': 'xlm-roberta-large'
    }

    @classmethod
    def remove_checkpoint_dir(cls, result_path: str):
        for rd in os.listdir(result_path):
            checkpoint_path = os.path.join(result_path, rd)
            if not rd.startswith('checkpoint'):
                continue
            if not os.path.isdir(checkpoint_path):
                continue
            moved = False
            for f in os.listdir(checkpoint_path):
                source_file_path = os.path.join(checkpoint_path, f)
                if not os.path.isfile(source_file_path):
                    continue
                target_file_path = os.path.join(result_path, f)
                shutil.move(source_file_path, target_file_path)
                moved = True
                logger.info('Moved [%s] -> [%s].', source_file_path, target_file_path)
            if moved:
                shutil.rmtree(checkpoint_path)
                logger.info('Removed checkpoint dir [%s].', checkpoint_path)

    def __init__(self, model_name_or_path: str, labeler: Labeler, cache_model_dir: Union[str, None] = None,
                 device: str = None, best_metric: str = 'macro'):
        super().__init__()

        self._labeler: Labeler = labeler
        self._model: Union[PreTrainedModel, None] = None
        self._metric = None
        self._best_metric = best_metric
        self._tokenizer: Union[PreTrainedTokenizer, None] = None
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=cache_model_dir
        )
        if device is None:
            use_cuda = torch.cuda.is_available()
            device = torch.device('cuda' if use_cuda else 'cpu')
            logger.info('Device was not set will use [%s].', device)
        self._device = device
        self._last_eval_metrics: Dict[str, Any] = {}
        self._eval_callback: Union[Callable, None] = None

    def max_len(self):
        return self._tokenizer.model_max_length

    def model(self):
        return self._model

    def eval(self):
        self._model.eval()

    def train(self: T, mode: bool = True) -> T:
        return super().train(mode)

    def labeler(self):
        return self._labeler

    def tokenizer(self):
        return self._tokenizer

    def metric(self):
        return self._metric

    def device(self):
        return self._device

    def destroy(self):
        del self._model
        del self._tokenizer
        torch.cuda.empty_cache()

    def compute_metrics(self, p):
        pass

    def last_eval_metrics(self):
        return self._last_eval_metrics

    def forward(self, input_id, mask, label):
        output = self._model(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output

    def build(self, args: TrainingArguments,
              train_set: ClassifyDataset,
              eval_set: ClassifyDataset,
              eval_callback: Union[Callable, None] = None):
        trainer = Trainer(
            model=self._model,
            args=args,
            train_dataset=train_set,
            eval_dataset=eval_set,
            tokenizer=self._tokenizer,
            compute_metrics=self.compute_metrics
        )
        logger.debug('Starting training...')
        self._eval_callback = None
        trainer.train()
        logger.info('Training done.')
        logger.debug('Starting evaluation...')
        self._eval_callback = eval_callback
        trainer.evaluate()
        logger.info('Evaluation done.')
        return self._last_eval_metrics

    def test(self, training_args: TrainingArguments, test_set: ClassifyDataset, callback: Callable = None):
        self._model.eval()
        trainer = Trainer(
            model=self._model,
            args=training_args,
            tokenizer=self._tokenizer,
            compute_metrics=self.compute_metrics
        )
        predictions, labels, _ = trainer.predict(test_set)
        self._eval_callback = callback
        self.compute_metrics((predictions, labels))
        return self._last_eval_metrics

    def infer_data_set(self, training_args: TrainingArguments, data_set: ClassifyDataset):
        self._model.eval()
        trainer = Trainer(
            model=self._model,
            args=training_args,
            tokenizer=self._tokenizer,
            compute_metrics=self.compute_metrics
        )
        logits, y_true, _ = trainer.predict(data_set)
        y_pred = np.argmax(logits, axis=1)

        if isinstance(self._labeler, MultiLabeler):  # here we have integer encoded every label combination
            decoded_predictions = []
            for p_sample in y_pred:
                decoded_predictions.append(self._labeler.binpowset(p_sample))
            decoded_labels = []
            y_pred = decoded_predictions

            if y_true is not None:
                for t_sample in y_true:
                    decoded_labels.append(self._labeler.binpowset(t_sample))
                y_true = decoded_labels

        return y_pred, y_true


class TokenClassifyModel(ModelContainer):

    def __init__(self, model_name_or_path: str, labeler: Labeler,
                 cache_model_dir: Union[str, None] = None, device: str = None, best_metric: str = 'overall_f1'):
        super(TokenClassifyModel, self).__init__(
            model_name_or_path, labeler, cache_model_dir, device, best_metric
        )

        self._metric = evaluate.load('seqeval')
        self._model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, cache_dir=cache_model_dir, num_labels=labeler.mun_labels(),
            id2label=labeler.ids2labels(), label2id=labeler.labels2ids()
        )
        self._model.to(self._device)

    def compute_metrics(self, p):
        logits, labels_list = p

        # select predicted index with maximum logit for each token
        predictions_list = np.argmax(logits, axis=2)

        tagged_predictions_list = []
        tagged_labels_list = []
        for predictions, labels in zip(predictions_list, labels_list):
            tagged_predictions = []
            tagged_labels = []
            for pid, lid in zip(predictions, labels):
                if lid != -100:
                    tagged_predictions.append(self._labeler.id2label(pid))
                    tagged_labels.append(self._labeler.id2label(lid))
            tagged_predictions_list.append(tagged_predictions)
            tagged_labels_list.append(tagged_labels)

        self._last_eval_metrics = self._metric.compute(
             references=tagged_labels_list, predictions=tagged_predictions_list, scheme='IOB2', mode='strict'
        )
        logger.info(
            'Using best metric %s from batch eval results: %s', self._best_metric, self._last_eval_metrics
        )
        if len(logger.handlers) > 0:
            logger.handlers[0].flush()
        if self._eval_callback is not None:
            self._eval_callback(self._labeler, tagged_labels_list, tagged_predictions_list)
        return {
            'precision': self._last_eval_metrics['overall_precision'],
            'recall': self._last_eval_metrics['overall_recall'],
            'f1': self._last_eval_metrics[self._best_metric],
            'accuracy': self._last_eval_metrics['overall_accuracy'],
        }

    def infer(self, word_list: Union[str, List[str]]) -> List[Dict[str, str]]:
        self._model.eval()
        is_split = False if isinstance(word_list, str) else True
        model_inputs = self._tokenizer(
            word_list,
            return_tensors='pt',
            truncation=False,
            is_split_into_words=is_split,
        )
        if len(model_inputs['input_ids'][0]) > self._tokenizer.model_max_length:
            sent = ' '.join(word_list)
            logger.warning(f'Truncated long input sentence:\n{sent}')
            model_inputs = self._tokenizer(
                word_list,
                return_tensors='pt',
                truncation=True,
                is_split_into_words=is_split,
            )

        model_inputs.to(self._device)
        with torch.no_grad():
            logits = self._model(**model_inputs)[0]
            # scores = logits.softmax(1).max(axis=1).values.numpy().tolist()
            predicted_classes = logits[0].argmax(axis=1).cpu().numpy().tolist()

        result: List[Dict[str, str]] = []
        word_ids = model_inputs.word_ids()
        for ix in range(1, len(model_inputs[0]) - 1):
            contd_cls_name = self._labeler.id2label(predicted_classes[ix], 'O')
            token_idx = word_ids[ix]
            if token_idx >= len(word_list):
                continue
            if token_idx < len(result):
                result[token_idx]['ner'] = contd_cls_name
            else:
                result.append({'text': word_list[token_idx], 'ner': contd_cls_name})
        return result


class SeqClassifyModel(ModelContainer):

    def __init__(self, model_name_or_path: str, labeler: Labeler,
                 cache_model_dir: Union[str, None] = None, device: str = None, best_metric: str = 'macro'):
        super(SeqClassifyModel, self).__init__(
            model_name_or_path, labeler, cache_model_dir, device, best_metric
        )
        self._metric = MultilabelMetrics()
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, cache_dir=cache_model_dir, num_labels=labeler.mun_labels(),
            id2label=labeler.ids2labels(), label2id=labeler.labels2ids()
        )
        self._model.to(self._device)

    def compute_metrics(self, p):
        logits, y_true = p

        # select predicted index with maximum logit for each token
        y_pred = np.argmax(logits, axis=1)

        labels = self._labeler.source_labels()
        if isinstance(self._labeler, MultiLabeler):  # here we have integer encoded every label combination
            decoded_predictions = []
            decoded_labels = []
            for p_sample, t_sample in zip(y_pred, y_true):
                decoded_predictions.append(self._labeler.binpowset(p_sample))
                decoded_labels.append(self._labeler.binpowset(t_sample))
            y_pred = decoded_predictions
            y_true = decoded_labels

        self._last_eval_metrics = self._metric.compute(references=y_true, predictions=y_pred, labels=labels)
        if self._eval_callback is not None:
            self._eval_callback(self._labeler, y_true, y_pred)

        logger.info(
            'Using best metric [%s] from batch eval results: [%s]',self._best_metric, self._last_eval_metrics
        )
        if len(logger.handlers) > 0:
            logger.handlers[0].flush()

        return {
            'precision': self._last_eval_metrics['avg'][self._best_metric]['p'],
            'recall': self._last_eval_metrics['avg'][self._best_metric]['r'],
            'f1': self._last_eval_metrics['avg'][self._best_metric]['f1'],
            'accuracy': self._last_eval_metrics['accuracy']
        }

    def infer(self, word_list: Union[str, List[str]]) -> str:
        self._model.eval()
        is_split = False if isinstance(word_list, str) else True
        model_inputs = self._tokenizer(
            word_list,
            return_tensors='pt',
            truncation=True,
            is_split_into_words=is_split
        )
        model_inputs.to(self._device)
        with torch.no_grad():
            logits = self._model(**model_inputs).logits
            predicted_class_id = logits.argmax().item()
            if isinstance(self._labeler, MultiLabeler):
                return self._labeler.binpowset(predicted_class_id)
            else:
                return self._labeler.id2label(predicted_class_id)
