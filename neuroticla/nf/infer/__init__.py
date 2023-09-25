import pandas as pd
from transformers import TrainingArguments

from .common import run_inference
from ...core.labels import BinaryLabeler
from ...core.trans import SeqClassifyModel
from ...nf.utils import *

logger = logging.getLogger('nf.infer')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.train(parser)
    CommonArguments.raw_data_dir(module_name, parser, ('-i', '--data_in_dir'))
    CommonArguments.result_dir(module_name, parser, ('-o', '--result_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    parser.add_argument(
        '--max_seq_len', help='Max sub-word tokens length.', type=int, default=512
    )
    parser.add_argument(
        '--tqdm', help='Enable TDQM.', action='store_true', default=False
    )
    parser.add_argument(
        '-n', '--model_name', type=str, default=None,
        help='Target model name. (overrides other settings used for model name construction)',
    )
    labels = ','.join(get_all_labels())
    parser.add_argument(
        '-u', '--subset', type=str, default=None, required=False,
        help='Subset of the labels to use for training (comma separated: ' + labels + ')',
    )
    text_fields = ','.join(get_all_text_fields())
    parser.add_argument(
        '-f', '--train_fields', type=str, default='body', required=False,
        help='Text fields to use for testing: ' + text_fields + ')',
    )
    parser.add_argument(
        '-p', '--pretrained_model', type=str, default=None, required=False,
        help='Pretrained model that was used for fine tuning (used only for model name construction)',
        choices=['mcbert', 'xlmrb', 'xlmrl']
    )
    parser.add_argument(
        '-m', '--metric', type=str, default='macro', required=False,
        help='Metric to select for best model selection (used only for model name construction)',
        choices=['micro-1', 'micro', 'macro-1', 'macro']
    )

    parser.add_argument('-c', '--corpora', type=str, default='aussda_manual', required=False,
                        help='Corpora prefix or path prefix (used only for model name construction)')
    parser.add_argument('input_file', type=str, default=None,
                        help='File prefix or path prefix to use for inference')


def _get_training_args(arg, result_path: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=result_path,
        per_device_train_batch_size=arg.batch,
        per_device_eval_batch_size=arg.batch,
        disable_tqdm=not arg.tqdm
    )


def infer_binrel(arg) -> int:
    labels = get_labels(arg)
    if not labels:
        return 1

    l_str = get_labels_str(labels)
    text_fields = get_train_fields(arg)
    logger.info('Started inference for labels %s and text fields %s.', labels, text_fields)

    computed_name = arg.model_name

    input_file = os.path.join(arg.data_in_dir, arg.input_file)
    if not os.path.exists(input_file):
        input_file = arg.input_file
    if not os.path.exists(input_file):
        raise ValueError(f'Missing input file [{input_file}]')
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    data: pd.DataFrame = pd.read_csv(input_file, encoding='utf-8')

    for label in labels:
        # model name w/o label
        arg.model_name = compute_model_name(arg, text_fields, None, True)
        result_path = os.path.join(compute_model_path(arg, 'binrel'), label)
        logger.info('Started predicting with model [%s] for label [%s] from path [%s].',
                    arg.model_name, label, result_path)

        logger.info('Predicting label: [%s] with device [%s]', label, arg.device)
        mc = SeqClassifyModel(
            result_path,
            labeler=BinaryLabeler(labels=[label]),
            device=arg.device
        )
        results = run_inference(
            arg, mc, _get_training_args(arg, result_path), data, label, text_fields
        )
        data['p_' + label] = results

        # reset model name back to original
        if computed_name:
            arg.model_name = computed_name

    # write predictions
    arg.model_name = compute_model_name(arg, text_fields, None, True)  # model name w/o label
    result_path = os.path.join(compute_model_path(arg, 'binrel'))
    test_pred_path = os.path.join(
        os.path.dirname(result_path), base_name + '.binrel.' + arg.model_name + l_str + '.cvs'
    )
    data.to_csv(test_pred_path, encoding='utf-8', index=False)
    return 0
