import argparse
import importlib
import logging
import os
import textwrap
import torch

from argparse import ArgumentParser
from typing import List, Tuple

logger = logging.getLogger('core.args')


class CommandArguments:

    def __init__(self, name: str, description: str, multi_action: bool = False):
        self._name = name
        self._description = description
        self._multi_action = multi_action
        if self._name == 'unittest':
            self._multi_action = True

    def get_name(self):
        return self._name

    def get_description(self):
        return self._description

    def is_multi_action(self):
        return self._multi_action


class ModuleArguments:
    def __init__(self, commands: List[CommandArguments]):
        self._commands = commands
        self._actions = None
        self._parser = None

    def init_parser(self, project: str, module_name: str, description: str):
        self._parser = argparse.ArgumentParser(
                description=description
            )
        self._actions = self._parser.add_subparsers(
            title='Select the module action', help='Help', dest='action', metavar='action', required=True
        )
        for x in self._commands:
            cmd_name = x.get_name()
            multi_action = x.is_multi_action()
            pym_name = project + '.' + module_name + '.' + cmd_name
            logger.debug('Loading Python module: [%s]', pym_name)
            py_module = importlib.import_module(pym_name)
            logger.debug('Loaded Python module: [%s]', pym_name)

            subparser = self._actions.add_parser(
                cmd_name,
                help=x.get_description(),
                formatter_class=argparse.RawTextHelpFormatter
            )

            if multi_action:
                sub_actions = []
                for n in dir(py_module):
                    if n.startswith(cmd_name + '_'):
                        sub_actions.append(n[len(cmd_name) + 1:])
                subparser.add_argument(
                    'sub_action',
                    help=cmd_name.capitalize() + ' functions to invoke', choices=sub_actions
                )
            py_module.add_args(module_name, subparser)
            logger.debug('Setting up arguments for [%s]', pym_name)

    def get_parser(self) -> ArgumentParser:
        return self._parser


class CommonArguments:

    @classmethod
    def data_split(cls, parser: ArgumentParser):
        parser.add_argument(
            '-s', '--data_split',
            help=textwrap.dedent('''\
            Data split in %% separated with a colon (default: %(default)s):
            For example '80:10' would produce 80%% train, 10%% evaluation and 10%% test data set size.
            '''),
            type=str,
            default='80:10'
        )

    @classmethod
    def _is_or_make_dir_path(cls, dir_name: str) -> str:
        if os.path.isdir(dir_name):
            return dir_name
        else:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                return dir_name
            raise NotADirectoryError(dir_name)

    @classmethod
    def _package_path(cls, path_type: str, package: str) -> str:
        path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(path, path_type, package)

    @classmethod
    def data_path(cls, package: str, sub_path: str) -> str:
        path = os.path.join(CommonArguments._package_path('data', package), sub_path)
        return path

    @classmethod
    def raw_data_dir(cls, package: str, parser: ArgumentParser, name_or_flags: Tuple[str, ...]) -> None:
        path = cls.data_path(package, 'raw')
        # os.path.join(CommonArguments._package_path('data', package), 'raw')
        parser.add_argument(
            *name_or_flags,
            help='Corpora raw directory (default: %(default)s)',
            type=CommonArguments._is_or_make_dir_path,
            default=path
        )

    @classmethod
    def processed_data_dir(cls, package: str, parser: ArgumentParser, name_or_flags: Tuple[str, ...]) -> None:
        path = cls.data_path(package, 'processed')
        # os.path.join(CommonArguments._package_path('data', package), 'processed')
        parser.add_argument(
            *name_or_flags,
            help='Processed data directory (default: %(default)s)',
            type=CommonArguments._is_or_make_dir_path,
            default=path
        )

    @classmethod
    def split_data_dir(cls, package: str, parser: ArgumentParser, name_or_flags: Tuple[str, ...]) -> None:
        path = cls.data_path(package, 'split')
        # os.path.join(CommonArguments._package_path('data', package), 'split')
        parser.add_argument(
            *name_or_flags,
            help='Split data directory (default: %(default)s)',
            type=CommonArguments._is_or_make_dir_path,
            default=path
        )

    @classmethod
    def tmp_dir(cls, package: str, parser: ArgumentParser, name_or_flags: Tuple[str, ...]) -> None:
        path = os.path.join(CommonArguments._package_path('tmp', package))
        parser.add_argument(
            *name_or_flags,
            help='Process working tmp directory (default: %(default)s)',
            type=CommonArguments._is_or_make_dir_path,
            default=path
        )

    @classmethod
    def result_dir(cls, package: str, parser: ArgumentParser, name_or_flags: Tuple[str, ...]) -> None:
        path = os.path.join(CommonArguments._package_path('result', package))
        parser.add_argument(
            *name_or_flags,
            help='Process result directory (default: %(default)s)',
            type=CommonArguments._is_or_make_dir_path,
            default=path
        )

    @classmethod
    def split(cls, package: str, parser: ArgumentParser, default_password: str = None):
        CommonArguments.processed_data_dir(package, parser, ('-i', '--data_in_dir'))
        CommonArguments.split_data_dir(package, parser, ('-o', '--data_out_dir'))
        CommonArguments.tmp_dir(package, parser, ('-t', '--tmp_dir'))
        CommonArguments.data_split(parser)
        parser.add_argument(
            '-u', '--subsets', type=str, default=None,
            help='Subsets of the files to use for each corpora (file name contains any of the comma separated strings)',
        )
        if default_password:
            parser.add_argument(
                '-p', '--password', type=str, default=default_password, help='Zip file password'
            )
        else:
            parser.add_argument(
                '-p', '--password', type=str, required=True, help='Zip file password'
            )
        parser.add_argument(
            '-r', '--non_reproducible_shuffle', action='store_true', default=False,
            help='Non reproducible data shuffle.',
        )

    @classmethod
    def device(cls, parser: ArgumentParser):
        have_cuda = torch.cuda.is_available()
        device = 'cuda' if have_cuda else 'cpu'
        parser.add_argument(
            '-d', '--device', type=str, help='Torch device to use.', default=device
        )

    @classmethod
    def batch_size(cls, parser: ArgumentParser, default: int = 32):
        parser.add_argument(
            '-b', '--batch', help='Batch size.', type=int, default=default
        )

    @classmethod
    def max_seq_len(cls, parser: ArgumentParser, default: int = 512):
        parser.add_argument(
            '--max_seq_len', help='Max sentence length in sub-word tokens.', type=int, default=default
        )

    @classmethod
    def train(cls, parser: ArgumentParser, batch_default: int = 32, max_seq_len_default: int = 512):
        cls.device(parser)
        cls.batch_size(parser, batch_default)
        cls.max_seq_len(parser, max_seq_len_default)
        parser.add_argument(
            '-l', '--learn_rate', help='Learning rate', type=float, default=2e-5
        )
        parser.add_argument(
            '-e', '--epochs', help='Number of epochs.', type=int, default=20
        )

    @classmethod
    def test(cls, parser: ArgumentParser, batch_default: int = 32):
        cls.device(parser)
        cls.batch_size(parser, batch_default)
