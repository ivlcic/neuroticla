import os
import logging
import argparse
import importlib
import textwrap

from typing import List, Tuple
from argparse import ArgumentParser

logger = logging.getLogger('neuroticla.args')


class CommandArguments:

    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description

    def get_name(self):
        return self._name

    def get_description(self):
        return self._description


class ModuleArguments:
    def __init__(self, commands: List[CommandArguments]):
        self._commands = commands
        self._actions = None
        self._parser = None

    def init_parser(self, package: str, description: str):
        self._parser = argparse.ArgumentParser(
                description=description
            )
        self._actions = self._parser.add_subparsers(
            title='Select the module action', help='Help', dest='action', metavar='action', required=True
        )
        for x in self._commands:
            pym_name = package + '.' + x.get_name()
            logger.debug('Loading Python module: [%s]', pym_name)
            py_module = importlib.import_module(pym_name)
            logger.debug('Loaded Python module: [%s]', pym_name)
            subparser = self._actions.add_parser(
                x.get_name(),
                help=x.get_description(),
                formatter_class=argparse.RawTextHelpFormatter
            )
            py_module.args(package, subparser)
            subparser.set_defaults(func=py_module.main)
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
            For example "80:10" would produce 80%% train, 10%% evaluation and 10%% test data set size.
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
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(path, path_type, package)

    @classmethod
    def raw_data_dir(cls, package: str, parser: ArgumentParser, name_or_flags: Tuple[str, ...]) -> None:
        path = os.path.join(CommonArguments._package_path('data', package), 'raw')
        parser.add_argument(
            *name_or_flags,
            help='Corpora raw directory (default: %(default)s)',
            type=CommonArguments._is_or_make_dir_path,
            default=path
        )

    @classmethod
    def processed_data_dir(cls, package: str, parser: ArgumentParser, name_or_flags: Tuple[str, ...]) -> None:
        path = os.path.join(CommonArguments._package_path('data', package), 'processed')
        parser.add_argument(
            *name_or_flags,
            help='Processed data directory (default: %(default)s)',
            type=CommonArguments._is_or_make_dir_path,
            default=path
        )

    @classmethod
    def tmp_dir(cls, package: str, parser: ArgumentParser, name_or_flags: Tuple[str, ...]) -> None:
        path = os.path.join(CommonArguments._package_path('tmp', package), 'processed')
        parser.add_argument(
            *name_or_flags,
            help='Process working tmp directory (default: %(default)s)',
            type=CommonArguments._is_or_make_dir_path,
            default=path
        )
