from __future__ import annotations

import importlib
import logging.config
import os
from typing import TypeVar

from .args import ModuleArguments, ArgumentParser


class DefaultLogFilter(logging.Filter):
    def filter(self, record):
        record.levelname = '[%s]' % record.levelname
        record.funcName = '[%s]' % record.funcName
        record.lineno = '[%s]' % record.lineno
        return True


LOGGING = {
    'version': 1,
    'formatters': {
        'my_formatter': {
          'format': '%(asctime)s %(levelname)-7s %(name)s %(lineno)-3s: %(message)s',
          'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'filters': {
        'myfilter': {
            '()': DefaultLogFilter
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'filters': ['myfilter'],
            'formatter': 'my_formatter',
        }
    },
    'root': {
        'level': 'INFO',
        'filters': ['myfilter'],
        'handlers': ['console']
    },
}

logging.config.dictConfig(LOGGING)

logger = logging.getLogger('core')


class ModuleDescriptor:

    project = os.path.basename(os.path.dirname(os.path.dirname(__file__)))

    def __init__(self, name: str, descr: str, arg: ModuleArguments):
        self._name = name
        self._description = descr
        self._args = arg
        self._args.init_parser(
            ModuleDescriptor.project, name, descr
        )

    def get_name(self) -> str:
        return self._name

    def get_description(self) -> str:
        return self._description

    def get_args(self) -> ModuleArguments:
        return self._args


TExecModule = TypeVar('TExecModule', bound='ExecModule')


class ExecModule:

    @classmethod
    def get(cls, file_as_module_name: str) -> TExecModule:
        package = os.path.splitext(os.path.basename(file_as_module_name))[0]
        logger.debug('Loading package: [%s]', package)
        py_module = importlib.import_module(ModuleDescriptor.project + '.' + package)
        logger.debug('Imported package: [%s]', package)
        m: ExecModule = ExecModule(py_module)
        logger.info('Loaded module: [%s]', package)
        return m

    def __init__(self, py_module):
        self._py_module = py_module
        self._descriptor: ModuleDescriptor = py_module.MODULE_DESCRIPTOR

    def execute(self) -> int:
        module_args: ModuleArguments = self._descriptor.get_args()
        parser: ArgumentParser = module_args.get_parser()
        arg = parser.parse_args()
        pym_name = ModuleDescriptor.project + '.' + self._descriptor.get_name() + '.' + arg.action
        logger.debug('Loading Python module: [%s]', pym_name)
        py_module = importlib.import_module(pym_name)

        arg.func = None
        arg.module_name = self._descriptor.get_name()
        if hasattr(arg, 'sub_action') and arg.sub_action is not None:
            fn = getattr(py_module, arg.action + '_' + arg.sub_action, None)
            if fn is not None:
                arg.func = fn
        if arg.func is None:
            arg.func = py_module.main
        return arg.func(arg)
