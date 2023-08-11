from __future__ import annotations

import os
import argparse
import importlib
import logging

from .args import ModuleArguments


def fmt_filter(record):
    record.levelname = '[%s]' % record.levelname
    record.funcName = '[%s]' % record.funcName
    record.lineno = '[%s]' % record.lineno
    return True


logging.basicConfig(
    format='%(asctime)s %(levelname)-7s %(name)s %(lineno)-3s: %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().addFilter(fmt_filter)
logger = logging.getLogger('neuroticla')


class ModuleDescriptor:

    def __init__(self, name: str, descr: str, args: ModuleArguments):
        self._name = name
        self._description = descr
        self._args = args
        self._args.init_parser(
            name, descr
        )

    def get_name(self) -> str:
        return self._name

    def get_description(self) -> str:
        return self._description

    def get_args(self) -> ModuleArguments:
        return self._args


class ExecModule:

    @classmethod
    def get(cls, file_as_module_name: str) -> Self:
        package = os.path.splitext(os.path.basename(file_as_module_name))[0]
        logger.debug('Loading package: [%s]', package)
        py_module = importlib.import_module(package)
        logger.debug('Imported package: [%s]', package)
        m: ExecModule = ExecModule(py_module)
        logger.info('Loaded module: [%s]', package)
        return m

    def __init__(self, py_module):
        self._py_module = py_module
        self._descr: ModuleDescriptor = py_module.NRCLA_MODULE

    def execute(self) -> int:
        module_args: ModuleArguments = self._descr.get_args()
        parser: ArgumentParser = module_args.get_parser()
        args = parser.parse_args()
        pym_name = self._descr.get_name() + '.' + args.action
        logger.debug('Loading Python module: [%s]', pym_name)
        py_module = importlib.import_module(pym_name)

        if args.action == 'test':
            fn = getattr(py_module, 'test_' + args.test, None)
            args.func = fn
        else:
            args.func = py_module.main

        args.func(args)
