from __future__ import annotations

import importlib
import logging
import os
from typing import Type

from .args import ModuleArguments, ArgumentParser


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
logger = logging.getLogger('neuroticla.core')


class ModuleDescriptor:

    project = 'neuroticla'

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


class ExecModule:

    @classmethod
    def get(cls, file_as_module_name: str) -> Type[ExecModule]:
        package = os.path.splitext(os.path.basename(file_as_module_name))[0]
        logger.debug('Loading package: [%s]', package)
        py_module = importlib.import_module(ModuleDescriptor.project + '.' + package)
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
        arg = parser.parse_args()
        pym_name = ModuleDescriptor.project + '.' + self._descr.get_name() + '.' + arg.action
        logger.debug('Loading Python module: [%s]', pym_name)
        py_module = importlib.import_module(pym_name)

        arg.func = py_module.main
        if hasattr(args, 'sub_action') and arg.sub_action is not None:
            fn = getattr(py_module, arg.action + '_' + arg.sub_action, None)
            if fn is not None:
                arg.func = fn

        return arg.func(arg)
