from argparse import ArgumentParser
from neuroticla.core import ModuleDescriptor
from neuroticla.core.args import ModuleArguments, CommandArguments, CommonArguments

NRCLA_MODULE = ModuleDescriptor(
    'ner',
    'Named Entity Recognition neural network module',
    ModuleArguments({
        CommandArguments('prep', 'Prepares the data'),
        CommandArguments('split', 'Splits the data'),
        CommandArguments('test', 'Internal tests')
    })
)
