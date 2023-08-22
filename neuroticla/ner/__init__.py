from neuroticla.core import ModuleDescriptor
from neuroticla.core.args import ModuleArguments, CommandArguments

NRCLA_MODULE = ModuleDescriptor(
    'ner',
    'Named Entity Recognition neural network module',
    ModuleArguments([
        CommandArguments('prep', 'Prepares the data'),
        CommandArguments('split', 'Splits the data'),
        CommandArguments('train', 'Trains the model'),
        CommandArguments('test', 'Unit test the model'),
        CommandArguments('infer', 'Infer the model'),
        CommandArguments('unittest', 'Internal tests')
    ])
)
