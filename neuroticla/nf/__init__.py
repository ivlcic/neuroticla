from neuroticla.core import ModuleDescriptor
from neuroticla.core.args import ModuleArguments, CommandArguments

NRCLA_MODULE = ModuleDescriptor(
    'nf',
    'Newsframes detection neural network module',
    ModuleArguments([
        CommandArguments('prep', 'Prepares the data'),
        CommandArguments('split', 'Splits the data'),
        CommandArguments('train', 'Train for the news framing detection', multi_action=True),
        CommandArguments('unittest', 'Internal tests'),
        CommandArguments('slomcor', 'Retrieve Slovenian Migration Corpus')
    ])
)
