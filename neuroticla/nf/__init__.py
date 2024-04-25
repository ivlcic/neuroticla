from ..core import ModuleDescriptor
from ..core.args import ModuleArguments, CommandArguments

MODULE_DESCRIPTOR = ModuleDescriptor(
    'nf',
    'News-framing detection neural network module',
    ModuleArguments([
        CommandArguments('prep', 'Prepares the data'),
        CommandArguments('split', 'Splits the data'),
        CommandArguments('embed', 'Embeds the data'),
        CommandArguments('train', 'Train for the news-framing detection', multi_action=True),
        CommandArguments('train_embed', 'Train for the news-framing detection on text embeddings', multi_action=True),
        CommandArguments('infer', 'Run inference for news-framing detection models', multi_action=True),
        CommandArguments('analyze', 'Analyze corpus', multi_action=True),
        CommandArguments('unittest', 'Internal tests'),
        CommandArguments('slomcor', 'Retrieve Slovenian Migration Corpus'),
        CommandArguments('tmpdl', 'Retrieve')
    ])
)
