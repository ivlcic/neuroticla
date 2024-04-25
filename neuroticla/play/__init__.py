from ..core import ModuleDescriptor
from ..core.args import ModuleArguments, CommandArguments

MODULE_DESCRIPTOR = ModuleDescriptor(
    'play',
    'Playground module',
    ModuleArguments([
        CommandArguments('cluster', 'Cluster compare', multi_action=True),
        CommandArguments('corpus', 'Corpus management', multi_action=True),
        CommandArguments('sentiment', 'Sentiment corpus', multi_action=True)
    ])
)
