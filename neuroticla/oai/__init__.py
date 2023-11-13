from ..core import ModuleDescriptor
from ..core.args import ModuleArguments, CommandArguments

MODULE_DESCRIPTOR = ModuleDescriptor(
    'oai',
    'OpenAI playground',
    ModuleArguments([
        CommandArguments('test_prompt', 'Tests OpenAI Prompts', multi_action=True)
    ])
)
