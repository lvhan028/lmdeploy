# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
import json
import uuid
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from mmengine import Registry
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin
from typing_extensions import TypeVar

from lmdeploy.utils import get_logger

_P = TypeVar('_P', bound=ProcessorMixin, default=ProcessorMixin)

logger = get_logger('lmdeploy')
MODELS = Registry('model', locations=['lmdeploy.model'])


def random_uuid() -> str:
    """Return a random uuid."""
    return str(uuid.uuid4().hex)


def get_text(content: Union[str, List[dict]]):
    """Within the OpenAI API, the content field may be specified as either a
    string or a list of ChatCompletionContentPartTextParam (defined in openai).

    When a list is provided, lmdeploy selects the first element to incorporate into the chat template, as the manner in
    which OpenAI processes lists is not explicitly defined.
    """

    if isinstance(content, str):
        return content
    return content[0]['text']


@dataclasses.dataclass
class ChatTemplateConfig:
    """Parameters for chat template.

    Args:
        model_name (str): the name of the deployed model. Determine which chat template will be applied.
            All the chat template names: `lmdeploy list`
        system (str | None): begin of the system prompt
        meta_instruction (str | None): system prompt
        eosys (str | None): end of the system prompt
        user (str | None): begin of the user prompt
        eoh (str | None): end of the user prompt
        assistant (str | None): begin of the assistant prompt
        eoa (str | None): end of the assistant prompt
        tool (str | None): begin of the tool prompt
        eotool (str | None): end of the tool prompt
    """  # noqa: E501

    model_name: str
    system: Optional[str] = None
    meta_instruction: Optional[str] = None
    eosys: Optional[str] = None
    user: Optional[str] = None
    eoh: Optional[str] = None
    assistant: Optional[str] = None
    eoa: Optional[str] = None
    tool: Optional[str] = None
    eotool: Optional[str] = None
    separator: Optional[str] = None
    stop_words: Optional[List[str]] = None

    @property
    def chat_template(self):
        attrs = {key: value for key, value in dataclasses.asdict(self).items() if value is not None}
        attrs.pop('model_name', None)
        if self.model_name in MODELS.module_dict.keys():
            model: BaseModel = MODELS.get(self.model_name)(**attrs)
        else:
            logger.warning(f'Could not find {self.model_name} in registered models. '
                           f'Register {self.model_name} using the BaseChatTemplate.')
            model = BaseChatTemplate(**attrs)
        return model

    def to_json(self, file_path=None):
        """Convert the dataclass instance to a JSON formatted string and
        optionally save to a file."""
        json_str = json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=4)
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, file_or_string):
        """Construct a dataclass instance from a JSON file or JSON string."""
        try:
            # Try to open the input_data as a file path
            with open(file_or_string, 'r', encoding='utf-8') as file:
                json_data = file.read()
        except FileNotFoundError:
            # If it's not a file path, assume it's a JSON string
            json_data = file_or_string
        except IOError:
            # If it's not a file path and not a valid JSON string, raise error
            raise ValueError('Invalid input. Must be a file path or a valid JSON string.')
        json_data = json.loads(json_data)
        if json_data.get('model_name', None) is None:
            json_data['model_name'] = random_uuid()
        if json_data['model_name'] not in MODELS.module_dict.keys():
            MODELS.register_module(json_data['model_name'], module=BaseChatTemplate)
        return cls(**json_data)


@MODELS.register_module(name='base')
class BaseModel:
    """Base model."""

    def __init__(self, stop_words=None, **kwargs):
        self.stop_words = stop_words

    @abstractmethod
    def messages2prompt(self, messages, **kwargs):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            messages (List): user's input prompt
        Returns:
            str: the concatenated prompt
        """
        pass


class BaseChatTemplate(BaseModel):
    """Base Chat template."""

    def __init__(self,
                 system='',
                 meta_instruction='',
                 eosys='',
                 user='',
                 eoh='',
                 assistant='',
                 eoa='',
                 separator='',
                 tool='',
                 eotool='',
                 **kwargs):
        super().__init__(**kwargs)
        self.system = system
        self.meta_instruction = meta_instruction
        self.user = user
        self.eoh = eoh
        self.eoa = eoa
        self.separator = separator
        self.eosys = eosys
        self.assistant = assistant
        self.tool = tool
        self.eotool = eotool

    def messages2prompt(self, messages, tools, **kwargs):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            messages (str | List): user's input prompt
        Returns:
            str: the concatenated prompt
        """
        if isinstance(messages, str):
            messages = [dict(role='user', content=messages)]
        box_map = dict(user=self.user, assistant=self.assistant, system=self.system, tool=self.tool)
        eox_map = dict(user=self.eoh, assistant=self.eoa + self.separator, system=self.eosys, tool=self.eotool)
        ret = ''
        if self.meta_instruction is not None:
            if len(messages) and messages[0]['role'] != 'system':
                ret += f'{self.system}{self.meta_instruction}{self.eosys}'
        for message in messages:
            role = message['role']
            content = get_text(message['content'])
            ret += f'{box_map[role]}{content}{eox_map[role]}'
        if len(messages) and messages[-1]['role'] == 'assistant':
            return ret[:-len(eox_map['assistant'])]  # prefix of response
        ret += f'{self.assistant}'
        return ret

    def apply_chat_template(self,
                            conversation: List[Dict[str, str]],
                            tools: Optional[List[Union[Dict, Callable]]] = None,
                            add_generation_prompt: bool = True,
                            continue_final_message: bool = False,
                            tokenize: bool = False,
                            **kwargs):
        """Apply the chat template to the messages.

        Args:
            messages (str | List): user's input prompt
        Returns:
            str: the concatenated prompt
        """
        return self.messages2prompt(conversation,
                                    tools=tools,
                                    add_generation_prompt=add_generation_prompt,
                                    continue_final_message=continue_final_message,
                                    tokenize=tokenize,
                                    **kwargs)


@MODELS.register_module(name='internlm')
class InternLMChat7B(BaseChatTemplate):
    """Chat template of InternLM model."""

    def __init__(
            self,
            system='<|System|>:',
            meta_instruction="""You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
""",  # noqa: E501
            eosys='\n',
            user='<|User|>:',
            eoh='\n',
            assistant='<|Bot|>:',
            eoa='<eoa>',
            separator='\n',
            stop_words=['<eoa>'],
            **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         stop_words=stop_words,
                         **kwargs)


@MODELS.register_module(name='internlm3')
@MODELS.register_module(name='internlm2')
class InternLM2Chat7B(InternLMChat7B):
    """Chat template and generation parameters of InternLM2-Chat-7B."""

    def __init__(self,
                 system='<|im_start|>system\n',
                 user='<|im_start|>user\n',
                 assistant='<|im_start|>assistant\n',
                 environment='<|im_start|>environment\n',
                 plugin='<|plugin|>',
                 interpreter='<|interpreter|>',
                 eosys='<|im_end|>\n',
                 eoh='<|im_end|>\n',
                 eoa='<|im_end|>',
                 eoenv='<|im_end|>\n',
                 separator='\n',
                 stop_words=['<|im_end|>', '<|action_end|>'],
                 **kwargs):
        self.plugin = plugin
        self.interpreter = interpreter
        self.environment = environment
        self.eoenv = eoenv
        super(InternLM2Chat7B, self).__init__(system=system,
                                              user=user,
                                              assistant=assistant,
                                              eosys=eosys,
                                              eoh=eoh,
                                              eoa=eoa,
                                              separator=separator,
                                              stop_words=stop_words,
                                              **kwargs)

    def messages2prompt(self, messages, tools=None, **kwargs):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            messages (str | List): user's input prompt
        Returns:
            str: the concatenated prompt
        """
        box_map = dict(user=self.user,
                       assistant=self.assistant,
                       system=self.system,
                       environment=self.environment,
                       tool=self.environment)
        eox_map = dict(user=self.eoh,
                       assistant=self.eoa + self.separator,
                       system=self.eosys,
                       environment=self.eoenv,
                       tool=self.eoenv)
        name_map = dict(plugin=self.plugin, interpreter=self.interpreter)
        ret = ''
        if self.meta_instruction is not None:
            if len(messages) and messages[0]['role'] != 'system':
                ret += f'{self.system}{self.meta_instruction}{self.eosys}'

        if tools:
            tools_prompt = dict(
                role='system',
                name='plugin',  # only support internlm2
                content=json.dumps(tools, ensure_ascii=False))
            insert_index = 0
            if messages[0]['role'] == 'system':
                insert_index = 1
            messages.insert(insert_index, tools_prompt)
        for message in messages:
            role = message['role']
            content = get_text(message['content'])
            if role == 'assistant' and message.get('tool_calls', None) is not None:
                for tool_call in message['tool_calls']:
                    function = tool_call.get('function', {})
                    function['name'] = function.get('name', '')
                    function['parameters'] = function.get('parameters', function.get('arguments', ''))
                    function.pop('arguments')
                    if isinstance(function['parameters'], str):
                        function['parameters'] = json.loads(function['parameters'])
                    content += f'<|action_start|><|plugin|>\n{json.dumps(function, ensure_ascii=False)}<|action_end|>'
            if 'name' in message and message['name'] in name_map:
                begin = box_map[role].strip() + f" name={name_map[message['name']]}\n"
            else:
                begin = box_map[role]
            ret += f'{begin}{content}{eox_map[role]}'
        if len(messages) and messages[-1]['role'] == 'assistant':
            return ret[:-len(eox_map['assistant'])]  # prefix of response
        ret += f'{self.assistant}'
        return ret


@MODELS.register_module(name='internvl-internlm2')
class InternVLInternLM2Chat(InternLM2Chat7B):

    def __init__(self, meta_instruction='You are an AI assistant whose name is InternLM (书生·浦语).', **kwargs):
        super().__init__(meta_instruction=meta_instruction, **kwargs)


@MODELS.register_module(name='internvl2-internlm2')
class InternVL2InternLM2(InternLM2Chat7B):

    def __init__(self,
                 meta_instruction='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
                 eosys='<|im_end|>',
                 eoh='<|im_end|>',
                 separator='',
                 stop_words=['<|im_start|>', '<|im_end|>'],
                 **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         eosys=eosys,
                         separator=separator,
                         eoh=eoh,
                         stop_words=stop_words,
                         **kwargs)


@MODELS.register_module(name='internvl2_5')
class InternVL2_5(InternLM2Chat7B):

    def __init__(
            self,
            meta_instruction='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',  # noqa
            **kwargs):
        super().__init__(meta_instruction=meta_instruction, **kwargs)


@MODELS.register_module(name=['internlm-xcomposer2', 'internlm-xcomposer2d5'])
class InternLMXComposer2Chat7B(InternLMChat7B):
    """Chat template and generation parameters of InternLM-XComposer2-7b."""

    def __init__(
            self,
            system='[UNUSED_TOKEN_146]system\n',
            meta_instruction="""You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).
- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.
- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.""",  # noqa: E501
            user='[UNUSED_TOKEN_146]user\n',
            assistant='[UNUSED_TOKEN_146]assistant\n',
            eosys='[UNUSED_TOKEN_145]\n',
            eoh='[UNUSED_TOKEN_145]\n',
            eoa='[UNUSED_TOKEN_145]\n',
            separator='\n',
            stop_words=['[UNUSED_TOKEN_145]'],
            **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         user=user,
                         assistant=assistant,
                         eosys=eosys,
                         eoh=eoh,
                         eoa=eoa,
                         separator=separator,
                         stop_words=stop_words,
                         **kwargs)


@MODELS.register_module(name=['internvl-zh'])
class InternVLZH(BaseChatTemplate):

    def __init__(self, user='<human>: ', eoh=' ', assistant='<bot>: ', eoa='</s>', **kwargs):
        super().__init__(user=user, eoh=eoh, assistant=assistant, eoa=eoa, **kwargs)

    def messages2prompt(self, messages, sequence_start=True, **kwargs):
        return super().messages2prompt(messages, sequence_start, **kwargs)[:-1]


@MODELS.register_module(name=['internvl-zh-hermes2'])
class ChatmlDirect(BaseChatTemplate):

    def __init__(self,
                 system='<|im_start|>system\n',
                 meta_instruction='Answer the questions.',
                 eosys='<|im_end|>',
                 user='<|im_start|>user\n',
                 eoh='<|im_end|>',
                 assistant='<|im_start|>assistant\n',
                 eoa='<|im_end|>',
                 separator='',
                 **kwargs):
        super().__init__(system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'llava' in path and 'v1.6-34b' in path:
            return 'llava-chatml'
        if 'internvl-chat' in path and 'v1-2' in path:
            return 'internvl-zh-hermes2'


@MODELS.register_module(name='phi-3')
class Phi3Instruct(BaseChatTemplate):
    """Chat template of InternLM model."""

    def __init__(self,
                 system='<|system|>\n',
                 meta_instruction=None,
                 eosys='<|end|>\n',
                 user='<|user|>\n',
                 eoh='<|end|>\n',
                 assistant='<|assistant|>\n',
                 eoa='<|end|>\n',
                 separator='',
                 stop_words=['<|end|>', '<|endoftext|>', '<|assistant|>'],
                 **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         stop_words=stop_words,
                         **kwargs)


@MODELS.register_module(name='internvl2-phi3')
class InternVL2Phi3(Phi3Instruct):

    def __init__(self, meta_instruction='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。', **kwargs):
        super().__init__(meta_instruction=meta_instruction, **kwargs)


@MODELS.register_module(name='internvl-phi3')
class InternVLPhi3(Phi3Instruct):
    """Chat template of InternVL Chat 4B model."""

    def __init__(self,
                 meta_instruction='You are an AI assistant whose name is Phi-3.',
                 eosys='<|end|>',
                 eoh='<|end|>',
                 eoa='<|end|>',
                 separator='',
                 **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         eosys=eosys,
                         eoh=eoh,
                         eoa=eoa,
                         separator=separator,
                         **kwargs)


class HfChatTemplate:

    def __init__(self, tokenizer, jinja_chat_template: str = None):
        self.tokenizer = tokenizer
        self.chat_template = jinja_chat_template

    def apply_chat_template(self,
                            messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
                            tools: Optional[List[Union[Dict, Callable]]] = None,
                            add_generation_prompt: bool = True,
                            **kwargs):
        """"""
        return self.tokenizer.apply_chat_template(
            conversation=messages,
            tools=tools,
            chat_template=self.chat_template,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )


def get_processor(
    processor_name: str,
    *args: Any,
    trust_remote_code: bool = True,
    processor_cls: Union[type[_P], tuple[type[_P], ...]] = ProcessorMixin,
    **kwargs: Any,
) -> _P:
    """Load a processor for the given model name via HuggingFace."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoProcessor

    processor_factory = (AutoProcessor
                         if processor_cls == ProcessorMixin or isinstance(processor_cls, tuple) else processor_cls)

    try:
        processor = processor_factory.from_pretrained(
            processor_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = ('Failed to load the processor. If the processor is '
                       'a custom processor not yet available in the HuggingFace '
                       'transformers library, consider setting '
                       '`trust_remote_code=True` in LLM or using the '
                       '`--trust-remote-code` flag in the CLI.')
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(processor, processor_cls):
        raise TypeError('Invalid type of HuggingFace processor. '
                        f'Expected type: {processor_cls}, but '
                        f'found type: {type(processor)}')

    return processor


def resolve_hf_chat_template(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                             chat_template_config: ChatTemplateConfig = None,
                             tools: Optional[list[dict[str, Any]]] = None) -> BaseChatTemplate:
    """modify from vllm."""

    # 1st priority: The given chat template
    if chat_template_config is not None and chat_template_config.model_name:
        return chat_template_config.chat_template

    # 2nd priority: AutoProcessor chat template, unless tool calling is enabled
    if tools is None:
        try:
            processor = get_processor(
                tokenizer.name_or_path,
                processor_cls=(PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin),
                trust_remote_code=True,
            )
            if isinstance(processor, ProcessorMixin) and \
                processor.chat_template is not None:
                return HfChatTemplate(tokenizer, processor.chat_template)
        except Exception:
            logger.debug('Failed to load AutoProcessor chat template for %s', tokenizer.name_or_path, exc_info=True)

    # 3rd priority: AutoTokenizer chat template
    try:
        chat_template = tokenizer.get_chat_template(tools=tools)
        return HfChatTemplate(tokenizer, chat_template)
    except Exception:
        logger.debug('Failed to load AutoTokenizer chat template for %s', tokenizer.name_or_path, exc_info=True)

    return None
