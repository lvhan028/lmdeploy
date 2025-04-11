# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
import json
import uuid
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Union

from mmengine import Registry

from lmdeploy.utils import get_logger

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


@MODELS.register_module(name='llama')
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


@MODELS.register_module(name='cogvlm')
class CogVLM(BaseChatTemplate):
    """Chat template of CogVLM model."""

    def __init__(self,
                 meta_instruction='',
                 eosys='',
                 user='Question: ',
                 separator='\n',
                 eoh=' ',
                 assistant='Answer:',
                 eoa='</s>',
                 stop_words=['</s>'],
                 **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         separator=separator,
                         assistant=assistant,
                         eoa=eoa,
                         stop_words=stop_words,
                         **kwargs)


@MODELS.register_module(name='cogvlm2')
class CogVLM2(CogVLM):
    """Chat template of CogVLM2 model."""

    def __init__(self, eoa='<|end_of_text|>', stop_words=['<|end_of_text|>'], **kwargs):
        super().__init__(eoa=eoa, stop_words=stop_words, **kwargs)


@MODELS.register_module(name='wizardlm')
@MODELS.register_module(name='vicuna')
class Vicuna(BaseChatTemplate):
    """Chat template of vicuna model."""

    def __init__(
            self,
            meta_instruction="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.""",  # noqa: E501
            eosys=' ',
            user='USER: ',
            eoh=' ',
            assistant='ASSISTANT: ',
            eoa='</s>',
            stop_words=['</s>'],
            **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         stop_words=stop_words,
                         **kwargs)

    def messages2prompt(self, messages, **kwargs):
        return super().messages2prompt(messages, **kwargs)[:-1]


@MODELS.register_module(name='llava-v1')
class Llavav1(Vicuna):
    """Chat template of llava-v1 model."""

    def __init__(
            self,
            meta_instruction="""A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.""",  # noqa: E501
            **kwargs):
        super().__init__(meta_instruction=meta_instruction, **kwargs)


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


@MODELS.register_module(name='baichuan2')
class Baichuan2(BaseChatTemplate):
    """Chat template and generation parameters of Baichuan2-7B-Base and
    Baichuan2-7B-Chat models."""

    def __init__(self, user='<reserved_106>', assistant='<reserved_107>', **kwargs):
        super().__init__(user=user, assistant=assistant, **kwargs)


@MODELS.register_module(name='llama2')
class Llama2(BaseChatTemplate):
    """Chat template of LLaMA2 model."""

    def __init__(
            self,
            system='[INST] <<SYS>>\n',
            meta_instruction="""\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",  # noqa: E501
            eosys='\n<</SYS>>\n\n',
            assistant=' [/INST] ',
            eoa='</s>',
            separator='<s>[INST] ',
            session_len=4096,
            **kwargs):
        super().__init__(system=system,
                         meta_instruction=meta_instruction,
                         eosys=eosys,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         session_len=session_len,
                         **kwargs)


@MODELS.register_module(name='minicpmv-2d6')
@MODELS.register_module(name='minicpm3')
@MODELS.register_module(name='qwen')
class Qwen7BChat(BaseChatTemplate):
    """Chat template for Qwen-7B-Chat."""

    def __init__(self,
                 system='<|im_start|>system\n',
                 meta_instruction='You are a helpful assistant.',
                 eosys='<|im_end|>\n',
                 user='<|im_start|>user\n',
                 eoh='<|im_end|>\n',
                 assistant='<|im_start|>assistant\n',
                 eoa='<|im_end|>',
                 separator='\n',
                 stop_words=['<|im_end|>'],
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


@MODELS.register_module(name='chatglm')
class ChatGLM2(BaseModel):

    def __init__(self, user='问：', eoh='\n\n', assistant='答：', eoa='\n\n', **kwargs):
        super().__init__(**kwargs)
        self._user = user
        self._assistant = assistant
        self._eoh = eoh
        self._eoa = eoa
        self.count = 0

    def messages2prompt(self, messages, **kwargs):
        """message to prompt."""
        ret = ''
        count = 0
        for message in messages:
            role = message['role']
            content = get_text(message['content'])
            if role == 'user':
                count += 1
                ret += f'[Round {count}]\n\n'
                ret += f'{self._user}{content}{self._eoh}'
                ret += f'{self._assistant}'
            if role == 'assistant':
                ret += f'{content}'
        return ret


@MODELS.register_module(name=['mistral', 'mixtral'])
class MistralChat(BaseChatTemplate):
    """Template of Mistral and Mixtral Instruct models.

    `https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1`
    `https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1`
    """

    def __init__(self, user='[INST] ', eoh=' [/INST]', eoa='</s>', **kwargs):
        super().__init__(user=user, eoh=eoh, eoa=eoa, **kwargs)


@MODELS.register_module(name=['gemma'])
class Gemma(BaseChatTemplate):
    """Template of Gemma models.

    `https://huggingface.co/google/gemma-7b-it`
    """

    def __init__(self,
                 user='<start_of_turn>user\n',
                 eoh='<end_of_turn>\n',
                 assistant='<start_of_turn>model\n',
                 eoa='<end_of_turn>\n',
                 stop_words=['<end_of_turn>'],
                 **kwargs):
        super().__init__(user=user, eoh=eoh, assistant=assistant, eoa=eoa, stop_words=stop_words, **kwargs)


@MODELS.register_module(name=['deepseek'])
class Deepseek(BaseChatTemplate):

    def __init__(self,
                 eosys='\n\n',
                 user='User: ',
                 eoh='\n\n',
                 assistant='Assistant: ',
                 eoa='<｜end▁of▁sentence｜>',
                 **kwargs):
        super().__init__(eosys=eosys, user=user, eoh=eoh, assistant=assistant, eoa=eoa, **kwargs)

    def messages2prompt(self, messages, sequence_start=True, **kwargs):
        return super().messages2prompt(messages, sequence_start, **kwargs)[:-1]


@MODELS.register_module(name=['internvl-zh'])
class InternVLZH(BaseChatTemplate):

    def __init__(self, user='<human>: ', eoh=' ', assistant='<bot>: ', eoa='</s>', **kwargs):
        super().__init__(user=user, eoh=eoh, assistant=assistant, eoa=eoa, **kwargs)

    def messages2prompt(self, messages, sequence_start=True, **kwargs):
        return super().messages2prompt(messages, sequence_start, **kwargs)[:-1]


@MODELS.register_module(name=['deepseek-vl'])
class DeepseekVL(BaseChatTemplate):

    def __init__(
            self,
            meta_instruction="""You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.""",  # noqa: E501
            eosys='\n\n',
            user='User: ',
            eoh='\n\n',
            assistant='Assistant: ',
            eoa='<｜end▁of▁sentence｜>',
            **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         **kwargs)

    def messages2prompt(self, messages, sequence_start=True, **kwargs):
        return super().messages2prompt(messages, sequence_start, **kwargs)[:-1]

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'deepseek-vl' in path and 'chat' in path:
            return 'deepseek-vl'


@MODELS.register_module(name=['deepseek-vl2'])
class DeepseekVL2(BaseChatTemplate):

    def __init__(self,
                 meta_instruction='',
                 eosys='',
                 user='<|User|>: ',
                 eoh='\n\n',
                 assistant='<|Assistant|>: ',
                 eoa='<｜end▁of▁sentence｜>',
                 **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         eosys=eosys,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         **kwargs)

    def messages2prompt(self, messages, sequence_start=True, **kwargs):
        return super().messages2prompt(messages, sequence_start, **kwargs)[:-1]


@MODELS.register_module(name=['yi-vl'])
class YiVL(BaseChatTemplate):

    def __init__(
            self,
            meta_instruction="""This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. Read all the images carefully, and respond to the human's questions with informative, helpful, detailed and polite answers. 这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。\n\n""",  # noqa: E501
            user='### Human: ',
            eoh='\n',
            assistant='### Assistant:',
            eoa='\n',
            stop_words=['###'],
            **kwargs):
        super().__init__(meta_instruction=meta_instruction,
                         user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         stop_words=stop_words,
                         **kwargs)


@MODELS.register_module(name=['llava-chatml', 'internvl-zh-hermes2'])
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


@MODELS.register_module(name='chatglm3')
class ChatGLM3(BaseChatTemplate):
    """Chat template of chatglm3 model."""

    def __init__(self,
                 system='<|system|>\n ',
                 meta_instruction=None,
                 eosys='',
                 user='<|user|>\n ',
                 eoh='',
                 assistant='<|assistant|>\n ',
                 eoa='',
                 separator='',
                 stop_words=['<eos>'],
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
        self.start = '[gMASK]sop'

    def messages2prompt(self, messages, **kwargs):
        """Return the prompt that is concatenated with other elements in the
        chat template.

        Args:
            messages (str | List): user's input prompt
        Returns:
            str: the concatenated prompt
        """
        return self.start + super().messages2prompt(messages, **kwargs)

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'chatglm3' in path:
            return 'chatglm3'


@MODELS.register_module(name='glm4')
class Glm4Chat(ChatGLM3):
    """Chat template of glm-4 model."""

    def __init__(self,
                 system='<|system|>\n',
                 user='<|user|>\n',
                 assistant='<|assistant|>\n',
                 stop_words=['<|user|>', '<|endoftext|>', '<|observation|>'],
                 **kwargs):
        super().__init__(system=system, user=user, assistant=assistant, stop_words=stop_words, **kwargs)
        self.start = '[gMASK]<sop>'

    @classmethod
    def match(cls, model_path: str) -> Optional[str]:
        """Return the model_name that was registered to MODELS.

        Args:
            model_path (str): the model path used for matching.
        """
        path = model_path.lower()
        if 'glm-4' in path:
            return 'glm4'


@MODELS.register_module(name='codegeex4')
class CodeGeeX4Chat(BaseChatTemplate):
    """Chat template of THUDM/codegeex4-all-9b model."""

    def __init__(self,
                 system='<|system|>\n',
                 meta_instruction='你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。',
                 eosys='',
                 user='<|user|>\n',
                 eoh='',
                 assistant='<|assistant|>\n',
                 eoa='',
                 separator='',
                 stop_words=['<|endoftext|>', '<|user|>', '<|observation|>'],
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


@MODELS.register_module(name='molmo')
class Molmo(BaseChatTemplate):

    def __init__(self,
                 user=' User: ',
                 eoh='',
                 assistant=' Assistant:',
                 eoa='',
                 separator=' ',
                 stop_words=['<|endoftext|>'],
                 **kwargs):
        super().__init__(user=user,
                         eoh=eoh,
                         assistant=assistant,
                         eoa=eoa,
                         separator=separator,
                         stop_words=stop_words,
                         **kwargs)


class HfChatTemplate:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self,
                            messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
                            tools: Optional[List[Union[Dict, Callable]]] = None,
                            jinja_chat_template: str = None,
                            add_generation_prompt: bool = True,
                            **kwargs):
        """"""
        return self.tokenizer.apply_chat_template(
            conversation=messages,
            tools=tools,
            chat_template=jinja_chat_template,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
