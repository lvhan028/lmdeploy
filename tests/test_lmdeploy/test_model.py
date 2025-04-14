import pytest
from transformers import AutoTokenizer

from lmdeploy.model import MODELS, ChatTemplateConfig, get_processor, resolve_hf_chat_template


def test_base_model():
    model = MODELS.get('llama')()
    assert model is not None
    assert model.stop_words is None


def test_prefix_response():
    model = MODELS.get('internlm2')()
    messages = [dict(role='assistant', content='prefix test')]
    prompt = model.messages2prompt(messages)
    assert prompt[-len('prefix test'):] == 'prefix test'


def test_internlm_chat():
    model = MODELS.get('internlm')
    assert model.stop_words is not None
    assert model.system == '<|System|>:'

    model = MODELS.get('internlm')(system='Provide answers in Python')
    assert model.system == 'Provide answers in Python'


def test_internlm_tool_call():
    messages = []
    messages.append({
        'role':
        'system',
        'name':
        'plugin',
        'content':
        '[{"description": "Compute the sum of two numbers", "name": "add", "parameters": {"type": "object", "properties": {"a": {"type": "int", "description": "A number"}, "b": {"type": "int", "description": "A number"}}, "required": ["a", "b"]}}, {"description": "Calculate the product of two numbers", "name": "mul", "parameters": {"type": "object", "properties": {"a": {"type": "int", "description": "A number"}, "b": {"type": "int", "description": "A number"}}, "required": ["a", "b"]}}]'  # noqa
    })
    messages.append({'role': 'user', 'content': 'Compute (3+5)*2'})
    messages.append({
        'content':
        '(3+5)*2 = 8*2 =',
        'role':
        'assistant',
        'tool_calls': [{
            'id': '1',
            'function': {
                'arguments': '{"a": 8, "b": 2}',
                'name': 'mul'
            },
            'type': 'function'
        }]
    })
    messages.append({'role': 'tool', 'content': '3+5=16', 'tool_call_id': '1'})
    model = MODELS.get('internlm2')()
    assert model.messages2prompt(
        messages
    ) == """<|im_start|>system name=<|plugin|>\n[{"description": "Compute the sum of two numbers", "name": "add", "parameters": {"type": "object", "properties": {"a": {"type": "int", "description": "A number"}, "b": {"type": "int", "description": "A number"}}, "required": ["a", "b"]}}, {"description": "Calculate the product of two numbers", "name": "mul", "parameters": {"type": "object", "properties": {"a": {"type": "int", "description": "A number"}, "b": {"type": "int", "description": "A number"}}, "required": ["a", "b"]}}]<|im_end|>\n<|im_start|>user\nCompute (3+5)*2<|im_end|>\n<|im_start|>assistant\n(3+5)*2 = 8*2 =<|action_start|><|plugin|>\n{"name": "mul", "parameters": {"a": 8, "b": 2}}<|action_end|><|im_end|>\n<|im_start|>environment\n3+5=16<|im_end|>\n<|im_start|>assistant\n"""  # noqa


def test_messages2prompt4internlm2_chat():
    model = MODELS.get('internlm2')()
    # Test with a single message
    messages = [
        {
            'role': 'system',
            'name': 'interpreter',
            'content': 'You have access to python environment.'
        },
        {
            'role': 'user',
            'content': 'use python drwa a line'
        },
        {
            'role': 'assistant',
            'content': '<|action_start|><|interpreter|>\ncode<|action_end|>\n'
        },
        {
            'role': 'environment',
            'name': 'interpreter',
            'content': "[{'type': 'image', 'content': 'image url'}]"
        },
    ]
    tools = [{
        'type': 'function',
        'function': {
            'name': 'add',
            'description': 'Compute the sum of two numbers',
            'parameters': {
                'type': 'object',
                'properties': {
                    'a': {
                        'type': 'int',
                        'description': 'A number',
                    },
                    'b': {
                        'type': 'int',
                        'description': 'A number',
                    },
                },
                'required': ['a', 'b'],
            },
        }
    }]
    import json
    expected_prompt = (model.system.strip() + ' name=<|interpreter|>\nYou have access to python environment.' +
                       model.eosys + model.system.strip() +
                       f' name={model.plugin}\n{json.dumps(tools, ensure_ascii=False)}' + model.eosys + model.user +
                       'use python drwa a line' + model.eoh + model.assistant +
                       '<|action_start|><|interpreter|>\ncode<|action_end|>\n' + model.eoa + model.separator +
                       model.environment.strip() +
                       " name=<|interpreter|>\n[{'type': 'image', 'content': 'image url'}]" + model.eoenv +
                       model.assistant)
    actual_prompt = model.messages2prompt(messages, tools=tools)
    assert actual_prompt == expected_prompt

    # Test with a message where 'name' is not in name_map
    messages_invalid_name = [
        {
            'role': 'system',
            'name': 'invalid_name',
            'content': 'You have access to python environment.'
        },
        {
            'role': 'user',
            'content': 'use python draw a line'
        },
        {
            'role': 'assistant',
            'content': '\ncode\n'
        },
        {
            'role': 'environment',
            'name': 'invalid_name',
            'content': "[{'type': 'image', 'content': 'image url'}]"
        },
    ]
    expected_prompt_invalid_name = (model.system.strip() + '\nYou have access to python environment.' + model.eosys +
                                    model.user + 'use python draw a line' + model.eoh + model.assistant + '\ncode\n' +
                                    model.eoa + model.separator + model.environment.strip() +
                                    "\n[{'type': 'image', 'content': 'image url'}]" + model.eoenv + model.assistant)
    actual_prompt_invalid_name = model.messages2prompt(messages_invalid_name)
    assert actual_prompt_invalid_name == expected_prompt_invalid_name


def test_internvl_phi3():
    model_path_and_name = 'OpenGVLab/Mini-InternVL-Chat-4B-V1-5'
    model = MODELS.get('internvl-phi3')()
    messages = [{
        'role': 'user',
        'content': 'who are you'
    }, {
        'role': 'assistant',
        'content': 'I am an AI'
    }, {
        'role': 'user',
        'content': 'hi'
    }]
    res = model.messages2prompt(messages)
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=model_path_and_name, filename='conversation.py', local_dir='.')

    try:
        import os

        from conversation import get_conv_template
        template = get_conv_template('phi3-chat')
        template.append_message(template.roles[0], messages[0]['content'])
        template.append_message(template.roles[1], messages[1]['content'])
        ref = template.get_prompt()
        assert res.startswith(ref)
        if os.path.exists('conversation.py'):
            os.remove('conversation.py')
    except ImportError:
        pass


def test_internvl2():
    model = MODELS.get('internvl2-internlm2')()
    messages = [{'role': 'user', 'content': 'who are you'}, {'role': 'assistant', 'content': 'I am an AI'}]
    expected = '<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的'\
        '书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'\
        '<|im_end|><|im_start|>user\nwho are you<|im_end|><|im_start|>'\
        'assistant\nI am an AI'
    res = model.messages2prompt(messages)
    assert res == expected


@pytest.parametrize('model_path',
                    ['Qwen/Qwen2.5-7B-Instruct', 'deepseek-ai/DeepSeek-R1', 'internlm/internlm3-8b-instruct'])
def test_resolve_hf_chat_template_tokenizer(model_path):
    conversation = [{
        'role': 'system',
        'content': 'You are a helpful assistant.'
    }, {
        'role': 'user',
        'content': 'Who won the world series in 2020?'
    }, {
        'role': 'assistant',
        'content': 'The Los Angeles Dodgers won the World Series in 2020.'
    }, {
        'role': 'user',
        'content': 'Where was it played?'
    }]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ref = tokenizer.apply_chat_template(conversation)
    chat_template = resolve_hf_chat_template(tokenizer)
    res = chat_template.apply_chat_template(conversation)
    assert res == ref


@pytest.parametrize('model_path, chat_template_name', [('internlm/internlm2_5-7b-chat', 'internlm2'),
                                                       ('internlm/internlm3-8b-instruct', 'internlm3')])
def test_resolve_hf_chat_template_config(model_path, chat_template_name):
    conversation = [{
        'role': 'system',
        'content': 'You are a helpful assistant.'
    }, {
        'role': 'user',
        'content': 'Who won the world series in 2020?'
    }, {
        'role': 'assistant',
        'content': 'The Los Angeles Dodgers won the World Series in 2020.'
    }, {
        'role': 'user',
        'content': 'Where was it played?'
    }]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ref = tokenizer.apply_chat_template(conversation)
    chat_template_config = ChatTemplateConfig(model_name=chat_template_name)
    chat_template = resolve_hf_chat_template(tokenizer, chat_template_config)
    res = chat_template.apply_chat_template(conversation)
    assert res == ref
