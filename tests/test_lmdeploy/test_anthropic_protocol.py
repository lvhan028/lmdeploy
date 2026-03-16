# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests for lmdeploy/serve/openai/anthropic_protocol.py.

These tests exercise the conversion helpers and Pydantic models without
requiring a running engine, GPU, or heavy deep-learning dependencies.

The module is loaded via ``importlib`` so that the ``lmdeploy`` package
``__init__.py`` (which requires torch) is not triggered during collection.
"""
import importlib.util
import json
import os

import pytest

# ---------------------------------------------------------------------------
# Load the module directly to avoid triggering lmdeploy/__init__.py
# ---------------------------------------------------------------------------
_MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', '..', 'lmdeploy', 'serve', 'openai', 'anthropic_protocol.py',
)
_spec = importlib.util.spec_from_file_location('anthropic_protocol', _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

AnthropicAnyToolChoice = _mod.AnthropicAnyToolChoice
AnthropicAutoToolChoice = _mod.AnthropicAutoToolChoice
AnthropicImageContent = _mod.AnthropicImageContent
AnthropicImageSource = _mod.AnthropicImageSource
AnthropicMessage = _mod.AnthropicMessage
AnthropicMessagesRequest = _mod.AnthropicMessagesRequest
AnthropicMessagesResponse = _mod.AnthropicMessagesResponse
AnthropicResponseTextContent = _mod.AnthropicResponseTextContent
AnthropicResponseToolUseContent = _mod.AnthropicResponseToolUseContent
AnthropicSpecificToolChoice = _mod.AnthropicSpecificToolChoice
AnthropicTextContent = _mod.AnthropicTextContent
AnthropicTool = _mod.AnthropicTool
AnthropicToolResultContent = _mod.AnthropicToolResultContent
AnthropicToolUseContent = _mod.AnthropicToolUseContent
AnthropicUsage = _mod.AnthropicUsage

_finish_reason_to_stop_reason = _mod._finish_reason_to_stop_reason
anthropic_messages_to_openai = _mod.anthropic_messages_to_openai
anthropic_tool_choice_to_openai = _mod.anthropic_tool_choice_to_openai
anthropic_tools_to_openai = _mod.anthropic_tools_to_openai
build_anthropic_response = _mod.build_anthropic_response
format_sse_event = _mod.format_sse_event
openai_tool_calls_to_anthropic = _mod.openai_tool_calls_to_anthropic


# ---------------------------------------------------------------------------
# _finish_reason_to_stop_reason
# ---------------------------------------------------------------------------

class TestFinishReasonToStopReason:

    def test_none(self):
        assert _finish_reason_to_stop_reason(None) is None

    def test_stop_without_sequence(self):
        assert _finish_reason_to_stop_reason('stop') == 'end_turn'

    def test_stop_with_sequence_matched(self):
        assert _finish_reason_to_stop_reason('stop', stop_sequence_matched='\n\nHuman:') == 'stop_sequence'

    def test_length(self):
        assert _finish_reason_to_stop_reason('length') == 'max_tokens'

    def test_tool_calls(self):
        assert _finish_reason_to_stop_reason('tool_calls') == 'tool_use'

    def test_error(self):
        # error/abort fall back to end_turn
        assert _finish_reason_to_stop_reason('error') == 'end_turn'
        assert _finish_reason_to_stop_reason('abort') == 'end_turn'


# ---------------------------------------------------------------------------
# anthropic_messages_to_openai
# ---------------------------------------------------------------------------

class TestAnthropicMessagesToOpenai:

    def test_simple_string_content(self):
        msgs = [AnthropicMessage(role='user', content='Hello')]
        result = anthropic_messages_to_openai(msgs)
        assert result == [{'role': 'user', 'content': 'Hello'}]

    def test_system_prompt_string(self):
        msgs = [AnthropicMessage(role='user', content='Hi')]
        result = anthropic_messages_to_openai(msgs, system='You are helpful.')
        assert result[0] == {'role': 'system', 'content': 'You are helpful.'}
        assert result[1] == {'role': 'user', 'content': 'Hi'}

    def test_system_prompt_text_blocks(self):
        system_blocks = [
            AnthropicTextContent(type='text', text='Part1 '),
            AnthropicTextContent(type='text', text='Part2'),
        ]
        msgs = [AnthropicMessage(role='user', content='Hi')]
        result = anthropic_messages_to_openai(msgs, system=system_blocks)
        assert result[0] == {'role': 'system', 'content': 'Part1 Part2'}

    def test_single_text_block(self):
        msgs = [AnthropicMessage(role='user', content=[AnthropicTextContent(type='text', text='Hello')])]
        result = anthropic_messages_to_openai(msgs)
        # Single text block should be unwrapped to a plain string
        assert result == [{'role': 'user', 'content': 'Hello'}]

    def test_multiple_text_blocks(self):
        msgs = [
            AnthropicMessage(role='user', content=[
                AnthropicTextContent(type='text', text='Line1'),
                AnthropicTextContent(type='text', text='Line2'),
            ])
        ]
        result = anthropic_messages_to_openai(msgs)
        assert result == [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Line1'},
            {'type': 'text', 'text': 'Line2'},
        ]}]

    def test_image_url_block(self):
        msgs = [
            AnthropicMessage(role='user', content=[
                AnthropicImageContent(
                    type='image',
                    source=AnthropicImageSource(type='url', url='https://example.com/img.png'),
                )
            ])
        ]
        result = anthropic_messages_to_openai(msgs)
        assert result == [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'https://example.com/img.png'}}
        ]}]

    def test_image_base64_block(self):
        msgs = [
            AnthropicMessage(role='user', content=[
                AnthropicImageContent(
                    type='image',
                    source=AnthropicImageSource(type='base64', media_type='image/jpeg', data='abc123'),
                )
            ])
        ]
        result = anthropic_messages_to_openai(msgs)
        assert result[0]['content'][0]['image_url']['url'].startswith('data:image/jpeg;base64,')

    def test_tool_use_in_assistant_message(self):
        msgs = [
            AnthropicMessage(role='assistant', content=[
                AnthropicToolUseContent(
                    type='tool_use',
                    id='toolu_01',
                    name='get_weather',
                    input={'location': 'Paris'},
                )
            ])
        ]
        result = anthropic_messages_to_openai(msgs)
        assert len(result) == 1
        msg = result[0]
        assert msg['role'] == 'assistant'
        assert len(msg['tool_calls']) == 1
        tc = msg['tool_calls'][0]
        assert tc['id'] == 'toolu_01'
        assert tc['function']['name'] == 'get_weather'
        assert json.loads(tc['function']['arguments']) == {'location': 'Paris'}

    def test_tool_result_in_user_message(self):
        msgs = [
            AnthropicMessage(role='user', content=[
                AnthropicToolResultContent(
                    type='tool_result',
                    tool_use_id='toolu_01',
                    content='Sunny, 72°F',
                )
            ])
        ]
        result = anthropic_messages_to_openai(msgs)
        assert len(result) == 1
        msg = result[0]
        assert msg['role'] == 'tool'
        assert msg['tool_call_id'] == 'toolu_01'
        assert msg['content'] == 'Sunny, 72°F'

    def test_tool_result_none_content(self):
        msgs = [
            AnthropicMessage(role='user', content=[
                AnthropicToolResultContent(type='tool_result', tool_use_id='toolu_02', content=None)
            ])
        ]
        result = anthropic_messages_to_openai(msgs)
        assert result[0]['content'] == ''

    def test_mixed_text_and_tool_use(self):
        msgs = [
            AnthropicMessage(role='assistant', content=[
                AnthropicTextContent(type='text', text='Calling tool:'),
                AnthropicToolUseContent(
                    type='tool_use',
                    id='toolu_02',
                    name='search',
                    input={'query': 'AI'},
                ),
            ])
        ]
        result = anthropic_messages_to_openai(msgs)
        assert len(result) == 1
        msg = result[0]
        assert msg['content'] == 'Calling tool:'
        assert len(msg['tool_calls']) == 1

    def test_multi_turn_conversation(self):
        msgs = [
            AnthropicMessage(role='user', content='Hello'),
            AnthropicMessage(role='assistant', content='Hi there!'),
            AnthropicMessage(role='user', content='How are you?'),
        ]
        result = anthropic_messages_to_openai(msgs)
        assert len(result) == 3
        assert result[0]['role'] == 'user'
        assert result[1]['role'] == 'assistant'
        assert result[2]['role'] == 'user'


# ---------------------------------------------------------------------------
# anthropic_tools_to_openai
# ---------------------------------------------------------------------------

class TestAnthropicToolsToOpenai:

    def test_basic_tool(self):
        tools = [
            AnthropicTool(
                name='get_weather',
                description='Get current weather',
                input_schema={
                    'type': 'object',
                    'properties': {'location': {'type': 'string'}},
                    'required': ['location'],
                },
            )
        ]
        result = anthropic_tools_to_openai(tools)
        assert len(result) == 1
        t = result[0]
        assert t['type'] == 'function'
        assert t['function']['name'] == 'get_weather'
        assert t['function']['description'] == 'Get current weather'
        assert t['function']['parameters']['required'] == ['location']

    def test_tool_without_description(self):
        tools = [AnthropicTool(name='noop', input_schema={})]
        result = anthropic_tools_to_openai(tools)
        assert result[0]['function']['description'] is None

    def test_multiple_tools(self):
        tools = [
            AnthropicTool(name='a', input_schema={}),
            AnthropicTool(name='b', input_schema={}),
        ]
        result = anthropic_tools_to_openai(tools)
        assert [t['function']['name'] for t in result] == ['a', 'b']


# ---------------------------------------------------------------------------
# anthropic_tool_choice_to_openai
# ---------------------------------------------------------------------------

class TestAnthropicToolChoiceToOpenai:

    def test_none(self):
        assert anthropic_tool_choice_to_openai(None) == 'auto'

    def test_auto(self):
        assert anthropic_tool_choice_to_openai(AnthropicAutoToolChoice()) == 'auto'

    def test_any(self):
        assert anthropic_tool_choice_to_openai(AnthropicAnyToolChoice()) == 'required'

    def test_specific_tool(self):
        result = anthropic_tool_choice_to_openai(AnthropicSpecificToolChoice(name='get_weather'))
        assert result == {'type': 'function', 'function': {'name': 'get_weather'}}


# ---------------------------------------------------------------------------
# openai_tool_calls_to_anthropic
# ---------------------------------------------------------------------------

class TestOpenaiToolCallsToAnthropic:

    def test_empty(self):
        assert openai_tool_calls_to_anthropic(None) == []
        assert openai_tool_calls_to_anthropic([]) == []

    def test_single_tool_call(self):
        tc = [{
            'id': 'toolu_01',
            'function': {
                'name': 'get_weather',
                'arguments': '{"location": "Paris"}',
            },
        }]
        result = openai_tool_calls_to_anthropic(tc)
        assert len(result) == 1
        block = result[0]
        assert isinstance(block, AnthropicResponseToolUseContent)
        assert block.id == 'toolu_01'
        assert block.name == 'get_weather'
        assert block.input == {'location': 'Paris'}

    def test_invalid_json_arguments(self):
        tc = [{'id': 'x', 'function': {'name': 'f', 'arguments': 'not-json'}}]
        result = openai_tool_calls_to_anthropic(tc)
        assert result[0].input == {}

    def test_multiple_tool_calls(self):
        tc = [
            {'id': 'a', 'function': {'name': 'f1', 'arguments': '{}'}},
            {'id': 'b', 'function': {'name': 'f2', 'arguments': '{"x": 1}'}},
        ]
        result = openai_tool_calls_to_anthropic(tc)
        assert len(result) == 2
        assert result[1].input == {'x': 1}


# ---------------------------------------------------------------------------
# build_anthropic_response
# ---------------------------------------------------------------------------

class TestBuildAnthropicResponse:

    def test_simple_text_response(self):
        resp = build_anthropic_response(
            message_id='msg_1',
            model='test-model',
            text='Hello!',
            finish_reason='stop',
            input_tokens=10,
            output_tokens=5,
        )
        assert isinstance(resp, AnthropicMessagesResponse)
        assert resp.id == 'msg_1'
        assert resp.model == 'test-model'
        assert resp.stop_reason == 'end_turn'
        assert len(resp.content) == 1
        assert isinstance(resp.content[0], AnthropicResponseTextContent)
        assert resp.content[0].text == 'Hello!'
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5

    def test_max_tokens_stop_reason(self):
        resp = build_anthropic_response(
            message_id='msg_2',
            model='m',
            text='truncated',
            finish_reason='length',
            input_tokens=1,
            output_tokens=1,
        )
        assert resp.stop_reason == 'max_tokens'

    def test_tool_use_response(self):
        tool_block = AnthropicResponseToolUseContent(id='t1', name='search', input={'q': 'AI'})
        resp = build_anthropic_response(
            message_id='msg_3',
            model='m',
            text='',
            finish_reason='tool_calls',
            input_tokens=5,
            output_tokens=10,
            tool_calls=[tool_block],
        )
        assert resp.stop_reason == 'tool_use'
        # No empty text block when text is empty
        assert all(isinstance(c, AnthropicResponseToolUseContent) for c in resp.content)
        assert resp.content[0].name == 'search'

    def test_empty_text_is_excluded(self):
        resp = build_anthropic_response(
            message_id='msg_4',
            model='m',
            text='',
            finish_reason='stop',
            input_tokens=0,
            output_tokens=0,
        )
        # Empty text string → no text block added
        assert len(resp.content) == 0

    def test_stop_sequence_matched(self):
        resp = build_anthropic_response(
            message_id='msg_5',
            model='m',
            text='Hi',
            finish_reason='stop',
            input_tokens=2,
            output_tokens=1,
            stop_sequence_matched='\n\nHuman:',
        )
        assert resp.stop_reason == 'stop_sequence'
        assert resp.stop_sequence == '\n\nHuman:'


# ---------------------------------------------------------------------------
# format_sse_event
# ---------------------------------------------------------------------------

class TestFormatSseEvent:

    def test_format(self):
        data = '{"type": "ping"}'
        result = format_sse_event('ping', data)
        assert result == f'event: ping\ndata: {data}\n\n'


# ---------------------------------------------------------------------------
# AnthropicMessagesRequest validation
# ---------------------------------------------------------------------------

class TestAnthropicMessagesRequestValidation:

    def test_minimal_valid_request(self):
        req = AnthropicMessagesRequest(
            model='claude-3-5-sonnet',
            messages=[AnthropicMessage(role='user', content='hi')],
            max_tokens=100,
        )
        assert req.model == 'claude-3-5-sonnet'
        assert req.max_tokens == 100
        assert req.stream is False

    def test_with_system_prompt(self):
        req = AnthropicMessagesRequest(
            model='m',
            messages=[AnthropicMessage(role='user', content='hi')],
            max_tokens=10,
            system='Be helpful.',
        )
        assert req.system == 'Be helpful.'

    def test_with_tools(self):
        req = AnthropicMessagesRequest(
            model='m',
            messages=[AnthropicMessage(role='user', content='hi')],
            max_tokens=10,
            tools=[AnthropicTool(name='f', input_schema={})],
        )
        assert len(req.tools) == 1
