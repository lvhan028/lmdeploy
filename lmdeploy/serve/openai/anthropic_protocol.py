# Copyright (c) OpenMMLab. All rights reserved.
# Anthropic Messages API protocol definitions.
# Spec: https://docs.anthropic.com/en/api/messages
import json
import time
from typing import Any, Dict, List, Literal, Optional, Union

import shortuuid
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Content block types used in *requests*
# ---------------------------------------------------------------------------

class AnthropicImageSource(BaseModel):
    """Source of an image content block."""
    type: Literal['base64', 'url']
    # For base64 type
    media_type: Optional[Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']] = None
    data: Optional[str] = None
    # For url type
    url: Optional[str] = None


class AnthropicTextContent(BaseModel):
    """Text content block."""
    type: Literal['text']
    text: str


class AnthropicImageContent(BaseModel):
    """Image content block."""
    type: Literal['image']
    source: AnthropicImageSource


class AnthropicToolUseContent(BaseModel):
    """Tool use content block (in assistant messages)."""
    type: Literal['tool_use']
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


class AnthropicToolResultContent(BaseModel):
    """Tool result content block (in user messages)."""
    type: Literal['tool_result']
    tool_use_id: str
    content: Optional[Union[str, List[Union[AnthropicTextContent, AnthropicImageContent]]]] = None
    is_error: Optional[bool] = None


# Discriminated union of all request content block types
AnthropicContentBlock = Union[
    AnthropicTextContent,
    AnthropicImageContent,
    AnthropicToolUseContent,
    AnthropicToolResultContent,
]


class AnthropicMessage(BaseModel):
    """A single message in the Anthropic conversation."""
    role: Literal['user', 'assistant']
    content: Union[str, List[AnthropicContentBlock]]


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class AnthropicTool(BaseModel):
    """Tool definition for the Anthropic Messages API."""
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class AnthropicAutoToolChoice(BaseModel):
    type: Literal['auto'] = 'auto'


class AnthropicAnyToolChoice(BaseModel):
    type: Literal['any'] = 'any'


class AnthropicSpecificToolChoice(BaseModel):
    type: Literal['tool'] = 'tool'
    name: str


AnthropicToolChoice = Union[AnthropicAutoToolChoice, AnthropicAnyToolChoice, AnthropicSpecificToolChoice]


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class AnthropicMessagesRequest(BaseModel):
    """Request body for POST /v1/messages."""
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    system: Optional[Union[str, List[AnthropicTextContent]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    metadata: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Response content block types
# ---------------------------------------------------------------------------

class AnthropicResponseTextContent(BaseModel):
    """Text content block in a response."""
    type: Literal['text'] = 'text'
    text: str


class AnthropicResponseToolUseContent(BaseModel):
    """Tool use content block in a response."""
    type: Literal['tool_use'] = 'tool_use'
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


AnthropicResponseContent = Union[AnthropicResponseTextContent, AnthropicResponseToolUseContent]


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

class AnthropicUsage(BaseModel):
    """Token usage information."""
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------

class AnthropicMessagesResponse(BaseModel):
    """Response body for POST /v1/messages (non-streaming)."""
    id: str = Field(default_factory=lambda: f'msg_{shortuuid.random()}')
    type: Literal['message'] = 'message'
    role: Literal['assistant'] = 'assistant'
    content: List[AnthropicResponseContent] = Field(default_factory=list)
    model: str
    stop_reason: Optional[Literal['end_turn', 'max_tokens', 'stop_sequence', 'tool_use']] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# ---------------------------------------------------------------------------
# Streaming event models
# ---------------------------------------------------------------------------

class AnthropicStreamMessageStart(BaseModel):
    """message_start event."""
    type: Literal['message_start'] = 'message_start'
    message: AnthropicMessagesResponse


class AnthropicStreamContentBlockStart(BaseModel):
    """content_block_start event."""
    type: Literal['content_block_start'] = 'content_block_start'
    index: int
    content_block: AnthropicResponseContent


class AnthropicStreamPing(BaseModel):
    """ping event."""
    type: Literal['ping'] = 'ping'


class AnthropicTextDelta(BaseModel):
    type: Literal['text_delta'] = 'text_delta'
    text: str


class AnthropicInputJsonDelta(BaseModel):
    type: Literal['input_json_delta'] = 'input_json_delta'
    partial_json: str


class AnthropicStreamContentBlockDelta(BaseModel):
    """content_block_delta event."""
    type: Literal['content_block_delta'] = 'content_block_delta'
    index: int
    delta: Union[AnthropicTextDelta, AnthropicInputJsonDelta]


class AnthropicStreamContentBlockStop(BaseModel):
    """content_block_stop event."""
    type: Literal['content_block_stop'] = 'content_block_stop'
    index: int


class AnthropicMessageDelta(BaseModel):
    stop_reason: Optional[Literal['end_turn', 'max_tokens', 'stop_sequence', 'tool_use']] = None
    stop_sequence: Optional[str] = None


class AnthropicStreamMessageDelta(BaseModel):
    """message_delta event."""
    type: Literal['message_delta'] = 'message_delta'
    delta: AnthropicMessageDelta
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


class AnthropicStreamMessageStop(BaseModel):
    """message_stop event."""
    type: Literal['message_stop'] = 'message_stop'


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _finish_reason_to_stop_reason(
        finish_reason: Optional[str],
        stop_sequence_matched: Optional[str] = None,
) -> Optional[Literal['end_turn', 'max_tokens', 'stop_sequence', 'tool_use']]:
    """Map LMDeploy finish_reason to Anthropic stop_reason."""
    if finish_reason is None:
        return None
    if finish_reason == 'stop':
        if stop_sequence_matched is not None:
            return 'stop_sequence'
        return 'end_turn'
    if finish_reason == 'length':
        return 'max_tokens'
    if finish_reason == 'tool_calls':
        return 'tool_use'
    # 'error' or 'abort' → end_turn as fallback
    return 'end_turn'


def anthropic_messages_to_openai(
        messages: List[AnthropicMessage],
        system: Optional[Union[str, List[AnthropicTextContent]]] = None,
) -> List[Dict[str, Any]]:
    """Convert Anthropic messages to the OpenAI / lmdeploy internal format.

    The converted list can be passed directly to ``AsyncEngine.generate``.

    Args:
        messages: List of ``AnthropicMessage`` objects from the request.
        system: Optional system prompt (string or list of text blocks).

    Returns:
        List of OpenAI-style message dicts.
    """
    result: List[Dict[str, Any]] = []

    # Prepend system prompt when provided
    if system is not None:
        if isinstance(system, str):
            system_text = system
        else:
            system_text = ''.join(block.text for block in system)
        result.append({'role': 'system', 'content': system_text})

    for msg in messages:
        role = msg.role
        content = msg.content

        # Simple string content – pass through as-is
        if isinstance(content, str):
            result.append({'role': role, 'content': content})
            continue

        # ---------------------------------------------------------------
        # Array content – may contain mixed block types
        # ---------------------------------------------------------------
        # Separate tool_use / tool_result blocks from text/image blocks.
        tool_use_blocks: List[AnthropicToolUseContent] = []
        tool_result_blocks: List[AnthropicToolResultContent] = []
        other_blocks: List[Union[AnthropicTextContent, AnthropicImageContent]] = []

        for block in content:
            if isinstance(block, AnthropicToolUseContent):
                tool_use_blocks.append(block)
            elif isinstance(block, AnthropicToolResultContent):
                tool_result_blocks.append(block)
            else:
                other_blocks.append(block)

        # tool_result blocks from the *user* role become separate "tool" messages
        if tool_result_blocks:
            for tr in tool_result_blocks:
                tool_content: Union[str, List[Dict[str, Any]]]
                if tr.content is None:
                    tool_content = ''
                elif isinstance(tr.content, str):
                    tool_content = tr.content
                else:
                    # List of text/image blocks
                    parts = []
                    for c in tr.content:
                        if isinstance(c, AnthropicTextContent):
                            parts.append({'type': 'text', 'text': c.text})
                        elif isinstance(c, AnthropicImageContent):
                            parts.append(_image_block_to_openai(c))
                    tool_content = parts if len(parts) > 1 else (parts[0]['text'] if parts else '')
                result.append({
                    'role': 'tool',
                    'tool_call_id': tr.tool_use_id,
                    'content': tool_content,
                })
            # If there are also regular content blocks, add them as a user message
            if other_blocks:
                result.append({'role': role, 'content': _blocks_to_openai_content(other_blocks)})
            continue

        # tool_use blocks in an assistant message → OpenAI tool_calls format
        if tool_use_blocks and role == 'assistant':
            tool_calls = []
            for tu in tool_use_blocks:
                tool_calls.append({
                    'id': tu.id,
                    'type': 'function',
                    'function': {
                        'name': tu.name,
                        'arguments': json.dumps(tu.input),
                    },
                })
            # Text content alongside tool calls
            text_content = None
            if other_blocks:
                text_parts = [b.text for b in other_blocks if isinstance(b, AnthropicTextContent)]
                text_content = ''.join(text_parts) or None
            result.append({
                'role': 'assistant',
                'content': text_content,
                'tool_calls': tool_calls,
            })
            continue

        # Pure text / image content
        openai_content = _blocks_to_openai_content(other_blocks)
        result.append({'role': role, 'content': openai_content})

    return result


def _image_block_to_openai(block: AnthropicImageContent) -> Dict[str, Any]:
    """Convert an Anthropic image block to OpenAI image_url format."""
    src = block.source
    if src.type == 'url':
        url = src.url
    else:
        # base64 → data URI
        url = f'data:{src.media_type};base64,{src.data}'
    return {'type': 'image_url', 'image_url': {'url': url}}


def _blocks_to_openai_content(
        blocks: List[Union[AnthropicTextContent, AnthropicImageContent]],
) -> Union[str, List[Dict[str, Any]]]:
    """Convert a list of Anthropic text/image blocks to OpenAI content format.

    Returns a plain string when the list contains only a single text block,
    otherwise returns the OpenAI array-of-parts format.
    """
    if len(blocks) == 1 and isinstance(blocks[0], AnthropicTextContent):
        return blocks[0].text
    parts = []
    for block in blocks:
        if isinstance(block, AnthropicTextContent):
            parts.append({'type': 'text', 'text': block.text})
        elif isinstance(block, AnthropicImageContent):
            parts.append(_image_block_to_openai(block))
    return parts


def anthropic_tools_to_openai(tools: List[AnthropicTool]) -> List[Dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI function tool format."""
    result = []
    for tool in tools:
        result.append({
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.input_schema,
            },
        })
    return result


def anthropic_tool_choice_to_openai(
        tool_choice: Optional[AnthropicToolChoice],
) -> Union[str, Dict[str, Any]]:
    """Convert Anthropic tool_choice to OpenAI tool_choice format."""
    if tool_choice is None:
        return 'auto'
    if isinstance(tool_choice, AnthropicAutoToolChoice):
        return 'auto'
    if isinstance(tool_choice, AnthropicAnyToolChoice):
        return 'required'
    if isinstance(tool_choice, AnthropicSpecificToolChoice):
        return {'type': 'function', 'function': {'name': tool_choice.name}}
    return 'auto'


def openai_tool_calls_to_anthropic(
        tool_calls: Optional[List[Dict[str, Any]]],
) -> List[AnthropicResponseToolUseContent]:
    """Convert OpenAI tool_calls to Anthropic tool_use content blocks."""
    if not tool_calls:
        return []
    result = []
    for tc in tool_calls:
        fn = tc.get('function', {})
        args_str = fn.get('arguments', '{}')
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            args = {}
        result.append(
            AnthropicResponseToolUseContent(
                id=tc.get('id', f'toolu_{shortuuid.random()}'),
                name=fn.get('name', ''),
                input=args,
            ))
    return result


def build_anthropic_response(
        message_id: str,
        model: str,
        text: str,
        finish_reason: Optional[str],
        input_tokens: int,
        output_tokens: int,
        tool_calls: Optional[List] = None,
        stop_sequence_matched: Optional[str] = None,
) -> AnthropicMessagesResponse:
    """Build a complete non-streaming Anthropic response."""
    content: List[AnthropicResponseContent] = []

    if text:
        content.append(AnthropicResponseTextContent(text=text))

    if tool_calls:
        for tc in tool_calls:
            content.append(tc)

    stop_reason = _finish_reason_to_stop_reason(finish_reason, stop_sequence_matched=stop_sequence_matched)

    return AnthropicMessagesResponse(
        id=message_id,
        model=model,
        content=content,
        stop_reason=stop_reason,
        stop_sequence=stop_sequence_matched,
        usage=AnthropicUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def format_sse_event(event_type: str, data: str) -> str:
    """Format a Server-Sent Event string."""
    return f'event: {event_type}\ndata: {data}\n\n'
