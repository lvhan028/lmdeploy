# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import torch

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig
from lmdeploy.utils import get_logger
from lmdeploy.vl.model.builder import load_vl_model

logger = get_logger('lmdeploy')


def _raise_exception_on_finish(task: asyncio.Task) -> None:
    """Raise exception on finish."""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as e:
        raise e


class ImageEncoder:
    """Image encoder."""

    def __init__(
        self,
        model_path: str,
        backend: str,
        vision_config: VisionConfig = None,
        backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None,
    ):
        self.model = load_vl_model(model_path, backend, backend_config=backend_config)
        if vision_config is None:
            vision_config = VisionConfig()
        self.vision_config = vision_config
        self.max_batch_size = vision_config.max_batch_size
        self.executor = ThreadPoolExecutor(max_workers=1)
        torch.cuda.empty_cache()

    async def preprocess(self, messages: List[Dict]) -> List[Dict]:
        """Preprocess multimodal data in the messages."""
        future = asyncio.get_event_loop().run_in_executor(self.executor, self.model.preprocess, messages)
        future.add_done_callback(_raise_exception_on_finish)
        outputs = await future
        return outputs

    async def async_infer(self, messages: List[Dict]) -> List[Dict]:
        """Get multimodal embedding.

        Args:
            messages (List[Dict]): a list of message, which is the output
            of `preprocess()`
        """
        future = asyncio.get_event_loop().run_in_executor(self.executor, self.model.forward, messages,
                                                          self.max_batch_size)
        future.add_done_callback(_raise_exception_on_finish)
        outputs = await future
        return outputs

    async def wrap_for_pytorch(self, messages: List[Dict], chat_template, tokenizer) -> List[Dict]:
        """Convert multimodal message data into a format suitable for the
        PyTorch engine.

        Args:
            messages (List[Dict]): A list of messages, expected to be the output of the `preprocess` method.
            chat_template: The chat template used to format the prompt information.
            tokenizer: The tokenizer used to convert text into input IDs.

        Returns:
            A dictionary that will be passed to the forward propagation method of the PyTorch engine instance.
            The dictionary structure is as follows:
            Dict(
                'prompt': 'The prompt after applying the chat template',
                'input_ids': [],
                'multimodal': {
                    'pixel_values': torch.Tensor,
                    ...
                }
            )
        """
        result = self.model.to_pytorch(messages, chat_template, tokenizer)
        # clear temp data
        for i, message in enumerate(messages):
            if isinstance(message['content'], List):
                messages[i]['preprocess'] = None
        return result

    async def wrap_for_turbomind(self, messages: List[Dict], chat_template, tokenizer) -> Dict:
        """This method is used to convert multimodal message data into a format
        suitable for the Turbomind engine.

        Args:
            messages (List[Dict]): A list of messages, which is expected to be the output of the `async_infer` method.
            chat_template: The chat template used to format the prompt information.
            tokenizer: The tokenizer used to convert text into input IDs.

        Returns:
            A dictionary that will be passed to the forward propagation method of the Turbomind engine instance.
            The dictionary structure is as follows:
            Dict(
                'prompt': 'The prompt after applying the chat template',
                'input_ids': [],
                'input_embeddings': list[torch.Tensor],
                'input_embedding_ranges': list[torch.Tensor],
                ...
            )
        """
        result = self.model.to_turbomind(messages, chat_template, tokenizer)
        # clear temp data
        for i, message in enumerate(messages):
            if isinstance(message['content'], List):
                messages[i]['preprocess'] = None
                messages[i]['forward'] = None
        return result
