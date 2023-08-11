# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype


def prepare_tensor(name, input_tensor):
    """Create grpcclient's InferInput instance according to a given tensor."""
    print(name, input_tensor.shape)
    t = grpcclient.InferInput(name, list(input_tensor.shape),
                              np_to_triton_dtype(input_tensor.dtype))
    t.set_data_from_numpy(input_tensor)
    return t


class Preprocessor:
    """Tokenize prompts.

    Args:
        tritonserver_addr (str): the communication address of the inference
          server
    """

    def __init__(self, tritonserver_addr: str):
        self.tritonserver_addr = tritonserver_addr
        self.model_name = 'preprocessing'

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def infer(self, prompts: Union[str, List[str]]) -> tuple:
        """Tokenize the input prompts.

        Args:
            prompts(str | List[str]): user's prompt, or a batch prompts

        Returns:
            Tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray): prompt's token
            ids, ids' length and requested output length
        """
        if isinstance(prompts, str):
            input0 = [[prompts]]
        elif isinstance(prompts, List):
            input0 = [[prompt] for prompt in prompts]
        else:
            assert 0, f'str or List[str] prompts are expected but got ' \
                      f'{type(prompts)}'

        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32)
        inputs = [
            prepare_tensor('QUERY', input0_data),
            prepare_tensor('REQUEST_OUTPUT_LEN', output0_len)
        ]

        with grpcclient.InferenceServerClient(self.tritonserver_addr) as \
                client:
            result = client.infer(self.model_name, inputs)
            output0 = result.as_numpy('INPUT_ID')
            output1 = result.as_numpy('REQUEST_INPUT_LEN')
        return output0, output1


class Postprocessor:
    """De-tokenize prompts.

    Args:
        tritonserver_addr (str): the communication address of the inference
          server
    """

    def __init__(self, tritonserver_addr: str):
        self.tritonserver_addr = tritonserver_addr

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def infer(self, prev_token_ids: np.ndarray, prev_token_texts: np.ndarray,
              token_ids: np.ndarray):
        """De-tokenize tokens for text.

        Args:
            prev_token_ids(np.ndarray): an array of token_id of
                previously decoded tokens
            prev_token_texts(np.ndarray): an array of string of
                previously decoded tokens
            token_ids(np.ndarray): an array of to-be-decoded tokens

        Returns:
            new_token_text: The new token as a string.
            output_text: The new output text as a string.
        """
        inputs = [
            prepare_tensor('prev_token_ids', prev_token_ids),
            prepare_tensor('prev_token_texts', prev_token_texts),
            prepare_tensor('token_ids', token_ids)
        ]

        model_name = 'postprocessing'
        with grpcclient.InferenceServerClient(self.tritonserver_addr) \
                as client:
            result = client.infer(model_name, inputs)
            new_token_text = result[0].as_numpy('new_token_text')
            output_text = result[1].as_numpy('output_text')

        return new_token_text, output_text
