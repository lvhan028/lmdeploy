# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

# This tokenizer is `lmdeploy/turbomind/tokenizer.py`. When an LLM is served
# by triton inference server, it has to be converted first by running
# `python lmdeploy/serve/turbomind/deploy.py`. Then
# `lmdeploy/turbomind/tokenizer.py` will be copied to `tokenizer/tokenizer.py`
from .tokenizer.tokenizer import Tokenizer


class TritonPythonModel:
    """Your Python model must use the same class name.

    Every Python model that is created must have "TritonPythonModel" as the
    class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device
          ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        self.model_config = model_config = json.loads(args['model_config'])

        # Parse model output configs
        output_config = pb_utils.get_output_config_by_name(
            model_config, 'new_token_text')

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type'])

        cur_folder = Path(__file__).parent

        self.tokenizer = Tokenizer(
            osp.join(
                cur_folder, self.model_config['parameters']['tokenizer_path']
                ['string_value']))

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get input tensors
            prev_token_ids = pb_utils.get_input_tensor_by_name(
                request, 'prev_token_ids').as_numpy().flatten().tolist()
            prev_token_texts = pb_utils.get_input_tensor_by_name(
                request, 'prev_token_texts').as_numpy().flatten().tolist()
            token_ids = pb_utils.get_input_tensor_by_name(
                request, 'token_ids').as_numpy().flatten().tolist()

            print(prev_token_ids, prev_token_texts, token_ids)

            prev_token_texts = [
                token_text.decode('utf-8') for token_text in prev_token_texts
            ]

            print(prev_token_ids, prev_token_texts, token_ids)

            # Postprocessing output data.
            new_token_text, output_text = self._postprocessing(
                prev_token_ids, prev_token_texts, token_ids)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            new_token_text = pb_utils.Tensor(
                'new_token_text',
                np.array(new_token_text).astype(self.output_dtype))
            output_text = pb_utils.Tensor(
                'output_text',
                np.array(output_text).astype(self.output_dtype))
            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[new_token_text, output_text])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.

        Implementing `finalize` function is optional. This function allows the
        model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

    def _postprocessing(self, prev_token_ids, prev_token_texts, new_token_ids):
        """decode token ids into texts."""

        for new_token_id in new_token_ids:
            new_token, output_text = self.tokenizer.decode_incrementally(
                prev_token_ids, prev_token_texts, new_token_id)
            if new_token is not None:
                prev_token_texts.append(new_token)
            prev_token_ids.append(new_token_id)

        # print(f'{new_token}')
        return [new_token], [output_text]
        # for prev_token_ids, prev_token_texts, new_token_id in zip(
        #         prev_token_batch, prev_token_text_batch, new_token_batch):
        #     new_token, output_text = self.tokenizer.decode_incrementally(
        #         prev_token_ids,
        #         prev_token_texts,
        #         new_token_id)
        #     outputs.append((new_token, output_text))
        # return outputs
