# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Sequence, Union

import torch


class SentencePieceTokenizer:
    """Tokenizer of sentencepiece.

    Args:
        model_file (str): the path of the tokenizer model
    """

    def __init__(self, model_file: str):
        from sentencepiece import SentencePieceProcessor
        self.model = SentencePieceProcessor(model_file=model_file)

    @property
    def vocab_size(self):
        """vocabulary size."""
        return self.model.vocab_size()

    @property
    def bos_token_id(self):
        """begine of the sentence token id."""
        return self.model.bos_id()

    @property
    def eos_token_id(self):
        """end of the sentence token id."""
        return self.model.eos_id()

    def encode(self, s: str):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
        Returns:
            list[int]: token ids
        """
        add_bos = False
        add_eos = False
        if s.find('<BOS>') != -1:
            s = s.replace('<BOS>', '')
            add_bos = True
        if s == '<EOS>':
            s = ''
            add_eos = True
        return self.model.Encode(s, add_bos=add_bos, add_eos=add_eos)

    def decode(self, t: Sequence[int]):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
        Returns:
            str: text of decoding tokens
        """
        if isinstance(t, torch.Tensor):
            t = t.tolist()
        return self.model.Decode(t)

    def decode_incrementally(self,
                             prev_output_tokens: List[Union[str, int]],
                             new_token_id: int,
                             skip_special_tokens: bool = True):
        prev_output_tokens.append(new_token_id)
        return self.model.Decode(prev_output_tokens)

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        import addict
        add_bos = False
        add_eos = False

        input_ids = self.model.Encode(s, add_bos=add_bos, add_eos=add_eos)
        return addict.Addict(input_ids=input_ids)


class HuggingFaceTokenizer:
    """Tokenizer of sentencepiece.

    Args:
        model_dir (str): the directory of the tokenizer model
    """

    def __init__(self, model_dir: str):
        from transformers import AutoTokenizer
        model_file = osp.join(model_dir, 'tokenizer.model')
        backend_tokenizer_file = osp.join(model_dir, 'tokenizer.json')
        model_file_exists = osp.exists(model_file)
        if not osp.exists(backend_tokenizer_file) and model_file_exists:
            print('WARNING: Can not find tokenizer.json. '
                  'It may take long time to initialize the tokenizer.')
        self.model = AutoTokenizer.from_pretrained(model_dir,
                                                   trust_remote_code=True)
        # save tokenizer.json to reuse
        if not osp.exists(backend_tokenizer_file) and model_file_exists:
            if hasattr(self.model, 'backend_tokenizer'):
                self.model.backend_tokenizer.save(backend_tokenizer_file)

    @property
    def vocab_size(self):
        """vocabulary size."""
        return self.model.vocab_size

    @property
    def bos_token_id(self):
        """begine of the sentence token id."""
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        """end of the sentence token id."""
        return self.model.eos_token_id

    def encode(self, s: str):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
        Returns:
            list[int]: token ids
        """
        add_special_tokens = False
        if s.find('<BOS>') != -1:
            s = s.replace('<BOS>', '<s>')
        if s == '<EOS>':
            s = '</s>'
        if len(s) == 0:
            add_special_tokens = True
        return self.model.encode(s, add_special_tokens=add_special_tokens)

    def decode(self, t: Sequence[int]):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
        Returns:
            str: text of decoding tokens
        """
        skip_special_tokens = True
        return self.model.decode(t, skip_special_tokens=skip_special_tokens)

    def decode_incrementally(self,
                             prev_token_ids: Sequence[int],
                             prev_output_tokens: List[str],
                             new_token_id: int,
                             skip_special_tokens: bool = True):
        #
        if skip_special_tokens and (new_token_id
                                    in self.model.all_special_ids):
            return None, prev_output_tokens
        new_token = self.model.convert_ids_to_tokens(
            new_token_id, skip_special_tokens=skip_special_tokens)
        output_tokens = prev_output_tokens + [new_token]

        # Convert the tokens to a string.
        # Optimization: If the tokenizer does not have `added_tokens_encoder`,
        # then we can directly use `convert_tokens_to_string`.
        if not getattr(self.model, 'added_tokens_encoder', {}):
            output_text = self.model.convert_tokens_to_string(output_tokens)
            return new_token, output_text

        # Adapted from
        # https://github.com/huggingface/transformers/blob/v4.28.0/
        # src/transformers/tokenization_utils.py#L921
        # NOTE(woosuk): The following code is slow because it runs a for
        # loop over the output_tokens. In Python, running a for loop over
        # a list can be slow even when the loop body is very simple.
        sub_texts = []
        current_sub_text = []
        for token in output_tokens:
            if skip_special_tokens and token in self.model.all_special_tokens:
                continue
            if token in self.model.added_tokens_encoder:
                if current_sub_text:
                    sub_text = tokenizer.convert_tokens_to_string(
                        current_sub_text)
                    sub_texts.append(sub_text)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_text = self.model.convert_tokens_to_string(current_sub_text)
            sub_texts.append(sub_text)
        output_text = ' '.join(sub_texts)
        return new_token, output_text

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        add_special_tokens = False
        return self.model(s, add_special_tokens=add_special_tokens)


class Tokenizer:
    """Tokenize prompts or de-tokenize tokens into texts.

    Args:
        model_file (str): the path of the tokenizer model
    """

    def __init__(self, model_file: str):
        if model_file.endswith('.model'):
            model_folder = osp.split(model_file)[0]
        else:
            model_folder = model_file
            model_file = osp.join(model_folder, 'tokenizer.model')
        tokenizer_config_file = osp.join(model_folder, 'tokenizer_config.json')

        model_file_exists = osp.exists(model_file)
        config_exists = osp.exists(tokenizer_config_file)
        use_hf_model = config_exists or not model_file_exists

        if not use_hf_model:
            self.model = SentencePieceTokenizer(model_file)
        else:
            self.model = HuggingFaceTokenizer(model_folder)

    @property
    def vocab_size(self):
        """vocabulary size."""
        return self.model.vocab_size

    @property
    def bos_token_id(self):
        """begine of the sentence token id."""
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        """end of the sentence token id."""
        return self.model.eos_token_id

    def encode(self, s: str):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
        Returns:
            list[int]: token ids
        """
        return self.model.encode(s)

    def decode(self, t: Sequence[int]):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
        Returns:
            str: text of decoding tokens
        """
        return self.model.decode(t)

    def decode_incrementally(self,
                             prev_token_ids: Sequence[int],
                             prev_output_tokens: List[str],
                             new_token_id: int,
                             skip_special_tokens: bool = True):
        """Detokenizes the new token in conjunction with the previous output
        tokens.

        NOTE: This function does not update prev_output_tokens.

        Returns:
            new_token: The new token as a string.
            output_text: The new output text as a string.
        """
        return self.model.decode_incrementally(prev_output_tokens,
                                               new_token_id,
                                               skip_special_tokens)

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        return self.model(s)


if __name__ == '__main__':
    # tokenizer_path = './llama-2-7b-chat/tokenizer.model'
    tokenizer_path = '/nvme/shared_data/chatpjlm-0/llamav4.model'

    tokenizer = Tokenizer(tokenizer_path)
    token_ids = tokenizer.encode('hi, welcome to shanghai')
    print(token_ids)

    print('de-tokenize one by one >> ')
    for i in range(len(token_ids)):
        text = tokenizer.decode(token_ids[i])
        print(text, end='', flush=True)
    print()

    print('de-tokenize all >>')
    for i in range(len(token_ids)):
        text = tokenizer.decode(token_ids[:i + 1])
        print(text, flush=True)
    print()

    print('de-tokenize incrementally >>')
    prev_token_ids = []
    prev_token_texts = []
    for i in range(len(token_ids)):
        new_token, output_text = tokenizer.decode_incrementally(
            prev_token_ids, prev_token_texts, token_ids[i])
        if new_token is not None:
            prev_token_texts.append(new_token)
        prev_token_ids.append(token_ids[i])
        print(f'output_text: {output_text}')
        # print(f'new_token_id: {token_ids[i]}, new_token_text: {new_token}, '
        #       f'output_text: {output_text}')
