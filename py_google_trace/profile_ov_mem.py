import csv
import os
from dataclasses import dataclass
from pathlib import Path

import psutil
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from optimum.intel.openvino import OVModelForCausalLM
import openvino.runtime as ov
import torch

from new_greedy_search import new_greedy_search
from my_py_profile import MyProfile, release_manager

@dataclass
class GenerationParameters:
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0

    max_new_tokens: int = 60
    min_new_tokens: int = 0

def main():
    RUN_OV=1
    if RUN_OV:
        model_name = "chgk13/decicoder-1b-openvino-int8"
        model_name="../../llm_internal_test/models/decicoder-1b-openvino-int8/pytorch/dldt/FP32/"
        model_name="../../llm_internal_test/models/llama-2-7b-chat/INT8_compressed_weights/"
        print(f"model_name={model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = OVModelForCausalLM.from_pretrained(model_name, cache_dir="./tmp")
    else:
        pytorch_model="../models_original/llama-2-7b-chat/pytorch"
        print(f"model_name={pytorch_model}")
        tokenizer = AutoTokenizer.from_pretrained(pytorch_model)
        model = AutoModelForCausalLM.from_pretrained(pytorch_model, trust_remote_code=True).to("cpu", dtype=torch.float32)

    text = '''# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeBERTa-v2 model."""

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import softmax_backward_data
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DebertaV2Config"
_CHECKPOINT_FOR_DOC = "microsoft/deberta-v2-xlarge"
_QA_TARGET_START_INDEX = 2
_QA_TARGET_END_INDEX = 9

DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/deberta-v2-xlarge",
    "microsoft/deberta-v2-xxlarge",
    "microsoft/deberta-v2-xlarge-mnli",
    "microsoft/deberta-v2-xxlarge-mnli",
]


# Copied from transformers.models.deberta.modeling_deberta.ContextPooler
class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


# Copied from transformers.models.deberta.modeling_deberta.XSoftmax with deberta->deberta_v2
class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`torch.tensor`): The input tensor that will apply softmax.
        mask (`torch.IntTensor`):
            The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example:

    ```python
    >>> import torch
    >>> from transformers.models.deberta_v2.modeling_deberta_v2 import XSoftmax

    >>> # Make a tensor
    >>> x = torch.randn([4, 20, 100])

    >>> # Create a mask
    >>> mask = (x > 0).int()

    >>> # Specify the dimension to apply softmax
    >>> dim = -1

    >>> y = XSoftmax.apply(x, mask, dim)
    ```"""

    @staticmethod
    def forward(self, input, mask, dim):
    '''
    input_tokens = 0
    output_tokens = 0

    data_file = Path("ram_usage.csv")

    def get_memory_usage():
        info = psutil.Process().memory_full_info()
        return info.rss, info.uss, info.shared, info.vms, info.swap

    # Hook greedy search.
    bound_method = new_greedy_search.__get__(model, model.__class__)
    model.greedy_search = bound_method
    
    with open(data_file, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["InputTokens", "OutputTokens", "RSS", "USS", "Shared", "VMS", "Swap"])
        writer.writerow([input_tokens, output_tokens, *get_memory_usage()])
        text_old=text

        text=text_old
        token_ids = tokenizer(text, return_tensors="pt").input_ids
        for _ in tqdm(range(2)):
            input_tokens = token_ids.shape[-1]
            if RUN_OV:
                model.compile()
            token_ids = model.generate(token_ids, **(GenerationParameters(top_p=1).__dict__))
            output_tokens = token_ids.shape[-1]
            writer.writerow([input_tokens, output_tokens, *get_memory_usage()])

        text = "def fibonacci(n):"
        token_ids = tokenizer(text, return_tensors="pt").input_ids
        for _ in tqdm(range(2)):
            input_tokens = token_ids.shape[-1]
            token_ids = model.generate(token_ids, **(GenerationParameters(top_p=1).__dict__))
            output_tokens = token_ids.shape[-1]
            writer.writerow([input_tokens, output_tokens, *get_memory_usage()])

        text = "def fibonacci(n):"
        token_ids = tokenizer(text, return_tensors="pt").input_ids
        for _ in tqdm(range(2)):
            input_tokens = token_ids.shape[-1]
            token_ids = model.generate(token_ids, **(GenerationParameters(top_p=1).__dict__))
            output_tokens = token_ids.shape[-1]
            writer.writerow([input_tokens, output_tokens, *get_memory_usage()])

        text = "def fibonacci(n):"
        token_ids = tokenizer(text, return_tensors="pt").input_ids
        for _ in tqdm(range(2)):
            input_tokens = token_ids.shape[-1]
            token_ids = model.generate(token_ids, **(GenerationParameters(top_p=1).__dict__))
            output_tokens = token_ids.shape[-1]
            writer.writerow([input_tokens, output_tokens, *get_memory_usage()])
        
        text=text_old
        token_ids = tokenizer(text, return_tensors="pt").input_ids
        for _ in tqdm(range(2)):
            input_tokens = token_ids.shape[-1]
            if RUN_OV:
                model.compile()
            token_ids = model.generate(token_ids, **(GenerationParameters(top_p=1).__dict__))
            output_tokens = token_ids.shape[-1]
            writer.writerow([input_tokens, output_tokens, *get_memory_usage()])

if __name__ == "__main__":
    print(f"os.getpid()={os.getpid()}")
    print(f"ov.get_version()={ov.get_version()}")
    main()
    release_manager()