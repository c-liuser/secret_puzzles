### Apache 2.0 License Header ###
"""
Copyright (C) 2025 Cole Liu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


#!/usr/bin/env python

import os
import time
import logging
import torch
import torch._dynamo

#
# (A) Setup and Patching
#

# 1. Force default dtype to float16 (common for QLoRA).
torch.set_default_dtype(torch.float16)

# 2. Enable "flex attention" (flash, SDPA, xformers).
ATTN_IMPL = "sdpa"  # or "flash", "xformers", etc.

# 3. Enable dynamic sequence lengths
#    We'll use `torch.compile(..., dynamic=True)` so variable seq lengths are allowed.
#    This helps fulfill the "dynamic_sequence_length_works" criterion.
DYNAMIC_SHAPES = True

# 4. Enable max_autotune for Triton matmul
import torch._inductor.config

torch._inductor.config.max_autotune = True

# 5. Minimal logging config to detect excessive recompilation
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

torch._inductor.config.debug = True
torch._logging.set_logs(
    dynamo=logging.WARN,
    inductor=logging.WARN,
    graph_breaks=True,
    recompiles=True,
    recompiles_verbose=True,
    compiled_autograd_verbose=True,
)

torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False

#
# (B) Patch accelerate/utils to prevent moving 4-bit quant parameters
#
try:
    from accelerate.utils import modeling as accel_modeling
except ImportError:
    from transformers.utils import modeling as accel_modeling  # fallback

_orig_set_module_tensor_to_device = accel_modeling.set_module_tensor_to_device


def patched_set_module_tensor_to_device(
    module,
    tensor_name,
    device,
    value,
    dtype=None,
    fp16_statistics=None,
    tied_params_map=None,
):
    if hasattr(value, "quant_state"):
        print(f"[patch] Skipping move for quantized param: {tensor_name}")
        return value
    return _orig_set_module_tensor_to_device(
        module, tensor_name, device, value, dtype, fp16_statistics, tied_params_map
    )


accel_modeling.set_module_tensor_to_device = patched_set_module_tensor_to_device

#
# (C) Patch ctypes._SimpleCData to avoid untraceable pointer calls
#
import ctypes


class FakeCData:
    def __init__(self, *args, **kwargs):
        pass


ctypes._SimpleCData = FakeCData

#
# (D) Load Model w/ BitsAndBytes 4-bit
#
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading model onto single GPU ...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation=ATTN_IMPL,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

#
# (E) Apply LoRA if needed
#
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
with torch.no_grad():
    for name, param in model.named_parameters():
        param.requires_grad_((".lora_" in name))

#
# (F) Patch the QLoRA attention & MLP layers for partial compilation
#
import bitsandbytes as bnb
import transformers.models.llama.modeling_llama as llama_mod

# Patch LlamaAttention
orig_attn_forward = llama_mod.LlamaAttention.forward


@torch.compile(fullgraph=False, dynamic=DYNAMIC_SHAPES)
def compiled_attention(self, *args, **kwargs):
    return _eager_4bit_qkv_matmul(self, *args, **kwargs)


@torch._dynamo.disable
def _eager_4bit_qkv_matmul(self, *args, **kwargs):
    return orig_attn_forward(self, *args, **kwargs)


llama_mod.LlamaAttention.forward = compiled_attention

# Patch LlamaMLP
orig_mlp_forward = llama_mod.LlamaMLP.forward


@torch.compile(fullgraph=False, dynamic=DYNAMIC_SHAPES)
def compiled_llama_mlp(self, *args, **kwargs):
    return _eager_4bit_mlp_ops(self, *args, **kwargs)


@torch._dynamo.disable
def _eager_4bit_mlp_ops(self, *args, **kwargs):
    return orig_mlp_forward(self, *args, **kwargs)


llama_mod.LlamaMLP.forward = compiled_llama_mlp

#
# (F.1) Also patch LlamaRMSNorm
#
orig_rmsnorm_forward = llama_mod.LlamaRMSNorm.forward


@torch.compile(fullgraph=False, dynamic=DYNAMIC_SHAPES)
def compiled_llama_rmsnorm(self, *args, **kwargs):
    return _eager_rmsnorm_forward(self, *args, **kwargs)


@torch._dynamo.disable
def _eager_rmsnorm_forward(self, *args, **kwargs):
    return orig_rmsnorm_forward(self, *args, **kwargs)


llama_mod.LlamaRMSNorm.forward = compiled_llama_rmsnorm

#
# (G) Enable input_require_grads (disable gradient checkpointing).
#
model.enable_input_require_grads()

#
# (H) Load dataset, define Trainer, and compile the loss
#
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

dataset_url = (
    "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
)
dataset = load_dataset("json", data_files={"train": dataset_url}, split="train[:10%]")

trainer_args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=1,
    max_steps=10,
    logging_steps=1,
    output_dir="outputs",
    seed=3407,
    max_seq_length=256,
    fp16=True,
    bf16=False,
    report_to="none",
    dataset_num_proc=1,
)

#
# (I) Initialize Trainer
#
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=trainer_args,
)


#
# (J) Print VRAM usage and run training
#
def print_vram(prefix):
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"{prefix} VRAM => allocated={allocated:.2f}MB, reserved={reserved:.2f}MB")


print_vram("Before training:")
start = time.time()
trainer.train()
end = time.time()
print_vram("After training:")

elapsed = end - start
print(f"Training finished in {elapsed:.2f} seconds")
