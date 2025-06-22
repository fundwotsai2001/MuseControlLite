# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils import deprecate, logging
from safetensors.torch import load_file
from diffusers.loaders import AttnProcsLayers
from utils.extract_conditions import compute_melody, compute_melody_v2, compute_dynamics, extract_melody_one_hot, evaluate_f1_rhythm
from madmom.features.downbeats import DBNDownBeatTrackingProcessor,RNNDownBeatProcessor
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.stable_audio_dataset_utils import load_audio_file
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
import soundfile as sf

# For zero initialized 1D CNN in the attention processor
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

# Original attention processor for 
class StableAudioAttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """

    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
    def apply_partial_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        rot_dim = freqs_cis[0].shape[-1]
        x_to_rotate, x_unrotated = x[..., :rot_dim], x[..., rot_dim:]

        x_rotated = apply_rotary_emb(x_to_rotate, freqs_cis, use_real=True, use_real_unbind_dim=-2)

        out = torch.cat((x_rotated, x_unrotated), dim=-1)
        return out

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed 
        if rotary_emb is not None:
            query_dtype = query.dtype
            key_dtype = key.dtype
            query = query.to(torch.float32)
            key = key.to(torch.float32)

            rot_dim = rotary_emb[0].shape[-1]
            query_to_rotate, query_unrotated = query[..., :rot_dim], query[..., rot_dim:]
            query_rotated = apply_rotary_emb(query_to_rotate, rotary_emb, use_real=True, use_real_unbind_dim=-2)

            query = torch.cat((query_rotated, query_unrotated), dim=-1)

            if not attn.is_cross_attention:
                key_to_rotate, key_unrotated = key[..., :rot_dim], key[..., rot_dim:]
                key_rotated = apply_rotary_emb(key_to_rotate, rotary_emb, use_real=True, use_real_unbind_dim=-2)

                key = torch.cat((key_rotated, key_unrotated), dim=-1)

            query = query.to(query_dtype)
            key = key.to(key_dtype)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # print("hidden_states", hidden_states.shape)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
# The attention processor used in MuseControlLite, using 1 decoupled cross-attention layer
class StableAudioAttnProcessor2_0_rotary(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """
    def __init__(self, layer_id, hidden_size, name, cross_attention_dim=None, num_tokens=4, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.name = name
        self.conv_out = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))     
        self.rotary_emb = LlamaRotaryEmbedding(dim = 64)
        self.to_k_ip.weight.requires_grad = True
        self.to_v_ip.weight.requires_grad = True
        self.conv_out.weight.requires_grad = True
    def rotate_half(self, x):
        x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1, x2 = x.unbind(-1)
        return torch.cat((-x2, x1), dim=-1)


    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_con: Optional[torch.Tensor] = None,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # The original cross attention in Stable-audio
        ###############################################################
        query = attn.to_q(hidden_states)
        ip_hidden_states = encoder_hidden_states_con
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        ###############################################################


        # The decupled cross attention in used in MuseControlLite, to deal with additional conditions
        ###############################################################
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        ip_key = ip_key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        ip_key_length = ip_key.shape[2]
        ip_value = ip_value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            ip_key = torch.repeat_interleave(ip_key, heads_per_kv_head, dim=1)
            ip_value = torch.repeat_interleave(ip_value, heads_per_kv_head, dim=1)
        ip_value_length = ip_value.shape[2]
        seq_len_query = query.shape[2]

        # Generate position_ids for query, keys, values
        position_ids_query = torch.arange(seq_len_query, dtype=torch.long, device=query.device) * (ip_key_length / seq_len_query)
        position_ids_query = position_ids_query.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_query]
        position_ids_key = torch.arange(ip_key_length, dtype=torch.long, device=key.device)
        position_ids_key = position_ids_key.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        position_ids_value = torch.arange(ip_value_length, dtype=torch.long, device=value.device)
        position_ids_value = position_ids_value.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        
        # Rotate query, keys, values 
        cos, sin = self.rotary_emb(query, position_ids_query)
        query_pos = (query * cos.unsqueeze(1)) + (self.rotate_half(query) * sin.unsqueeze(1))
        cos, sin = self.rotary_emb(ip_key, position_ids_key)
        ip_key = (ip_key * cos.unsqueeze(1)) + (self.rotate_half(ip_key) * sin.unsqueeze(1))
        cos, sin = self.rotary_emb(ip_value, position_ids_value)
        ip_value = (ip_value * cos.unsqueeze(1)) + (self.rotate_half(ip_value) * sin.unsqueeze(1))
        
        ip_hidden_states = F.scaled_dot_product_attention(
                query_pos, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)
        ip_hidden_states = self.conv_out(ip_hidden_states)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)
        ###############################################################

        # Combine the output of the two cross-attention layers
        hidden_states = hidden_states + self.scale * ip_hidden_states
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
# The attention processor used in MuseControlLite, using 2 decoupled cross-attention layer. It needs further examination, don't use it now.
class StableAudioAttnProcessor2_0_rotary_double(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """
    def __init__(self, layer_id, hidden_size, name, cross_attention_dim=None, num_tokens=4, scale=1.0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.layer_id = layer_id
        self.scale = scale
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_k_ip_audio = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip_audio = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.name = name
        self.conv_out = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))
        self.conv_out_audio = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))
        self.rotary_emb = LlamaRotaryEmbedding(64)
        self.to_k_ip.weight.requires_grad = True
        self.to_v_ip.weight.requires_grad = True
        self.conv_out.weight.requires_grad = True
        # Below is for copying the weight of the original weight to the decoupled cross-attention
    def rotate_half(self, x):
        x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1, x2 = x.unbind(-1)
        return torch.cat((-x2, x1), dim=-1)


    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_con: Optional[torch.Tensor] = None,
        encoder_hidden_states_audio: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # The original cross attention in Stable-audio
        ###############################################################
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        # if self.layer_id == "0":
        #     hidden_states_sliced = hidden_states[:,1:,:]
        #     # Create a tensor of zeros with shape (bs, 1, 768)
        #     bs, _, dim2 = hidden_states_sliced.shape
        #     zeros = torch.zeros(bs, 1, dim2).cuda()
        #     # Concatenate the zero tensor along the second dimension (dim=1)
        #     hidden_states_sliced = torch.cat((hidden_states_sliced, zeros), dim=1)
        #     query_sliced = attn.to_q(hidden_states_sliced)
        #     query_sliced = query_sliced.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #     query = query_sliced
        ip_hidden_states = encoder_hidden_states_con
        ip_hidden_states_audio = encoder_hidden_states_audio
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        ip_key = ip_key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        ip_key_length = ip_key.shape[2]
        ip_value = ip_value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        ip_key_audio = self.to_k_ip_audio(ip_hidden_states_audio)
        ip_value_audio = self.to_v_ip_audio(ip_hidden_states_audio)
        ip_key_audio = ip_key_audio.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        ip_key_audio_length = ip_key_audio.shape[2]
        ip_value_audio = ip_value_audio.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            ip_key = torch.repeat_interleave(ip_key, heads_per_kv_head, dim=1)
            ip_value = torch.repeat_interleave(ip_value, heads_per_kv_head, dim=1)
            ip_key_audio = torch.repeat_interleave(ip_key_audio, heads_per_kv_head, dim=1)
            ip_value_audio = torch.repeat_interleave(ip_value_audio, heads_per_kv_head, dim=1)
            
        ip_value_length = ip_value.shape[2]
        seq_len_query = query.shape[2]
        ip_value_audio_length = ip_value_audio.shape[2]

        position_ids_query = torch.arange(seq_len_query, dtype=torch.long, device=query.device) * (ip_key_length / seq_len_query)
        position_ids_query = position_ids_query.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_query]
        
        # Generate position_ids for keys
        position_ids_key = torch.arange(ip_key_length, dtype=torch.long, device=key.device)
        position_ids_key = position_ids_key.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        position_ids_value = torch.arange(ip_value_length, dtype=torch.long, device=value.device)
        position_ids_value = position_ids_value.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        # Generate position_ids for keys
        position_ids_query_audio = torch.arange(seq_len_query, dtype=torch.long, device=query.device) * (ip_key_audio_length / seq_len_query)
        position_ids_query_audio = position_ids_query_audio.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_query]
        position_ids_key_audio = torch.arange(ip_key_audio_length, dtype=torch.long, device=key.device)
        position_ids_key_audio = position_ids_key_audio.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        position_ids_value_audio = torch.arange(ip_value_audio_length, dtype=torch.long, device=value.device)
        position_ids_value_audio = position_ids_value_audio.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        cos, sin = self.rotary_emb(query, position_ids_query)
        cos_audio, sin_audio = self.rotary_emb(query, position_ids_query_audio)
        query_pos = (query * cos.unsqueeze(1)) + (self.rotate_half(query) * sin.unsqueeze(1))
        query_pos_audio = (query * cos_audio.unsqueeze(1)) + (self.rotate_half(query) * sin_audio.unsqueeze(1))

        cos, sin = self.rotary_emb(ip_key, position_ids_key)
        cos_audio, sin_audio = self.rotary_emb(ip_key_audio, position_ids_key_audio)
        ip_key = (ip_key * cos.unsqueeze(1)) + (self.rotate_half(ip_key) * sin.unsqueeze(1))
        ip_key_audio = (ip_key_audio * cos_audio.unsqueeze(1)) + (self.rotate_half(ip_key_audio) * sin_audio.unsqueeze(1)) 

        cos, sin = self.rotary_emb(ip_value, position_ids_value)
        cos_audio, sin_audio = self.rotary_emb(ip_value_audio, position_ids_value_audio)
        ip_value = (ip_value * cos.unsqueeze(1)) + (self.rotate_half(ip_value) * sin.unsqueeze(1))
        ip_value_audio = (ip_value_audio * cos_audio.unsqueeze(1)) + (self.rotate_half(ip_value_audio) * sin_audio.unsqueeze(1))   

        with torch.amp.autocast(device_type='cuda'):
            ip_hidden_states = F.scaled_dot_product_attention(
                    query_pos, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
        with torch.amp.autocast(device_type='cuda'):
            ip_hidden_states_audio = F.scaled_dot_product_attention(
                    query_pos_audio, ip_key_audio, ip_value_audio, attn_mask=None, dropout_p=0.0, is_causal=False
                )
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)

        ip_hidden_states_audio = ip_hidden_states_audio.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states_audio = ip_hidden_states_audio.to(query.dtype)
        ip_hidden_states_audio = ip_hidden_states_audio.transpose(1, 2)

        with torch.amp.autocast(device_type='cuda'):
            ip_hidden_states = self.conv_out(ip_hidden_states)
        ip_hidden_states = ip_hidden_states.transpose(1, 2)
        
        with torch.amp.autocast(device_type='cuda'):
            ip_hidden_states_audio = self.conv_out_audio(ip_hidden_states_audio)
        ip_hidden_states_audio = ip_hidden_states_audio.transpose(1, 2)

        # Combine the tensors
        hidden_states = hidden_states + self.scale * ip_hidden_states  + ip_hidden_states_audio
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
def setup_MuseControlLite(config, weight_dtype, transformer_ckpt):
    """
    Setup AP-adapter pipeline with attention processors and load checkpoints.
    
    Args:
        config: Configuration dictionary
        weight_dtype: Weight data type for the pipeline
        transformer_ckpt: Path to transformer checkpoint    
    Returns:
        tuple: (pipe, transformer) - Configured pipeline and transformer
    """
    if 'audio' in config['condition_type'] and len(config['condition_type'])!=1:
        from pipeline.stable_audio_multi_cfg_pipe_audio import StableAudioPipeline
        attn_processor = StableAudioAttnProcessor2_0_rotary_double
        audio_state_dict = load_file(config["audio_transformer_ckpt"], device="cpu")
        config['no_text'] = True
    else:
        from pipeline.stable_audio_multi_cfg_pipe import StableAudioPipeline
        attn_processor = StableAudioAttnProcessor2_0_rotary
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=weight_dtype)
    pipe.scheduler.config.sigma_max = config["sigma_max"]
    pipe.scheduler.config.sigma_min = config["sigma_min"]
    transformer = pipe.transformer
    attn_procs = {}
    for name in transformer.attn_processors.keys():
        if name.endswith("attn1.processor"):
            attn_procs[name] = StableAudioAttnProcessor2_0()
        else:
            attn_procs[name] = attn_processor(
                layer_id = name.split(".")[1],
                hidden_size=768,
                name=name,
                cross_attention_dim=768,
                scale=config['ap_scale'],
            ).to("cuda", dtype=torch.float)            
    if transformer_ckpt is not None:
        state_dict = load_file(transformer_ckpt, device="cuda")
        for name, processor in attn_procs.items():
            if isinstance(processor, attn_processor):
                weight_name_v = name + ".to_v_ip.weight"
                weight_name_k = name + ".to_k_ip.weight"
                conv_out_weight = name + ".conv_out.weight"
                processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].to(torch.float32))
                processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].to(torch.float32))
                processor.conv_out.weight = torch.nn.Parameter(state_dict[conv_out_weight].to(torch.float32))
                if attn_processor == StableAudioAttnProcessor2_0_rotary_double:
                    audio_weight_name_v = name + ".to_v_ip.weight"
                    audio_weight_name_k = name + ".to_k_ip.weight"
                    audio_conv_out_weight = name + ".conv_out.weight"
                    processor.to_v_ip_audio.weight = torch.nn.Parameter(audio_state_dict[audio_weight_name_v].to(torch.float32))
                    processor.to_k_ip_audio.weight = torch.nn.Parameter(audio_state_dict[audio_weight_name_k].to(torch.float32))
                    processor.conv_out_audio.weight = torch.nn.Parameter(audio_state_dict[audio_conv_out_weight].to(torch.float32))
    transformer.set_attn_processor(attn_procs)
    class _Wrapper(AttnProcsLayers):
        def forward(self, *args, **kwargs):
            return pipe.transformer(*args, **kwargs)
    transformer = _Wrapper(pipe.transformer.attn_processors)
    
    return pipe
def initialize_condition_extractors(config):
    """
    Initialize condition extractors based on configuration.
    
    Args:
        config: Configuration dictionary containing condition types and checkpoint paths
        
    Returns:
        tuple: (condition_extractors, transformer_ckpt, extractor_ckpt)
    """
    condition_extractors = {}
    extractor_ckpt = {}
    from utils.feature_extractor import dynamics_extractor, rhythm_extractor, melody_extractor_mono, melody_extractor_stereo, melody_extractor_full_mono, melody_extractor_full_stereo, dynamics_extractor_full_stereo
    if not ("rhythm" in config['condition_type'] or "dynamics" in config['condition_type']):
        if "melody_stereo" in config['condition_type']:
            transformer_ckpt = config['transformer_ckpt_melody_stero']
            extractor_ckpt = config['extractor_ckpt_melody_stero']
            print(f"using model: {transformer_ckpt}, {extractor_ckpt}")
            melody_conditoner = melody_extractor_full_stereo().cuda().float()
            condition_extractors["melody"] = melody_conditoner
        elif "melody_mono" in config['condition_type']:
            transformer_ckpt = config['transformer_ckpt_melody_mono']
            extractor_ckpt = config['extractor_ckpt_melody_mono']
            print(f"using model: {transformer_ckpt}, {extractor_ckpt}")
            melody_conditoner = melody_extractor_full_mono().cuda().float()
            condition_extractors["melody"] = melody_conditoner
        elif "audio" in config['condition_type']:
            transformer_ckpt = config['audio_transformer_ckpt']
            print(f"using model: {transformer_ckpt}")
    else:
        dynamics_conditoner = dynamics_extractor().cuda().float()
        condition_extractors["dynamics"] = dynamics_conditoner
        rhythm_conditoner = rhythm_extractor().cuda().float()
        condition_extractors["rhythm"] = rhythm_conditoner
        melody_conditoner = melody_extractor_mono().cuda().float()
        condition_extractors["melody"] = melody_conditoner
        transformer_ckpt = config['transformer_ckpt_musical']
        extractor_ckpt = config['extractor_ckpt_musical']
        print(f"using model: {transformer_ckpt}, {extractor_ckpt}")
    
    for conditioner_type, ckpt_path in extractor_ckpt.items(): 
        state_dict = load_file(ckpt_path, device="cpu")
        condition_extractors[conditioner_type].load_state_dict(state_dict)
        condition_extractors[conditioner_type].eval()
    
    return condition_extractors, transformer_ckpt
def evaluate_and_plot_results(audio_file, gen_file_path, output_dir, i):
    """
    Evaluate and plot results comparing original and generated audio.
    
    Args:
        audio_file (str): Path to the original audio file
        gen_file_path (str): Path to the generated audio file
        output_dir (str): Directory to save the plot
        i (int): Index for naming the output file
    
    Returns:
        tuple: (dynamics_score, rhythm_score, melody_score)
    """

    dynamics_condition = compute_dynamics(audio_file)
    gen_dynamics = compute_dynamics(gen_file_path)
    min_len_dynamics = min(gen_dynamics.shape[0], dynamics_condition.shape[0])
    pearson_corr = np.corrcoef(gen_dynamics[:min_len_dynamics], dynamics_condition[:min_len_dynamics])[0, 1]
    print("pearson_corr", pearson_corr)
    
    melody_condition = extract_melody_one_hot(audio_file)      
    gen_melody = extract_melody_one_hot(gen_file_path)
    min_len_melody = min(gen_melody.shape[1], melody_condition.shape[1])
    matches = ((gen_melody[:, :min_len_melody] == melody_condition[:, :min_len_melody]) & (gen_melody[:, :min_len_melody] == 1)).sum()
    accuracy = matches / min_len_melody
    print("melody accuracy", accuracy)
    
    # Adjust layout to avoid overlap
    processor = RNNDownBeatProcessor()
    input_probabilities = processor(audio_file)
    generated_probabilities = processor(gen_file_path)
    hmm_processor = DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=100)
    input_timestamps = hmm_processor(input_probabilities)
    generated_timestamps = hmm_processor(generated_probabilities)
    precision, recall, f1 = evaluate_f1_rhythm(input_timestamps, generated_timestamps)
    # Output results
    print(f"F1 Score: {f1:.2f}")
    
    # Plotting
    frame_rate = 100  # Frames per second
    input_time_axis = np.linspace(0, len(input_probabilities) / frame_rate, len(input_probabilities))
    generate_time_axis = np.linspace(0, len(generated_probabilities) / frame_rate, len(generated_probabilities))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Adjust figsize as needed

    # ----------------------------
    # Subplot (0,0): Dynamics Plot
    ax = axes[0, 0]
    ax.plot(dynamics_condition[:min_len_dynamics].squeeze(), linewidth=1, label='Dynamics condition')
    ax.set_title('Dynamics')
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Dynamics (dB)')
    ax.legend(fontsize=8)
    ax.grid(True)
    # ----------------------------
    # Subplot (0,0): Dynamics Plot
    ax = axes[1, 0]
    ax.plot(gen_dynamics[:min_len_dynamics].squeeze(), linewidth=1, label='Generated Dynamics')
    ax.set_title('Dynamics')
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Dynamics (dB)')
    ax.legend(fontsize=8)
    ax.grid(True)

    # ----------------------------
    # Subplot (0,2): Melody Condition (Chromagram)
    ax = axes[0, 1]
    im2 = ax.imshow(melody_condition[:, :min_len_melody], aspect='auto', origin='lower',
                    interpolation='nearest', cmap='plasma')
    ax.set_title('Melody Condition')
    ax.set_xlabel('Time')
    ax.set_ylabel('Chroma Features')

    # ----------------------------
    # Subplot (0,1): Generated Melody (Chromagram)
    ax = axes[1, 1]
    im1 = ax.imshow(gen_melody[:, :min_len_melody], aspect='auto', origin='lower',
                    interpolation='nearest', cmap='viridis')
    ax.set_title('Generated Melody')
    ax.set_xlabel('Time')
    ax.set_ylabel('Chroma Features')

    # ----------------------------
    # Subplot (1,0): Rhythm Input Probabilities
    ax = axes[0, 2]
    ax.plot(input_time_axis, input_probabilities,
            label="Input Beat Probability")
    ax.plot(input_time_axis, input_probabilities,
            label="Input Downbeat Probability", alpha=0.8)
    ax.set_title('Rhythm: Input')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(True)

    # ----------------------------
    # Subplot (1,1): Rhythm Generated Probabilities
    ax = axes[1, 2]
    ax.plot(generate_time_axis, generated_probabilities,
            color='orange', label="Generated Beat Probability")
    ax.plot(generate_time_axis, generated_probabilities,
            alpha=0.8, color='red', label="Generated Downbeat Probability")
    ax.set_title('Rhythm: Generated')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(True)

    # Adjust layout and save the combined image
    plt.tight_layout()
    combined_path = os.path.join(output_dir, f"combined_{i}.png")
    plt.savefig(combined_path)
    plt.close()

    print(f"Combined plot saved to {combined_path}")
    
    return pearson_corr, f1, accuracy

def process_musical_conditions(config, audio_file, condition_extractors, output_dir, i, weight_dtype, MuseControlLite):
    """
    Process and extract musical conditions (dynamics, rhythm, melody) from audio file.
    
    Args:
        config: Configuration dictionary
        audio_file: Path to the audio file
        condition_extractors: Dictionary of condition extractors
        output_dir: Output directory path
        i: Index for file naming
        weight_dtype: Weight data type for torch tensors
        MuseControlLite: The MuseControlLite model instance
        audio_mask_start: Start index for audio mask
        audio_mask_end: End index for audio mask
        musical_attribute_mask_start: Start index for musical attribute mask
        musical_attribute_mask_end: End index for musical attribute mask
    
    Returns:
        tuple: (final_condition, extracted_condition, final_condition_audio)
    """
    total_seconds = 2097152/44100
    audio_mask_start = int(config["audio_mask_start_seconds"] / total_seconds * 1024) # 1024 is the latent length for 2097152/44100 seconds
    audio_mask_end = int(config["audio_mask_end_seconds"] / total_seconds * 1024)
    musical_attribute_mask_start = int(config["musical_attribute_mask_start_seconds"] / total_seconds * 1024)
    musical_attribute_mask_end = int(config["musical_attribute_mask_end_seconds"] / total_seconds * 1024)
    if "dynamics" in config["condition_type"]:
        dynamics_condition = compute_dynamics(audio_file)
        dynamics_condition = torch.from_numpy(dynamics_condition).cuda()
        dynamics_condition = dynamics_condition.unsqueeze(0).unsqueeze(0)
        print("dynamics_condition", dynamics_condition.shape)
        extracted_dynamics_condition = condition_extractors["dynamics"](dynamics_condition.to(torch.float32))
        masked_extracted_dynamics_condition =  torch.zeros_like(extracted_dynamics_condition)
        extracted_dynamics_condition = F.interpolate(extracted_dynamics_condition, size=1024, mode='linear', align_corners=False) 
        masked_extracted_dynamics_condition = F.interpolate(masked_extracted_dynamics_condition, size=1024, mode='linear', align_corners=False)
    else: 
        extracted_dynamics_condition = torch.zeros((1, 192, 1024), device="cuda")
        masked_extracted_dynamics_condition = extracted_dynamics_condition
    if "rhythm" in config["condition_type"]:
        rnn_processor = RNNDownBeatProcessor()
        wave = load_audio_file(audio_file)
        if wave is not None:
            original_path = os.path.join(output_dir, f"original_{i}.wav")
            sf.write(original_path, wave.T.float().cpu().numpy(), 44100)
            rhythm_curve = rnn_processor(original_path)
            rhythm_condition = torch.from_numpy(rhythm_curve).cuda()
            rhythm_condition = rhythm_condition.transpose(0,1).unsqueeze(0)
            extracted_rhythm_condition = condition_extractors["rhythm"](rhythm_condition.to(torch.float32))
            masked_extracted_rhythm_condition = torch.zeros_like(extracted_rhythm_condition)
            extracted_rhythm_condition = F.interpolate(extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)
            masked_extracted_rhythm_condition = F.interpolate(masked_extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)      
        else:
            extracted_rhythm_condition = torch.zeros((1, 192, 1024), device="cuda")
            masked_extracted_rhythm_condition = extracted_rhythm_condition
    else: 
        extracted_rhythm_condition = torch.zeros((1, 192, 1024), device="cuda")
        masked_extracted_rhythm_condition = extracted_rhythm_condition
    
    if "melody_mono" in config["condition_type"]:
        melody_condition = compute_melody(audio_file)
        melody_condition = torch.from_numpy(melody_condition).cuda().unsqueeze(0)
        print("melody_condition", melody_condition.shape)
        extracted_melody_condition = condition_extractors["melody"](melody_condition.to(torch.float32))
        masked_extracted_melody_condition = torch.zeros_like(extracted_melody_condition)
        extracted_melody_condition = F.interpolate(extracted_melody_condition, size=1024, mode='linear', align_corners=False)
        masked_extracted_melody_condition = F.interpolate(masked_extracted_melody_condition, size=1024, mode='linear', align_corners=False)
    elif "melody_stereo" in config["condition_type"]:
        melody_condition = compute_melody_v2(audio_file)
        melody_condition = torch.from_numpy(melody_condition).cuda().unsqueeze(0)
        print("melody_condition", melody_condition.shape)
        extracted_melody_condition = condition_extractors["melody"](melody_condition)
        masked_extracted_melody_condition = torch.zeros_like(extracted_melody_condition)
        extracted_melody_condition = F.interpolate(extracted_melody_condition, size=1024, mode='linear', align_corners=False)
        masked_extracted_melody_condition = F.interpolate(masked_extracted_melody_condition, size=1024, mode='linear', align_corners=False)
    else: 
        if not ("rhythm" in config['condition_type'] or "dynamics" in config['condition_type']):
            extracted_melody_condition = torch.zeros((1, 768, 1024), device="cuda")
        else:
            extracted_melody_condition = torch.zeros((1, 192, 1024), device="cuda")
        masked_extracted_melody_condition = extracted_melody_condition

    # Use multiple cfg
    if not ("rhythm" in config['condition_type'] or "dynamics" in config['condition_type']):
        extracted_condition = extracted_melody_condition
        final_condition = torch.concat((masked_extracted_melody_condition, masked_extracted_melody_condition, extracted_melody_condition), dim=0)
    else:
        extracted_blank_condition = torch.zeros((1, 192, 1024), device="cuda")
        extracted_condition = torch.concat((extracted_rhythm_condition, extracted_dynamics_condition, extracted_melody_condition, extracted_blank_condition), dim=1)
        masked_extracted_condition = torch.concat((masked_extracted_rhythm_condition, masked_extracted_dynamics_condition, masked_extracted_melody_condition, extracted_blank_condition), dim=1)
        final_condition = torch.concat((masked_extracted_condition, masked_extracted_condition, extracted_condition), dim=0)
    if "audio" in config["condition_type"]:
        desired_repeats = 768 // 64  # Number of repeats needed
        audio = load_audio_file(audio_file)
        if audio is not None:
            audio_condition = MuseControlLite.vae.encode(audio.unsqueeze(0).to(weight_dtype).cuda()).latent_dist.sample()
            extracted_audio_condition = audio_condition.repeat_interleave(desired_repeats, dim=1).float()
            masked_extracted_audio_condition = torch.zeros_like(extracted_audio_condition)
            if len(config["condition_type"]) == 1:
                final_condition = torch.concat((masked_extracted_audio_condition, masked_extracted_audio_condition, extracted_audio_condition), dim=0)
            else:
                final_condition_audio = torch.concat((masked_extracted_audio_condition, masked_extracted_audio_condition, masked_extracted_audio_condition, extracted_audio_condition), dim=0)
                final_condition = torch.concat((final_condition, extracted_condition), dim=0)
                final_condition_audio = final_condition_audio.transpose(1, 2)
        else:
            final_condition_audio = None
    final_condition = final_condition.transpose(1, 2)
    if "audio" in config["condition_type"] and len(config["condition_type"])==1 and config['use_audio_mask']:
        final_condition[:,audio_mask_start:audio_mask_end,:] = 0
        config["guidance_scale_con"] = config["guidance_scale_audio"]
    elif "audio" in config["condition_type"] and len(config["condition_type"])!=1:
        final_condition[:,:audio_mask_start,:] = 0
        final_condition[:,audio_mask_end:,:] = 0
        if 'final_condition_audio' in locals() and final_condition_audio is not None:
            final_condition_audio[:,audio_mask_start:audio_mask_end,:] = 0
    elif config['use_musical_attribute_mask']:
        final_condition[:,musical_attribute_mask_start:musical_attribute_mask_end,:] = 0
    
    return final_condition, final_condition_audio if 'final_condition_audio' in locals() else None

