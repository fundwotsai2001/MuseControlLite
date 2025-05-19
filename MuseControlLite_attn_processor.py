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


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

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
class StableAudioAttnProcessor2_0_no_rotary(torch.nn.Module):
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

        # # Generate position_ids for query, keys, values
        # position_ids_query = torch.arange(seq_len_query, dtype=torch.long, device=query.device) * (ip_key_length / seq_len_query)
        # position_ids_query = position_ids_query.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_query]
        # position_ids_key = torch.arange(ip_key_length, dtype=torch.long, device=key.device)
        # position_ids_key = position_ids_key.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        # position_ids_value = torch.arange(ip_value_length, dtype=torch.long, device=value.device)
        # position_ids_value = position_ids_value.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, seq_len_key]
        
        # # Rotate query, keys, values 
        # cos, sin = self.rotary_emb(query, position_ids_query)
        # query_pos = (query * cos.unsqueeze(1)) + (self.rotate_half(query) * sin.unsqueeze(1))
        # cos, sin = self.rotary_emb(ip_key, position_ids_key)
        # ip_key = (ip_key * cos.unsqueeze(1)) + (self.rotate_half(ip_key) * sin.unsqueeze(1))
        # cos, sin = self.rotary_emb(ip_value, position_ids_value)
        # ip_value = (ip_value * cos.unsqueeze(1)) + (self.rotate_half(ip_value) * sin.unsqueeze(1))
        
        ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
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
class StableAudioAttnProcessor2_0_rotary_no_cnn(torch.nn.Module):
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
        # self.conv_out = zero_module(nn.Conv1d(1536,1536,kernel_size=1, padding=0, bias=False))     
        self.rotary_emb = LlamaRotaryEmbedding(dim = 64)
        self.to_k_ip.weight.requires_grad = True
        self.to_v_ip.weight.requires_grad = True
        # self.conv_out.weight.requires_grad = True
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
        # ip_hidden_states = ip_hidden_states.transpose(1, 2)
        # ip_hidden_states = self.conv_out(ip_hidden_states)
        # ip_hidden_states = ip_hidden_states.transpose(1, 2)
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
class StableAudioAttnProcessor2_0_rotary_scale_up(torch.nn.Module):
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
        ip_kv_heads = ip_key.shape[-1] // head_dim
        ip_key = ip_key.view(batch_size, -1, ip_kv_heads, head_dim).transpose(1, 2)
        ip_key_length = ip_key.shape[2]
        ip_value = ip_value.view(batch_size, -1, ip_kv_heads, head_dim).transpose(1, 2)
        # print("ip_kv_heads", ip_kv_heads)
        # print("attn.heads", attn.heads)
        # print("ip_key", ip_key.shape)
        # print("ip_value", ip_value.shape)
        # print("query", query.shape)
        if ip_kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // ip_kv_heads
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