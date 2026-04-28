# coding=utf-8
# Copyright 2026 The HunYuan team & SGLang team.
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

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_world_size,
    get_tensor_model_parallel_world_size,
    moe_expert_parallel_all_reduce,
    moe_tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import should_skip_post_experts_all_reduce
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import ForwardBatch
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import is_cuda, is_npu
from sglang.srt.utils.hf_transformers_utils import get_rope_config


def _get_device_stream():
    if is_npu():
        return torch.npu.Stream()
    if is_cuda():
        return torch.cuda.Stream()
    return None


def _get_current_stream():
    if is_npu():
        return torch.npu.current_stream()
    return torch.cuda.current_stream()


class HYImage3FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        return out


class HYImage3MoEFused(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream=None,
    ):
        super().__init__()
        self.tp_size = get_moe_tensor_parallel_world_size()
        self.ep_size = get_moe_expert_parallel_world_size()
        self.layer_id = layer_id
        self.alt_stream = alt_stream

        num_experts = config.num_experts
        if isinstance(num_experts, list):
            num_experts = num_experts[layer_id]
        self.n_routed_experts = num_experts

        top_k = config.moe_topk
        if isinstance(top_k, list):
            top_k = top_k[layer_id]

        intermediate_size = config.moe_intermediate_size
        if isinstance(intermediate_size, list):
            intermediate_size = intermediate_size[layer_id]

        num_shared_expert = getattr(config, "num_shared_expert", 1)
        if isinstance(num_shared_expert, list):
            num_shared_expert = num_shared_expert[layer_id]

        router_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        if isinstance(router_scaling_factor, bool):
            router_scaling_factor = 1.0

        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=False,
            quant_config=None,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.topk = TopK(
            top_k=top_k,
            use_grouped_topk=False,
            num_expert_group=1,
            topk_group=1,
            renormalize=getattr(config, "norm_topk_prob", True),
        )

        if num_shared_expert > 0:
            self.shared_mlp = HYImage3FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size * num_shared_expert,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_mlp",
                reduce_results=False,
            )
        else:
            self.shared_mlp = None

        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=top_k,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=False,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self.alt_stream is not None
            and self.shared_mlp is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            return self._forward_dual_stream(hidden_states)
        return self._forward_single_stream(hidden_states)

    def _forward_single_stream(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))
        topk_output = self.topk(hidden_states, router_logits)
        if self.shared_mlp is not None:
            shared_output = self.shared_mlp(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, topk_output=topk_output
            )
            final_hidden_states = final_hidden_states + shared_output
        else:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, topk_output=topk_output
            )

        if self.ep_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=False,
        ):
            final_hidden_states = moe_expert_parallel_all_reduce(final_hidden_states)

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
        ):
            final_hidden_states = moe_tensor_model_parallel_all_reduce(
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)

    def _forward_dual_stream(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        current_stream = _get_current_stream()
        self.alt_stream.wait_stream(current_stream)

        shared_output = self.shared_mlp(hidden_states)

        if is_npu():
            with torch.npu.stream(self.alt_stream):
                router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))
                topk_output = self.topk(hidden_states, router_logits)
                final_hidden_states = self.experts(
                    hidden_states=hidden_states, topk_output=topk_output
                )
        else:
            with torch.cuda.stream(self.alt_stream):
                router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))
                topk_output = self.topk(hidden_states, router_logits)
                final_hidden_states = self.experts(
                    hidden_states=hidden_states, topk_output=topk_output
                )

        current_stream.wait_stream(self.alt_stream)
        final_hidden_states = final_hidden_states + shared_output

        if self.ep_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=False,
        ):
            final_hidden_states = moe_expert_parallel_all_reduce(final_hidden_states)

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
        ):
            final_hidden_states = moe_tensor_model_parallel_all_reduce(
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)


class HYImage3Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = getattr(config, "head_dim", hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.use_qk_norm = getattr(
            config, "use_qk_norm", getattr(config, "qk_norm", False)
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )
        if self.use_qk_norm:
            rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)
            self.q_norm = RMSNorm(self.head_dim, rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.use_qk_norm:
            q = self.q_norm(q.reshape(-1, self.head_dim))
            q = q.view(-1, self.q_size)
            k = self.k_norm(k.reshape(-1, self.head_dim))
            k = k.view(-1, self.kv_size)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class HYImage3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream=None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        rope_theta, _ = get_rope_config(config)
        self.self_attn = HYImage3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        moe_layer_num_skipped = getattr(config, "moe_layer_num_skipped", 0)
        num_experts = config.num_experts
        if isinstance(num_experts, list):
            has_experts = max(num_experts) > 1
        else:
            has_experts = num_experts > 1

        if has_experts and layer_id >= moe_layer_num_skipped:
            self.mlp = HYImage3MoEFused(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                alt_stream=alt_stream,
            )
            self.block_type = "moe"
        else:
            intermediate_size = config.intermediate_size
            if isinstance(intermediate_size, list):
                intermediate_size = intermediate_size[layer_id]
            self.mlp = HYImage3FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.block_type = "feedforward"

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class HYImage3Model(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.wte",
        )

        self.alt_stream = _get_device_stream()

        self.layers = nn.ModuleList(
            [
                HYImage3DecoderLayer(
                    config=config,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                    alt_stream=self.alt_stream,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class HunyuanImage3ForCausalMM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.model = HYImage3Model(config, quant_config, prefix=f"{prefix}.model")
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head",
        )
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        if is_npu():
            torch.npu.empty_cache()
            torch.npu.synchronize()
        elif is_cuda():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _rearrange_qkv(self, loaded_weight: torch.Tensor) -> torch.Tensor:
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = getattr(
            self.config, "head_dim", self.config.hidden_size // num_heads
        )
        num_q_per_kv = num_heads // num_kv_heads

        qkv = loaded_weight.reshape(
            num_kv_heads, num_q_per_kv + 2, head_dim, loaded_weight.shape[1]
        )
        q = qkv[:, :num_q_per_kv, :, :].reshape(num_heads * head_dim, -1)
        k = qkv[:, num_q_per_kv, :, :].reshape(num_kv_heads * head_dim, -1)
        v = qkv[:, num_q_per_kv + 1, :, :].reshape(num_kv_heads * head_dim, -1)
        return torch.cat([q, k, v], dim=0)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        num_experts = self.config.num_experts
        if isinstance(num_experts, list):
            num_experts = max(num_experts)

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
        )

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if "lm_head.weight" in name and getattr(
                self.config, "tie_word_embeddings", False
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            if (
                name.startswith("vae.")
                or name.startswith("vision_model.")
                or name.startswith("vision_aligner.")
                or name.startswith("timestep_emb.")
                or name.startswith("patch_embed.")
                or name.startswith("time_embed.")
                or name.startswith("final_layer.")
                or name.startswith("time_embed_2.")
                or name.startswith("guidance_emb.")
                or name.startswith("timestep_r_emb.")
            ):
                continue

            if name == "model.ln_f.weight":
                name = "model.norm.weight"
            elif name == "model.wte.weight":
                name = "model.embed_tokens.weight"

            # Remap Q/K norm names
            if "self_attn.query_layernorm.weight" in name:
                name = name.replace(
                    "self_attn.query_layernorm.weight", "self_attn.q_norm.weight"
                )
            elif "self_attn.key_layernorm.weight" in name:
                name = name.replace(
                    "self_attn.key_layernorm.weight", "self_attn.k_norm.weight"
                )

            # Handle QKV: rearrange interleaved layout to [Q, K, V]
            if "self_attn.qkv_proj.weight" in name:
                loaded_weight = self._rearrange_qkv(loaded_weight)

            # Remap gate weight
            if "mlp.gate.wg.weight" in name:
                name = name.replace("mlp.gate.wg.weight", "mlp.gate.weight")

            # Handle shared_mlp: rename gate_and_up_proj -> gate_up_proj
            # The weight is already fused on disk, load with shard_id=None
            if "mlp.shared_mlp.gate_and_up_proj" in name:
                name = name.replace(
                    "mlp.shared_mlp.gate_and_up_proj", "mlp.shared_mlp.gate_up_proj"
                )
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, loaded_shard_id=None)
                continue

            # Handle routed experts: split gate_and_up_proj into gate_proj + up_proj
            # then load through the standard expert_params_mapping
            if "mlp.experts." in name and "gate_and_up_proj.weight" in name:
                gate_w, up_w = loaded_weight.chunk(2, dim=0)
                base = name.replace("gate_and_up_proj.weight", "")
                for split_name, split_w in [
                    (base + "gate_proj.weight", gate_w),
                    (base + "up_proj.weight", up_w),
                ]:
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in split_name:
                            continue
                        mapped = split_name.replace(weight_name, param_name)
                        if mapped not in params_dict:
                            continue
                        param = params_dict[mapped]
                        param.weight_loader(
                            param,
                            split_w,
                            mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        break
                continue

            # Handle routed expert down_proj
            if "mlp.experts." in name and "down_proj.weight" in name:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    if mapped not in params_dict:
                        continue
                    param = params_dict[mapped]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                continue

            if "router.gate." in name:
                name = name.replace("router.", "")

            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = [HunyuanImage3ForCausalMM]
