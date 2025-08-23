import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers, logging
# from diffusers.models.transformer_2d import Transformer2DModelOutput
from diffusers.models.modeling_outputs import Transformer2DModelOutput


# @functools.lru_cache(maxsize=None)
def _compute_video_freqs(self, frame, height, width, idx=0, h_offset=0, w_offset=0):
    seq_lens = frame * height * width
    freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
    freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

    freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
    if self.scale_rope:
        h_low = -(height - height // 2) + h_offset
        h_high = height // 2 + h_offset
        if h_low >= 0:
            freqs_height = freqs_pos[1][h_low : h_high]
        else:
            freqs_height = torch.cat([freqs_neg[1][h_low :], freqs_pos[1][: h_high]], dim=0)
        freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
        w_low = -(width - width // 2) + w_offset
        w_high = width // 2 + w_offset
        if w_low >= 0:
            freqs_width = freqs_pos[2][w_low : w_high]
        else:
            freqs_width = torch.cat([freqs_neg[2][w_low :], freqs_pos[2][: w_high]], dim=0)
        freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
    else:
        freqs_height = freqs_pos[1][h_offset:h_offset+height].view(1, height, 1, -1).expand(frame, height, width, -1)
        freqs_width = freqs_pos[2][w_offset:w_offset+width].view(1, 1, width, -1).expand(frame, height, width, -1)

    freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
    return freqs.clone().contiguous()

def rope_forward(self, video_fhw, txt_seq_lens, device):
    """
    Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
    txt_length: [bs] a list of 1 integers representing the length of the text
    """
    if self.pos_freqs.device != device:
        self.pos_freqs = self.pos_freqs.to(device)
        self.neg_freqs = self.neg_freqs.to(device)

    if isinstance(video_fhw, list):
        video_fhw = video_fhw[0]
    if not isinstance(video_fhw, list):
        video_fhw = [video_fhw]

    vid_freqs = []
    max_vid_index = 0
    for i, fhw in enumerate(video_fhw):
        frame, height, width, idx, h_offset, w_offset = fhw
        rope_key = f"{idx}_{height}_{width}"

        if not torch.compiler.is_compiling():
            if rope_key not in self.rope_cache:
                self.rope_cache[rope_key] = _compute_video_freqs(self, frame, height, width, idx, h_offset, w_offset)
            video_freq = self.rope_cache[rope_key]
        else:
            video_freq = _compute_video_freqs(self, frame, height, width, idx, h_offset, w_offset)
        video_freq = video_freq.to(device)
        vid_freqs.append(video_freq)

        # print("freq", video_freq)

        if self.scale_rope:
            max_vid_index = max(height // 2, width // 2, max_vid_index)
        else:
            max_vid_index = max(height, width, max_vid_index)

    max_len = max(txt_seq_lens)
    txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
    vid_freqs = torch.cat(vid_freqs, dim=0)

    # print(txt_freqs.shape)

    return vid_freqs, txt_freqs


def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
    txt_seq_lens: Optional[List[int]] = None,
    guidance: torch.Tensor = None,  # TODO: this should probably be removed
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    The [`QwenTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
            Mask of the input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    hidden_states = self.img_in(hidden_states)

    timestep = timestep.to(hidden_states.dtype)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states)
    )

    image_rotary_emb = rope_forward(self.pos_embed, img_shapes, txt_seq_lens, device=hidden_states.device)

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                image_rotary_emb,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )

    # Use only the image part (hidden_states) from the dual-stream blocks
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)