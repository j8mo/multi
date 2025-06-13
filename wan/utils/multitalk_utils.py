import os
from einops import rearrange

import torch
import torch.nn as nn

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from einops import rearrange, repeat
from functools import lru_cache
import imageio
import uuid
from tqdm import tqdm
import numpy as np
import subprocess
import soundfile as sf


VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
ASPECT_RATIO_627 = {
     '0.26': ([320, 1216], 1), '0.38': ([384, 1024], 1), '0.50': ([448, 896], 1), '0.67': ([512, 768], 1),
     '0.82': ([576, 704], 1),  '1.00': ([640, 640], 1),  '1.22': ([704, 576], 1), '1.50': ([768, 512], 1),
     '1.86': ([832, 448], 1),  '2.00': ([896, 448], 1),  '2.50': ([960, 384], 1), '2.83': ([1088, 384], 1),
     '3.60': ([1152, 320], 1), '3.80': ([1216, 320], 1), '4.00': ([1280, 320], 1)}


ASPECT_RATIO_960 = {
     '0.22': ([448, 2048], 1), '0.29': ([512, 1792], 1), '0.36': ([576, 1600], 1), '0.45': ([640, 1408], 1),
     '0.55': ([704, 1280], 1), '0.63': ([768, 1216], 1), '0.76': ([832, 1088], 1), '0.88': ([896, 1024], 1),
     '1.00': ([960, 960], 1), '1.14': ([1024, 896], 1), '1.31': ([1088, 832], 1), '1.50': ([1152, 768], 1),
     '1.58': ([1216, 768], 1), '1.82': ([1280, 704], 1), '1.91': ([1344, 704], 1), '2.20': ([1408, 640], 1),
     '2.30': ([1472, 640], 1), '2.67': ([1536, 576], 1), '2.89': ([1664, 576], 1), '3.62': ([1856, 512], 1),
     '3.75': ([1920, 512], 1)}



def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()



@torch.compile
def calculate_x_ref_attn_map(visual_q, ref_k, ref_target_masks, mode='mean', attn_bias=None):

    ref_k = ref_k.to(visual_q.dtype).to(visual_q.device)
    scale = 1.0 / visual_q.shape[-1] ** 0.5
    visual_q = visual_q * scale
    visual_q = visual_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = visual_q @ ref_k.transpose(-2, -1)

    if attn_bias is not None:
        attn = attn + attn_bias

    x_ref_attn_map_source = attn.softmax(-1) # B, H, x_seqlens, ref_seqlens


    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.to(visual_q.dtype)
    x_ref_attn_map_source = x_ref_attn_map_source.to(visual_q.dtype)

    for class_idx, ref_target_mask in enumerate(ref_target_masks):
        torch_gc()
        ref_target_mask = ref_target_mask[None, None, None, ...]
        x_ref_attn_map = x_ref_attn_map_source.clone()
        x_ref_attn_map = x_ref_attn_map * ref_target_mask
        x_ref_attn_map = x_ref_attn_map.sum(-1) / ref_target_mask.sum() # B, H, x_seqlens, ref_seqlens --> B, H, x_seqlens
        x_ref_attn_map = x_ref_attn_map.permute(0, 2, 1) # B, x_seqlens, H

        if mode == 'mean':
            x_ref_attn_map = x_ref_attn_map.mean(-1) # B, x_seqlens
        elif mode == 'max':
            x_ref_attn_map = x_ref_attn_map.max(-1) # B, x_seqlens

        x_ref_attn_maps.append(x_ref_attn_map)

    del attn
    del x_ref_attn_map_source
    torch_gc()

    return torch.stack([x_ref_attn_map for x_ref_attn_map in x_ref_attn_maps], dim=1) # B, class_num, x_seqlens

def get_attn_map_with_target(visual_q, ref_k, shape, ref_target_masks=None, split_num=2, enable_sp=False):
    """Args:
        query (torch.tensor): B M H K
        key (torch.tensor): B M H K
        shape (tuple): (N_t, N_h, N_w)
        ref_target_masks: [B, N_h * N_w]
    """

    batch_size = visual_q.shape[0]
    N_t, N_h, N_w = shape
    if enable_sp:
        ref_k = get_sp_group().all_gather(ref_k, dim=1)

    x_seqlens = N_h * N_w
    # visual_q  = query # 1, 3024, 12, 128
    ref_k     = ref_k[:, :x_seqlens] # 1, 144, 12, 128
    _, seq_lens, heads, _ = visual_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = torch.zeros(batch_size, class_num, seq_lens).to(visual_q.device).to(visual_q.dtype)

    split_chunk = heads // split_num

    for i in range(split_num):
        x_ref_attn_maps_perhead = calculate_x_ref_attn_map(visual_q[:, :, i*split_chunk:(i+1)*split_chunk, :], ref_k[:, :, i*split_chunk:(i+1)*split_chunk, :], ref_target_masks)
        x_ref_attn_maps += x_ref_attn_maps_perhead

    return x_ref_attn_maps / split_num


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RotaryPositionalEmbedding1D(nn.Module):

    def __init__(self,
                 head_dim,
                 ):
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000


    @lru_cache(maxsize=32)
    def precompute_freqs_cis_1d(self, pos_indices):

        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))
        freqs = freqs.to(pos_indices.device)
        freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def forward(self, x, pos_indices):
        """1D RoPE.

        Args:
            query (torch.tensor): [B, head, seq, head_dim]
            pos_indices (torch.tensor): [seq,]
        Returns:
            query with the same shape as input.
        """
        freqs_cis = self.precompute_freqs_cis_1d(pos_indices)

        x_ = x.float()

        freqs_cis = freqs_cis.float().to(x.device)
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = rearrange(cos, 'b n d -> b 1 n d'), rearrange(sin, 'b n d -> b 1 n d')
        x_ = (x_ * cos) + (rotate_half(x_) * sin)

        return x_.type_as(x)


def save_video_ffmpeg(gen_video_samples, save_path, vocal_audio_list, fps=25, quality=5):

    def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
        writer = imageio.get_writer(
            save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
        )
        for frame in tqdm(frames, desc="Saving video"):
            frame = np.array(frame)
            writer.append_data(frame)
        writer.close()
    save_path_tmp = save_path + "-temp.mp4"

    video_audio = (gen_video_samples+1)/2 # C T H W
    video_audio = video_audio.permute(1, 2, 3, 0).cpu().numpy()
    video_audio = np.clip(video_audio * 255, 0, 255).astype(np.uint8)  # to [0, 255]
    save_video(video_audio, save_path_tmp, fps=fps, quality=quality)


    # crop audio according to video length
    _, T, _, _ = gen_video_samples.shape
    duration = T / fps
    save_path_crop_audio = save_path + "-cropaudio.wav"
    final_command = [
        "ffmpeg",
        "-i",
        vocal_audio_list[0],
        "-t",
        f'{duration}',
        save_path_crop_audio,
    ]
    subprocess.run(final_command, check=True)


    # generate video with audio
    save_path = save_path + ".mp4"
    final_command = [
        "ffmpeg",
        "-y",
        "-i",
        save_path_tmp,
        "-i",
        save_path_crop_audio,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        save_path,
    ]
    subprocess.run(final_command, check=True)
    os.remove(save_path_tmp)
    os.remove(save_path_crop_audio)


