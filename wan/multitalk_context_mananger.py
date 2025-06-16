import types
from contextlib import contextmanager
from xfuser.core.distributed import get_sequence_parallel_world_size
from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from .distributed.xdit_context_parallel import (
    usp_dit_forward_multitalk,
    usp_attn_forward_multitalk,
)

from torch import distributed as dist

@contextmanager
def parallel_context(model, use_usp, ulysses_size, ring_size, para_batch_size):
    original_attn_forwards = []
    original_model_forward = model.forward
    original_enable_sp = getattr(model, "enable_sp", False)

    if use_usp:
        world_size=dist.get_world_size()
        rank = dist.get_rank()
        if ulysses_size > 1 or ring_size > 1 or para_batch_size > 1:
            assert ulysses_size * ring_size * para_batch_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
            assert para_batch_size == 1 or para_batch_size == 3, f"The para_batch_size should be 1 or 3, but got {para_batch_size}."

            init_distributed_environment(
                rank=rank, world_size=world_size)

            initialize_model_parallel(
                classifier_free_guidance_degree=para_batch_size,
                sequence_parallel_degree = ulysses_size * ring_size,
                ring_degree=ring_size,
                ulysses_degree=ulysses_size,
            )

            # Save original attention forwards
            for block in model.blocks:
                original_attn_forwards.append(block.self_attn.forward)
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward_multitalk, block.self_attn
                )

            # Patch model forward
            model.forward = types.MethodType(usp_dit_forward_multitalk, model)
            model.enable_sp = True
            sp_size = get_sequence_parallel_world_size()
    else:
        sp_size = 1

    try:
        yield sp_size
    finally:
        if use_usp:
            # Restore original attention forwards
            for block, original_forward in zip(model.blocks, original_attn_forwards):
                block.self_attn.forward = original_forward

            # Restore model forward and original sp flag
            model.forward = original_model_forward
            model.enable_sp = original_enable_sp
