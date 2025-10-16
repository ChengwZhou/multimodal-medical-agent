import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all processes, supporting backward propagation.

    This is a standalone version that doesn't depend on fairseq_signals.

    Usage:
        # Gather tensors from all GPUs
        gathered = GatherLayer.apply(local_tensor)
        # gathered is a tuple of tensors from all processes
    """

    @staticmethod
    def forward(ctx, input):
        """
        Forward pass: gather tensors from all processes

        Args:
            input: local tensor to gather

        Returns:
            tuple of tensors from all processes
        """
        ctx.save_for_backward(input)

        # Check if distributed is initialized
        if not dist.is_available() or not dist.is_initialized():
            return (input,)

        # Get world size
        world_size = dist.get_world_size()

        if world_size == 1:
            return (input,)

        # Gather tensors from all processes
        output = [torch.zeros_like(input) for _ in range(world_size)]
        dist.all_gather(output, input)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        """
        Backward pass: distribute gradients back to corresponding process

        Args:
            grads: tuple of gradients from all processes

        Returns:
            gradient for local process
        """
        (input,) = ctx.saved_tensors

        # Check if distributed is initialized
        if not dist.is_available() or not dist.is_initialized():
            return grads[0]

        # Each process only receives its own gradient
        grad_out = torch.zeros_like(input)
        rank = dist.get_rank()
        grad_out[:] = grads[rank]

        return grad_out


def batch_all_gather(tensor, group=None):
    """
    Helper function for gathering batched tensors

    Args:
        tensor: input tensor to gather
        group: process group (None for default group)

    Returns:
        list of gathered tensors from all processes
    """
    if not dist.is_available() or not dist.is_initialized():
        return [tensor]

    world_size = dist.get_world_size(group)

    if world_size == 1:
        return [tensor]

    # Gather tensors
    output = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(output, tensor, group=group)

    return output


# Example usage
if __name__ == "__main__":
    # Initialize distributed (example)
    # dist.init_process_group(backend='nccl')

    # Create dummy tensor
    x = torch.randn(4, 128, requires_grad=True)

    # Gather from all processes (supports backprop)
    gathered = GatherLayer.apply(x)

    # gathered is a tuple of tensors from all GPUs
    # e.g., if 4 GPUs: (tensor_gpu0, tensor_gpu1, tensor_gpu2, tensor_gpu3)

    # Can concatenate if needed
    if len(gathered) > 1:
        all_x = torch.cat(gathered, dim=0)  # [world_size*4, 128]
    else:
        all_x = gathered[0]

    # Backward pass works correctly
    loss = all_x.sum()
    loss.backward()

    print(f"Gathered {len(gathered)} tensors")
    print(f"Local gradient shape: {x.grad.shape if x.grad is not None else 'None'}")