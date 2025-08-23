"""
Device management utilities for DeepCausalMMM.

This module handles:
- GPU/CPU device selection
- Memory management
- Mixed precision training
- Multi-GPU support
"""

import torch
import logging
from typing import Union, Tuple, Optional

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for model training/inference.
    
    Args:
        device: Device specification ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
               If None or 'auto', will use CUDA if available
    
    Returns:
        torch.device: Selected device
    """
    if device is None or device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device.startswith('cuda') and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    device = torch.device(device)
    
    if device.type == 'cuda':
        # Log GPU info
        gpu_name = torch.cuda.get_device_name(device.index or 0)
        memory_allocated = torch.cuda.memory_allocated(device.index or 0) / 1024**3
        memory_total = torch.cuda.get_device_properties(device.index or 0).total_memory / 1024**3
        
        logger.info(f"Using GPU: {gpu_name}")
        logger.info(f"GPU Memory: {memory_allocated:.2f}GB used / {memory_total:.2f}GB total")
    else:
        logger.info("Using CPU")
    
    return device


def get_amp_settings(
    device: torch.device,
    mixed_precision: bool = True
) -> Tuple[torch.cuda.amp.GradScaler, bool]:
    """
    Get Automatic Mixed Precision (AMP) settings.
    
    Args:
        device: Current device
        mixed_precision: Whether to enable mixed precision training
    
    Returns:
        Tuple of (gradient scaler, use mixed precision flag)
    """
    use_amp = mixed_precision and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    if use_amp:
        logger.info("Using Automatic Mixed Precision (AMP)")
    
    return scaler, use_amp


def move_to_device(
    data: Union[torch.Tensor, dict, list, tuple],
    device: torch.device
) -> Union[torch.Tensor, dict, list, tuple]:
    """
    Recursively move data to specified device.
    
    Args:
        data: Data to move (can be tensor, dict, list, or tuple)
        device: Target device
    
    Returns:
        Data on target device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(x, device) for x in data)
    return data


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU memory cache")


class DeviceContext:
    """
    Context manager for device management.
    
    Example:
        with DeviceContext(device='auto', mixed_precision=True) as ctx:
            model = model.to(ctx.device)
            for batch in dataloader:
                with ctx.autocast():
                    output = model(batch)
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        mixed_precision: bool = True
    ):
        """
        Initialize device context.
        
        Args:
            device: Device specification
            mixed_precision: Whether to use mixed precision
        """
        self.device = get_device(device)
        self.scaler, self.use_amp = get_amp_settings(self.device, mixed_precision)
        self.autocast = torch.cuda.amp.autocast if self.use_amp else nullcontext
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_gpu_memory()


class nullcontext:
    """Null context manager for CPU fallback."""
    def __init__(self, *args, **kwargs):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass 