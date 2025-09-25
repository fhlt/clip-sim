import torch
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def get_gpu_info() -> Dict[str, Any]:
    """Get information about available GPUs"""
    if not torch.cuda.is_available():
        return {"available": False, "count": 0, "devices": []}
    
    gpu_count = torch.cuda.device_count()
    devices = []
    
    for i in range(gpu_count):
        device_props = torch.cuda.get_device_properties(i)
        memory_total = device_props.total_memory / (1024**3)  # GB
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        memory_cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
        
        devices.append({
            "id": i,
            "name": device_props.name,
            "memory_total_gb": round(memory_total, 2),
            "memory_allocated_gb": round(memory_allocated, 2),
            "memory_cached_gb": round(memory_cached, 2),
            "memory_free_gb": round(memory_total - memory_allocated, 2),
            "compute_capability": f"{device_props.major}.{device_props.minor}"
        })
    
    return {
        "available": True,
        "count": gpu_count,
        "devices": devices
    }

def print_gpu_info():
    """Print GPU information to console"""
    gpu_info = get_gpu_info()
    
    if not gpu_info["available"]:
        print("CUDA is not available. Using CPU.")
        return
    
    print(f"Found {gpu_info['count']} GPU(s):")
    for device in gpu_info["devices"]:
        print(f"  GPU {device['id']}: {device['name']}")
        print(f"    Memory: {device['memory_allocated_gb']:.2f}GB / {device['memory_total_gb']:.2f}GB used")
        print(f"    Free: {device['memory_free_gb']:.2f}GB")
        print(f"    Compute Capability: {device['compute_capability']}")

def get_optimal_batch_size(model, device: str, max_batch_size: int = 64) -> int:
    """Find optimal batch size for the given model and device"""
    if device == "cpu":
        return min(16, max_batch_size)
    
    try:
        # Test different batch sizes
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            if batch_size > max_batch_size:
                break
                
            try:
                # Create dummy data
                dummy_images = [torch.randn(3, 224, 224) for _ in range(batch_size)]
                dummy_texts = [f"test text {i}" for i in range(batch_size)]
                
                # Test forward pass
                with torch.no_grad():
                    _ = model.compute_similarity_batch(dummy_images, dummy_texts)
                
                # Clear cache
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    return max(1, batch_size // 2)
                else:
                    raise e
        
        return max_batch_size
        
    except Exception as e:
        logger.warning(f"Could not determine optimal batch size: {e}")
        return min(16, max_batch_size)

def monitor_gpu_usage(device: str, interval: float = 1.0):
    """Monitor GPU usage (for debugging)"""
    if device == "cpu" or not torch.cuda.is_available():
        return
    
    device_id = int(device.split(":")[1]) if ":" in device else 0
    
    try:
        while True:
            memory_allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
            memory_cached = torch.cuda.memory_reserved(device_id) / (1024**3)
            memory_total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
            
            print(f"GPU {device_id}: {memory_allocated:.2f}GB / {memory_total:.2f}GB allocated, "
                  f"{memory_cached:.2f}GB cached")
            
            import time
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("GPU monitoring stopped")

def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
