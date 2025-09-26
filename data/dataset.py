import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict, Any
import torch.distributed as dist
import random
import numpy as np

logger = logging.getLogger(__name__)

def worker_init_fn(worker_id):
    """Initialize worker with reproducible random seed"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class CLIPDataset(Dataset):
    """Dataset for CLIP similarity evaluation"""
    
    def __init__(self, image_paths: List[str], captions: List[str], transform=None):
        """
        Args:
            image_paths: List of image file paths
            captions: List of corresponding captions
            transform: Optional image transformations
        """
        self.image_paths = image_paths
        self.captions = captions
        self.transform = transform
        
        # Filter out invalid entries
        self.valid_indices = []
        for i, (path, caption) in enumerate(zip(image_paths, captions)):
            if path and os.path.exists(path) and caption:
                self.valid_indices.append(i)
        
        logger.info(f"Dataset initialized with {len(self.valid_indices)} valid samples out of {len(image_paths)} total")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        image_path = self.image_paths[actual_idx]
        caption = self.captions[actual_idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms if provided
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image,
                'caption': caption,
                'image_path': image_path,
                'index': actual_idx
            }
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {str(e)}")
            # Return a placeholder
            return {
                'image': None,
                'caption': caption,
                'image_path': image_path,
                'index': actual_idx
            }

def create_dataloader(
    image_paths: List[str], 
    captions: List[str], 
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    transform=None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = None
) -> DataLoader:
    """
    Create DataLoader with optional DDP support
    
    Args:
        image_paths: List of image file paths
        captions: List of corresponding captions
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        transform: Optional image transformations
        distributed: Whether to use distributed training
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        DataLoader instance
    """
    dataset = CLIPDataset(image_paths, captions, transform)
    
    # Create sampler for distributed training
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=shuffle
        )
        shuffle = False  # Don't shuffle when using DistributedSampler
    
    # Set up worker initialization for reproducibility
    worker_init = worker_init_fn if seed is not None else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=worker_init
    )
    
    return dataloader

def collate_fn(batch):
    """Custom collate function to handle None images"""
    images = []
    captions = []
    image_paths = []
    indices = []
    
    for item in batch:
        if item['image'] is not None:
            images.append(item['image'])
            captions.append(item['caption'])
            image_paths.append(item['image_path'])
            indices.append(item['index'])
    
    return {
        'images': images,
        'captions': captions,
        'image_paths': image_paths,
        'indices': indices
    }

def load_data_from_csv(
    csv_path: str,
    image_dir: str = "outputs/images",
    url_column: str = "url",
    caption_column: str = "caption"
) -> Tuple[List[str], List[str]]:
    """
    Load data from CSV file and match with local images
    
    Args:
        csv_path: Path to CSV file
        image_dir: Directory containing downloaded images
        url_column: Name of URL column
        caption_column: Name of caption column
        
    Returns:
        Tuple of (image_paths, captions)
    """
    df = pd.read_csv(csv_path)
    
    image_paths = []
    captions = []
    
    for _, row in df.iterrows():
        url = row[url_column]
        caption = row[caption_column]
        
        # Generate expected filename
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename or '.' not in filename:
            # Use row index as filename
            filename = f"image_{len(image_paths)}.jpg"
        
        image_path = os.path.join(image_dir, filename)
        
        if os.path.exists(image_path):
            image_paths.append(image_path)
            captions.append(caption)
        else:
            logger.warning(f"Image not found: {image_path}")
    
    logger.info(f"Loaded {len(image_paths)} valid image-caption pairs from {csv_path}")
    return image_paths, captions

def setup_ddp(rank: int, world_size: int, backend: str = "nccl"):
    """Setup distributed training with proper GPU mapping"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
        print(f"Process {rank} assigned to GPU {rank}")
    else:
        device = "cpu"
        print(f"Process {rank} using CPU (CUDA not available)")
    
    return device

def cleanup_ddp():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def run_distributed_worker(rank: int, world_size: int, config: Dict, image_paths: List[str], captions: List[str]):
    """Worker function for distributed processing using torch.multiprocessing"""
    import logging
    from utils.logger import setup_logger
    from utils.evaluator import SimilarityEvaluator
    
    # Setup logging for this process
    logger = setup_logger(
        name=f"clip_sim_rank_{rank}",
        level=config.get('log_level', 'INFO'),
        verbose=config.get('verbose', True)
    )
    
    try:
        # Setup DDP
        device = setup_ddp(rank, world_size, config.get('backend', 'nccl'))
        config['device'] = device
        config['rank'] = rank
        config['world_size'] = world_size
        config['distributed'] = True
        
        logger.info(f"Worker {rank} started on device {device}")
        
        # Create model based on model type
        model_type = config.get('model_type', 'clip').lower()
        if model_type == "siglip":
            from models.siglip_evaluator import SigLIPSimilarityModel
            model = SigLIPSimilarityModel(device=device)
        else:  # default to CLIP
            from models.clip_evaluator import CLIPSimilarityModel
            model = CLIPSimilarityModel(device=device)
        
        # Create data loader
        dataloader = create_dataloader(
            image_paths=image_paths,
            captions=captions,
            batch_size=config.get('batch_size', 32),
            num_workers=config.get('num_workers', 4),
            shuffle=config.get('shuffle', True),
            transform=None,
            distributed=True,
            rank=rank,
            world_size=world_size,
            seed=config.get('seed', None)
        )
        
        logger.info(f"Worker {rank}: Created data loader with {len(dataloader)} batches")
        
        # Process data
        all_similarities = []
        all_image_paths = []
        all_captions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if not batch['images']:
                    continue
                    
                similarities = model.compute_similarity_batch(
                    batch['images'], 
                    batch['captions']
                )
                
                all_similarities.extend(similarities)
                all_image_paths.extend(batch['image_paths'])
                all_captions.extend(batch['captions'])
                
                if config.get('verbose', True) and batch_idx % 10 == 0:
                    logger.info(f"Worker {rank}: Processed batch {batch_idx}/{len(dataloader)}")
        
        logger.info(f"Worker {rank}: Completed processing {len(all_similarities)} samples")
        
        # Gather results from all processes
        gathered_similarities = [None for _ in range(world_size)]
        gathered_paths = [None for _ in range(world_size)]
        gathered_captions = [None for _ in range(world_size)]
        
        dist.all_gather_object(gathered_similarities, all_similarities)
        dist.all_gather_object(gathered_paths, all_image_paths)
        dist.all_gather_object(gathered_captions, all_captions)
        
        # Only rank 0 processes the final results
        if rank == 0:
            # Flatten gathered results
            final_similarities = []
            final_paths = []
            final_captions = []
            
            for sims, paths, caps in zip(gathered_similarities, gathered_paths, gathered_captions):
                final_similarities.extend(sims)
                final_paths.extend(paths)
                final_captions.extend(caps)
            
            logger.info(f"Gathered results from all workers: {len(final_similarities)} total samples")
            
            # Generate evaluation report
            evaluator = SimilarityEvaluator(output_dir=config.get('output_dir', 'outputs'))
            report = evaluator.generate_report(
                similarities=final_similarities,
                image_paths=final_paths,
                captions=final_captions,
                threshold=0.5
            )
            
            logger.info("Evaluation completed successfully!")
            logger.info(f"Results saved to: {config.get('output_dir', 'outputs')}")
            
            # Print summary
            metrics = report['metrics']
            logger.info("=== Summary ===")
            logger.info(f"Mean similarity: {metrics['mean_similarity']:.4f}")
            logger.info(f"Std similarity: {metrics['std_similarity']:.4f}")
            logger.info(f"Min similarity: {metrics['min_similarity']:.4f}")
            logger.info(f"Max similarity: {metrics['max_similarity']:.4f}")
            
            return report
        
        # Synchronize all processes
        dist.barrier()
        return None
        
    except Exception as e:
        logger.error(f"Worker {rank} error: {str(e)}")
        raise
    finally:
        cleanup_ddp()

def launch_distributed_training(config: Dict, image_paths: List[str], captions: List[str]):
    """Launch distributed training using torch.multiprocessing.spawn"""
    world_size = config.get('world_size', 1)
    
    if world_size == 1:
        # Single process mode
        config['distributed'] = False
        config['rank'] = 0
        config['world_size'] = 1
        
        # Run single process
        from poc import evaluate_similarities
        from utils.logger import setup_logger
        
        logger = setup_logger(
            name="clip_sim",
            level=config.get('log_level', 'INFO'),
            verbose=config.get('verbose', True)
        )
        
        similarities, final_paths, final_captions = evaluate_similarities(
            type('Config', (), config), image_paths, captions, logger
        )
        
        # Generate report
        from utils.evaluator import SimilarityEvaluator
        evaluator = SimilarityEvaluator(output_dir=config.get('output_dir', 'outputs'))
        report = evaluator.generate_report(
            similarities=similarities,
            image_paths=final_paths,
            captions=final_captions,
            threshold=0.5
        )
        
        return report
    else:
        # Multi-process mode using torch.multiprocessing.spawn
        logger = logging.getLogger(__name__)
        logger.info(f"Launching {world_size} processes using torch.multiprocessing.spawn")
        
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)
        
        # Use spawn to launch processes
        try:
            mp.spawn(
                run_distributed_worker,
                args=(world_size, config, image_paths, captions),
                nprocs=world_size,
                join=True
            )
            # For multi-GPU mode, return a simple success indicator
            return {"status": "completed", "world_size": world_size}
        except Exception as e:
            logger.error(f"Error in multiprocessing spawn: {e}")
            raise
