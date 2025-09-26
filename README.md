# CLIP/SigLIP Similarity Evaluation

A complete pipeline for evaluating CLIP and SigLIP similarity between images and text captions, with support for multi-threaded image downloading and distributed evaluation.

## Features

- **Multi-threaded Image Download**: Download images from URLs in CSV files with configurable worker threads
- **DDP Support**: Distributed Data Parallel evaluation support for multi-GPU setups
- **Multi-Model Support**: Supports both OpenAI's CLIP and Google's SigLIP models for image-text similarity computation
- **Comprehensive Evaluation**: Statistical analysis and visualization of similarity results
- **Flexible Configuration**: Command-line arguments and configuration management

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd clip-sim
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the evaluation pipeline with default settings (CLIP model):

```bash
python poc.py --csv_path data.csv
```

Run with SigLIP model:

```bash
python poc.py --csv_path data.csv --model_type siglip
```

### Advanced Usage

#### Custom Configuration
```bash
# Using CLIP model
python poc.py \
    --csv_path data.csv \
    --model_type clip \
    --batch_size 64 \
    --max_workers 20 \
    --output_dir results \
    --verbose

# Using SigLIP model
python poc.py \
    --csv_path data.csv \
    --model_type siglip \
    --batch_size 64 \
    --max_workers 20 \
    --output_dir results \
    --verbose
```

#### Distributed Evaluation with torch.multiprocessing

**Method 1: Using the main script**
```bash
# Single GPU with CLIP
python poc.py --csv_path data.csv --world_size 1 --model_type clip

# Single GPU with SigLIP
python poc.py --csv_path data.csv --world_size 1 --model_type siglip

# Multi-GPU (2 GPUs) with CLIP
python poc.py --csv_path data.csv --distributed --world_size 2 --model_type clip

# Multi-GPU (4 GPUs) with SigLIP
python poc.py --csv_path data.csv --distributed --world_size 4 --model_type siglip
```

**Method 2: Using the DDP script**
```bash
# Multi-GPU processing with CLIP
python run_ddp.py --csv_path data.csv --distributed --world_size 2 --model_type clip

# Multi-GPU processing with SigLIP
python run_ddp.py --csv_path data.csv --distributed --world_size 2 --model_type siglip
```

**Method 3: Using torch.distributed.launch (legacy)**
```bash
# Single node, multiple GPUs
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    poc.py \
    --csv_path data.csv \
    --distributed \
    --world_size 2
```

#### Multi-node Distributed Training
```bash
# Node 0
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=12355 \
    poc.py \
    --csv_path data.csv \
    --distributed \
    --world_size 4

# Node 1
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=12355 \
    poc.py \
    --csv_path data.csv \
    --distributed \
    --world_size 4
```

## Data Format

The CSV file should contain at least two columns:
- `url`: Image URL
- `caption`: Text caption

Example:
```csv
url,caption
https://example.com/image1.jpg,"A beautiful sunset over the ocean"
https://example.com/image2.jpg,"A cat sitting on a windowsill"
```

## Configuration Options

### Data Options
- `--csv_path`: Path to CSV file with URLs and captions
- `--image_dir`: Directory to save downloaded images (default: `outputs/images`)
- `--output_dir`: Output directory for results (default: `outputs`)

### Model Options
- `--model_name`: Model name (default: `openai/clip-vit-base-patch32` for CLIP, `google/siglip-base-patch16-224` for SigLIP)
- `--model_type`: Model type (`clip` or `siglip`, default: `clip`)
- `--device`: Device to use (`cuda`/`cpu`)

### Download Options
- `--max_workers`: Number of download workers (default: 10)
- `--timeout`: Download timeout in seconds (default: 30)
- `--force_download`: Force re-download of existing images

### Training Options
- `--batch_size`: Batch size for evaluation (default: 32)
- `--num_workers`: Number of data loader workers (default: 4)
- `--no_shuffle`: Don't shuffle data

### Distributed Options
- `--distributed`: Enable distributed training
- `--rank`: Process rank for distributed training
- `--world_size`: Number of processes for distributed training
- `--backend`: Backend for distributed training (default: `nccl`)

### Evaluation Options
- `--save_results`: Save results to file
- `--results_file`: Results file name (default: `similarity_results.csv`)

### Logging Options
- `--log_level`: Logging level (`DEBUG`/`INFO`/`WARNING`/`ERROR`)
- `--verbose`: Verbose output

## Output Files

The pipeline generates several output files:

1. **Downloaded Images**: Saved to `outputs/images/` directory
2. **Results CSV**: `similarity_results.csv` with similarity scores
3. **Distribution Plot**: `similarity_distribution.png` showing score distribution
4. **Heatmap Plot**: `similarity_heatmap.png` showing similarity matrix

## Project Structure

```
clip-sim/
├── data/
│   ├── __init__.py
│   ├── image_downloader.py    # Multi-threaded image downloader
│   └── dataset.py             # Dataset and DataLoader with DDP support
├── models/
│   ├── __init__.py
│   ├── base_evaluator.py      # Base similarity model class
│   ├── clip_evaluator.py      # CLIP model implementation
│   └── siglip_evaluator.py    # SigLIP model implementation
├── utils/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── logger.py              # Logging utilities
│   ├── evaluator.py           # Evaluation and visualization
│   └── gpu_utils.py           # GPU utilities and monitoring
├── outputs/
│   └── images/                # Downloaded images
├── poc.py                     # Main entry point
├── run_ddp.py                 # DDP-specific entry point
├── example_ddp.py             # DDP usage examples
├── test_pipeline.py           # Test script
├── data.csv                   # Input data
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Examples

### Example 1: Basic Evaluation
```bash
# Using CLIP model
python poc.py --csv_path data.csv --batch_size 16 --verbose --model_type clip

# Using SigLIP model
python poc.py --csv_path data.csv --batch_size 16 --verbose --model_type siglip
```

### Example 2: High-throughput Download
```bash
# Using CLIP model
python poc.py \
    --csv_path data.csv \
    --model_type clip \
    --max_workers 50 \
    --timeout 60 \
    --batch_size 64

# Using SigLIP model
python poc.py \
    --csv_path data.csv \
    --model_type siglip \
    --max_workers 50 \
    --timeout 60 \
    --batch_size 64
```

### Example 3: Distributed Evaluation with torch.multiprocessing
```bash
# Using the main script with CLIP
python poc.py --csv_path data.csv --distributed --world_size 4 --batch_size 16 --model_type clip

# Using the main script with SigLIP
python poc.py --csv_path data.csv --distributed --world_size 4 --batch_size 16 --model_type siglip

# Using the DDP script with CLIP
python run_ddp.py --csv_path data.csv --distributed --world_size 4 --batch_size 16 --model_type clip

# Using torch.distributed.launch (legacy) with SigLIP
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    poc.py \
    --csv_path data.csv \
    --distributed \
    --world_size 4 \
    --batch_size 16 \
    --model_type siglip
```

### Example 4: Test DDP Examples
```bash
# Run DDP examples
python example_ddp.py
```

## Model Comparison

### CLIP vs SigLIP

| Feature | CLIP | SigLIP |
|---------|------|--------|
| **Model Type** | Contrastive Learning | Sigmoid Loss |
| **Text Length** | 77 tokens | 64 tokens |
| **Similarity Score** | Cosine similarity (0-1) | Sigmoid probability (0-1) |
| **Performance** | Good general performance | Often better on specific tasks |
| **Speed** | Standard | Similar to CLIP |
| **Use Case** | General image-text matching | When you need probability scores |

### Choosing a Model

- **Use CLIP** (`--model_type clip`): For general image-text similarity tasks, when you need cosine similarity scores
- **Use SigLIP** (`--model_type siglip`): When you need probability-based scores, or for tasks where SigLIP has shown better performance

## Performance Tips

1. **Download Optimization**: Increase `--max_workers` for faster downloads
2. **Memory Optimization**: Reduce `--batch_size` if running out of memory
3. **Distributed Training**: Use multiple GPUs with `--distributed` for large datasets
4. **Data Loading**: Increase `--num_workers` for faster data loading

## License

This project is licensed under the MIT License.
