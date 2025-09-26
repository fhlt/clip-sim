# CLIP Similarity Evaluation

A complete pipeline for evaluating CLIP similarity between images and text captions, with support for multi-threaded image downloading and distributed training.

## Features

- **Multi-threaded Image Download**: Download images from URLs in CSV files with configurable worker threads
- **DDP Support**: Distributed Data Parallel training support for multi-GPU setups
- **CLIP Integration**: Uses OpenAI's CLIP model for image-text similarity computation
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

Run the evaluation pipeline with default settings:

```bash
python poc.py --csv_path data.csv
```

### Advanced Usage

#### Custom Configuration
```bash
python poc.py \
    --csv_path data.csv \
    --batch_size 64 \
    --max_workers 20 \
    --output_dir results \
    --verbose
```

#### Distributed Training with torch.multiprocessing

**Method 1: Using the main script**
```bash
# Single GPU
python poc.py --csv_path data.csv --world_size 1

# Multi-GPU (2 GPUs)
python poc.py --csv_path data.csv --distributed --world_size 2

# Multi-GPU (4 GPUs)
python poc.py --csv_path data.csv --distributed --world_size 4
```

**Method 2: Using the DDP script**
```bash
# Multi-GPU processing
python run_ddp.py --csv_path data.csv --distributed --world_size 2
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
- `--model_name`: CLIP model name (default: `openai/clip-vit-base-patch32`)
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
│   └── clip_evaluator.py      # CLIP model implementation
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
python poc.py --csv_path data.csv --batch_size 16 --verbose
```

### Example 2: High-throughput Download
```bash
python poc.py \
    --csv_path data.csv \
    --max_workers 50 \
    --timeout 60 \
    --batch_size 64
```

### Example 3: Distributed Evaluation with torch.multiprocessing
```bash
# Using the main script
python poc.py --csv_path data.csv --distributed --world_size 4 --batch_size 16

# Using the DDP script
python run_ddp.py --csv_path data.csv --distributed --world_size 4 --batch_size 16

# Using torch.distributed.launch (legacy)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    poc.py \
    --csv_path data.csv \
    --distributed \
    --world_size 4 \
    --batch_size 16
```

### Example 4: Test DDP Examples
```bash
# Run DDP examples
python example_ddp.py
```

## Performance Tips

1. **Download Optimization**: Increase `--max_workers` for faster downloads
2. **Memory Optimization**: Reduce `--batch_size` if running out of memory
3. **Distributed Training**: Use multiple GPUs with `--distributed` for large datasets
4. **Data Loading**: Increase `--num_workers` for faster data loading

## License

This project is licensed under the MIT License.
