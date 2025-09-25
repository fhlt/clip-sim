import os
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from PIL import Image
import logging
from typing import List, Tuple, Optional
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ImageDownloader:
    """Multi-threaded image downloader"""
    
    def __init__(self, output_dir: str = "outputs/images", max_workers: int = 10, timeout: int = 30):
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_filename_from_url(self, url: str, index: int) -> str:
        """Extract filename from URL or create one"""
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename or '.' not in filename:
            filename = f"image_{index}.jpg"
        
        return filename
    
    def _download_single_image(self, url: str, index: int) -> Tuple[int, bool, str, Optional[str]]:
        """Download a single image"""
        try:
            filename = self._get_filename_from_url(url, index)
            filepath = os.path.join(self.output_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(filepath):
                return index, True, filepath, None
            
            response = self.session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Check if it's actually an image
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                return index, False, None, f"Invalid content type: {content_type}"
            
            # Download and save image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify it's a valid image
            try:
                with Image.open(filepath) as img:
                    img.verify()
                return index, True, filepath, None
            except Exception as e:
                os.remove(filepath)
                return index, False, None, f"Invalid image file: {str(e)}"
                
        except requests.exceptions.RequestException as e:
            return index, False, None, f"Request error: {str(e)}"
        except Exception as e:
            return index, False, None, f"Unexpected error: {str(e)}"
    
    def download_images(self, urls: List[str], show_progress: bool = True) -> Tuple[List[str], List[str]]:
        """
        Download images from URLs using multi-threading
        
        Args:
            urls: List of image URLs
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (successful_filepaths, error_messages)
        """
        successful_paths = [None] * len(urls)
        error_messages = [None] * len(urls)
        
        logger.info(f"Starting download of {len(urls)} images with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._download_single_image, url, i): i 
                for i, url in enumerate(urls)
            }
            
            # Process completed tasks
            if show_progress:
                futures = tqdm(as_completed(future_to_index), total=len(urls), desc="Downloading images")
            else:
                futures = as_completed(future_to_index)
            
            for future in futures:
                index, success, filepath, error = future.result()
                
                if success:
                    successful_paths[index] = filepath
                else:
                    error_messages[index] = error
                    logger.warning(f"Failed to download image {index}: {error}")
        
        # Filter out None values
        successful_paths = [path for path in successful_paths if path is not None]
        error_messages = [msg for msg in error_messages if msg is not None]
        
        logger.info(f"Download completed: {len(successful_paths)} successful, {len(error_messages)} failed")
        
        return successful_paths, error_messages
    
    def download_from_csv(self, csv_path: str, url_column: str = "url", 
                         caption_column: str = "caption", show_progress: bool = True) -> Tuple[List[str], List[str], List[str]]:
        """
        Download images from CSV file
        
        Args:
            csv_path: Path to CSV file
            url_column: Name of URL column
            caption_column: Name of caption column
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (successful_filepaths, captions, error_messages)
        """
        import pandas as pd
        
        # Read CSV
        df = pd.read_csv(csv_path)
        urls = df[url_column].tolist()
        captions = df[caption_column].tolist()
        
        # Download images
        successful_paths, error_messages = self.download_images(urls, show_progress)
        
        # Filter captions to match successful downloads
        successful_captions = []
        for i, path in enumerate(successful_paths):
            # Find the original index of this successful download
            for j, url in enumerate(urls):
                if path and self._get_filename_from_url(url, j) in path:
                    successful_captions.append(captions[j])
                    break
        
        return successful_paths, successful_captions, error_messages
