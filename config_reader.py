"""
Configuration reader for Voice-to-Image Pipeline
Reads model settings from models.config file
"""

import configparser
from pathlib import Path

class ModelConfig:
    """Read and manage model configuration"""
    
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = Path(__file__).parent / "models.config"
        
        self.config_file = Path(config_file)
        self.config = configparser.ConfigParser()
        
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        self.config.read(self.config_file)
        self._base_dir = self.config_file.parent
    
    # Helper method to generate local directory name from HF ID
    @staticmethod
    def _hf_id_to_local_dir(hf_id):
        """Convert HuggingFace model ID to local directory name"""
        # Extract model name from format like "owner/model-name"
        return hf_id.split('/')[-1]
    
    # Model paths
    @property
    def models_dir(self):
        """Base directory for all models"""
        return self._base_dir / "models"
    
    @property
    def whisper_hf_id(self):
        """HuggingFace ID for Whisper model"""
        return self.config.get('models', 'whisper_hf_id')
    
    @property
    def whisper_local_dir(self):
        """Local directory name for Whisper model (auto-generated)"""
        return self._hf_id_to_local_dir(self.whisper_hf_id)
    
    @property
    def whisper_path(self):
        """Full path to Whisper model"""
        return self.models_dir / self.whisper_local_dir
    
    @property
    def llm_hf_id(self):
        """HuggingFace ID for LLM model"""
        return self.config.get('models', 'llm_hf_id')
    
    @property
    def llm_local_dir(self):
        """Local directory name for LLM model (auto-generated)"""
        return self._hf_id_to_local_dir(self.llm_hf_id)
    
    @property
    def llm_path(self):
        """Full path to LLM model"""
        return self.models_dir / self.llm_local_dir
    
    @property
    def image_hf_id(self):
        """HuggingFace ID for Image model"""
        return self.config.get('models', 'image_hf_id')
    
    @property
    def image_local_dir(self):
        """Local directory name for Image model (auto-generated)"""
        return self._hf_id_to_local_dir(self.image_hf_id)
    
    @property
    def image_path(self):
        """Full path to Image model"""
        return self.models_dir / self.image_local_dir
    
    # Device defaults
    @property
    def whisper_default_device(self):
        """Default device for Whisper"""
        return self.config.get('devices', 'whisper_default', fallback='NPU')
    
    @property
    def llm_default_device(self):
        """Default device for LLM"""
        return self.config.get('devices', 'llm_default', fallback='GPU')
    
    @property
    def image_default_device(self):
        """Default device for Image Generation"""
        return self.config.get('devices', 'image_default', fallback='GPU')
    
    # Image sizes
    @property
    def image_sizes(self):
        """Get list of pre-compiled image sizes as (width, height) tuples"""
        sizes = []
        for key in ['size_1', 'size_2', 'size_3']:
            if self.config.has_option('image_sizes', key):
                size_str = self.config.get('image_sizes', key)
                width, height = [int(x.strip()) for x in size_str.split(',')]
                sizes.append((width, height))
        return sizes
    
    # Cache
    @property
    def cache_dir(self):
        """Cache directory for compiled models"""
        cache_name = self.config.get('cache', 'cache_dir', fallback='.cache')
        return self._base_dir / cache_name
    
    def get_cache_dir(self, device):
        """Get cache directory for specific device"""
        cache_subdir = self.cache_dir / f"{device.lower()}cache"
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir
    
    def print_config(self):
        """Print configuration summary"""
        print("=" * 70)
        print("MODEL CONFIGURATION")
        print("=" * 70)
        print(f"Config file: {self.config_file}")
        print()
        print("Models:")
        print(f"  Whisper:")
        print(f"    HuggingFace: {self.whisper_hf_id}")
        print(f"    Local: {self.whisper_path}")
        print(f"  LLM:")
        print(f"    HuggingFace: {self.llm_hf_id}")
        print(f"    Local: {self.llm_path}")
        print(f"  Image:")
        print(f"    HuggingFace: {self.image_hf_id}")
        print(f"    Local: {self.image_path}")
        print()
        print("Default Devices:")
        print(f"  Whisper: {self.whisper_default_device}")
        print(f"  LLM: {self.llm_default_device}")
        print(f"  Image: {self.image_default_device}")
        print()
        print("Image Sizes:")
        for i, (w, h) in enumerate(self.image_sizes, 1):
            print(f"  Size {i}: {w}x{h}")
        print()
        print(f"Cache Directory: {self.cache_dir}")
        print("=" * 70)

# Global config instance
_config = None

def get_config():
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = ModelConfig()
    return _config
