#!/usr/bin/env python3
"""
Complete setup script for Voice-to-Image Pipeline:
1. Downloads models from HuggingFace
2. Pre-compiles and caches models for fast loading

Models are configured in models.config file.
Run this once before using the application.
"""

import argparse
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download
import openvino_genai as ov_genai
import openvino as ov
from config_reader import get_config


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_model(repo_id, local_dir, description):
    """Download a model from HuggingFace Hub"""
    print(f"\n{'='*70}")
    print(f"Downloading {description}")
    print(f"Repository: {repo_id}")
    print(f"Destination: {local_dir}")
    print(f"{'='*70}\n")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ Successfully downloaded {description}\n")
        return True
    except Exception as e:
        print(f"✗ Error downloading {description}: {str(e)}\n")
        return False


def download_all_models(config):
    """Download all required models"""
    print("\n" + "="*70)
    print("STEP 1: DOWNLOADING MODELS")
    print("="*70)
    
    # Create models directory
    config.models_dir.mkdir(exist_ok=True)
    
    models = [
        {
            "repo_id": config.whisper_hf_id,
            "local_dir": config.whisper_path,
            "description": "Whisper (Speech Recognition)"
        },
        {
            "repo_id": config.llm_hf_id,
            "local_dir": config.llm_path,
            "description": "LLM (Prompt Enhancement)"
        },
        {
            "repo_id": config.image_hf_id,
            "local_dir": config.image_path,
            "description": "LCM-SDXL (Image Generation)"
        }
    ]
    
    results = []
    for model in models:
        success = download_model(
            model["repo_id"],
            model["local_dir"],
            model["description"]
        )
        results.append((model["description"], success))
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    for description, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {description}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n✓ All models downloaded successfully!")
    else:
        print("\n✗ Some models failed to download.")
    
    return all_success


# ============================================================================
# COMPILATION FUNCTIONS
# ============================================================================

def check_model_exists(model_path, model_name):
    """Check if model directory exists"""
    if not model_path.exists():
        print(f"❌ Error: {model_name} not found at {model_path}")
        return False
    return True


def precompile_whisper(device, config):
    """Pre-compile Whisper model"""
    print(f"\n{'='*70}")
    print(f"COMPILING WHISPER MODEL ON {device}")
    print(f"{'='*70}")
    
    if not check_model_exists(config.whisper_path, "Whisper model"):
        return False
    
    try:
        cache_config = {"CACHE_DIR": str(config.get_cache_dir(device))}
        print(f"Cache directory: {cache_config['CACHE_DIR']}")
        print(f"Loading model from: {config.whisper_path}")
        
        start_time = time.time()
        pipe = ov_genai.WhisperPipeline(str(config.whisper_path), device, **cache_config)
        end_time = time.time()
        
        print(f"✓ Whisper compiled successfully on {device}")
        print(f"  Compilation time: {end_time - start_time:.2f} seconds")
        
        # Test with dummy audio
        print(f"  Testing with dummy audio...")
        test_audio = [0.0] * 16000  # 1 second of silence
        result = pipe.generate(test_audio)
        print(f"  Test successful: '{result.texts[0]}'")
        
        del pipe
        return True
        
    except Exception as e:
        print(f"❌ Error compiling Whisper on {device}: {str(e)}")
        return False


def precompile_llm(device, config):
    """Pre-compile LLM model"""
    print(f"\n{'='*70}")
    print(f"COMPILING LLM MODEL ON {device}")
    print(f"{'='*70}")
    
    if not check_model_exists(config.llm_path, "LLM model"):
        return False
    
    try:
        cache_config = {"CACHE_DIR": str(config.get_cache_dir(device))}
        print(f"Cache directory: {cache_config['CACHE_DIR']}")
        print(f"Loading model from: {config.llm_path}")
        
        start_time = time.time()
        pipe = ov_genai.LLMPipeline(str(config.llm_path), device, **cache_config)
        end_time = time.time()
        
        print(f"✓ LLM compiled successfully on {device}")
        print(f"  Compilation time: {end_time - start_time:.2f} seconds")
        
        # Test with dummy prompt
        print(f"  Testing with sample prompt...")
        result = pipe.generate("Hello", max_new_tokens=10)
        print(f"  Test successful: '{result}'")
        
        del pipe
        return True
        
    except Exception as e:
        print(f"❌ Error compiling LLM on {device}: {str(e)}")
        return False


def precompile_image_pipeline(device, config, width=1024, height=1024):
    """Pre-compile image generation pipeline"""
    print(f"\n{'='*70}")
    print(f"COMPILING IMAGE PIPELINE ON {device} ({width}x{height})")
    print(f"{'='*70}")
    
    if not check_model_exists(config.image_path, "Image model"):
        return False
    
    try:
        cache_config = {"CACHE_DIR": str(config.get_cache_dir(device))}
        print(f"Cache directory: {cache_config['CACHE_DIR']}")
        print(f"Loading model from: {config.image_path}")
        
        load_start = time.time()
        pipe = ov_genai.Text2ImagePipeline(str(config.image_path))
        load_end = time.time()
        print(f"  Model loaded in {load_end - load_start:.2f} seconds")
        
        # Reshape for desired dimensions
        print(f"  Reshaping for {width}x{height}...")
        reshape_start = time.time()
        guidance_scale = pipe.get_generation_config().guidance_scale
        pipe.reshape(1, height, width, guidance_scale)
        reshape_end = time.time()
        print(f"  Reshaped in {reshape_end - reshape_start:.2f} seconds")
        
        # Compile
        print(f"  Compiling on {device}...")
        compile_start = time.time()
        pipe.compile(device, device, device, config=cache_config)
        compile_end = time.time()
        
        print(f"✓ Image pipeline compiled successfully on {device}")
        print(f"  Compilation time: {compile_end - compile_start:.2f} seconds")
        print(f"  Total time: {compile_end - load_start:.2f} seconds")
        
        # Test generation
        print(f"  Testing with sample prompt...")
        test_start = time.time()
        result = pipe.generate("a cat", num_inference_steps=1, rng_seed=42)
        test_end = time.time()
        print(f"  Test successful (took {test_end - test_start:.2f}s)")
        
        del pipe
        return True
        
    except Exception as e:
        print(f"❌ Error compiling Image pipeline on {device}: {str(e)}")
        return False


def compile_all_models(config, whisper_device, llm_device, image_device, multi_size=True):
    """Compile all models"""
    print("\n" + "="*70)
    print("STEP 2: PRE-COMPILING MODELS")
    print("="*70)
    print(f"Cache directory: {config.cache_dir}")
    print()
    print("Devices:")
    print(f"  Whisper: {whisper_device}")
    print(f"  LLM: {llm_device}")
    print(f"  Image: {image_device}")
    print()
    
    # Check available devices
    core = ov.Core()
    available_devices = core.available_devices
    print(f"Available OpenVINO devices: {', '.join(available_devices)}")
    
    results = []
    total_start = time.time()
    
    # Compile Whisper
    success = precompile_whisper(whisper_device, config)
    results.append(("Whisper", whisper_device, success))
    
    # Compile LLM
    success = precompile_llm(llm_device, config)
    results.append(("LLM", llm_device, success))
    
    # Compile Image Pipeline
    if multi_size:
        # Use sizes from config
        sizes = config.image_sizes
        print(f"\nCompiling image pipeline for {len(sizes)} sizes from config...")
        for width, height in sizes:
            success = precompile_image_pipeline(image_device, config, width, height)
            results.append((f"Image {width}x{height}", image_device, success))
    else:
        success = precompile_image_pipeline(image_device, config, 1024, 1024)
        results.append(("Image 1024x1024", image_device, success))
    
    total_end = time.time()
    
    # Summary
    print(f"\n{'='*70}")
    print("COMPILATION SUMMARY")
    print(f"{'='*70}")
    
    for model, device, success in results:
        status = "✓ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {model} on {device}")
    
    print(f"\nTotal compilation time: {total_end - total_start:.2f} seconds")
    
    all_success = all(success for _, _, success in results)
    
    if all_success:
        print("\n✓ All models compiled successfully!")
    else:
        print("\n⚠ Some models failed to compile.")
    
    return all_success


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete setup for Voice-to-Image Pipeline (Download + Compile)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full setup with default devices (Whisper: NPU, LLM & Image: GPU)
  python setup_models.py
  
  # Setup with custom devices (using long form)
  python setup_models.py --whisper-device NPU --llm-device GPU --image-device GPU
  
  # Setup with custom devices (using shorthand)
  python setup_models.py -w NPU -l GPU -i GPU
  
  # Download only (skip compilation)
  python setup_models.py --download-only
  python setup_models.py -d
  
  # Compile only (skip download, assumes models already exist)
  python setup_models.py --compile-only
  python setup_models.py -c
  
  # Compile only one image size
  python setup_models.py --single-size
  python setup_models.py -s
        """
    )
    
    parser.add_argument(
        "--whisper-device", "-w",
        choices=["CPU", "GPU", "NPU"],
        default="NPU",
        help="Device for Whisper model (default: NPU)"
    )
    parser.add_argument(
        "--llm-device", "-l",
        choices=["CPU", "GPU", "NPU"],
        default="GPU",
        help="Device for LLM model (default: GPU)"
    )
    parser.add_argument(
        "--image-device", "-i",
        choices=["CPU", "GPU", "NPU"],
        default="GPU",
        help="Device for Image Generation model (default: GPU)"
    )
    parser.add_argument(
        "--download-only", "-d",
        action="store_true",
        help="Only download models, skip compilation"
    )
    parser.add_argument(
        "--compile-only", "-c",
        action="store_true",
        help="Only compile models, skip download"
    )
    parser.add_argument(
        "--single-size", "-s",
        action="store_true",
        help="Compile image pipeline for single size (1024x1024) instead of multiple sizes"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = get_config()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure models.config exists in the same directory.")
        return 1
    
    print("=" * 70)
    print("VOICE-TO-IMAGE PIPELINE SETUP")
    print("=" * 70)
    print(f"Config file: {config.config_file}")
    print(f"Models directory: {config.models_dir}")
    print(f"Cache directory: {config.cache_dir}")
    print()
    print("Models to setup:")
    print(f"  • Whisper: {config.whisper_hf_id}")
    print(f"  • LLM: {config.llm_hf_id}")
    print(f"  • Image: {config.image_hf_id}")
    
    overall_start = time.time()
    download_success = True
    compile_success = True
    
    # Step 1: Download models
    if not args.compile_only:
        download_success = download_all_models(config)
        if not download_success:
            print("\n❌ Download failed. Aborting.")
            return 1
    
    # Step 2: Compile models
    if not args.download_only:
        compile_success = compile_all_models(
            config,
            args.whisper_device,
            args.llm_device,
            args.image_device,
            multi_size=not args.single_size
        )
        if not compile_success:
            print("\n⚠ Compilation had failures.")
            return 1
    
    overall_end = time.time()
    
    # Final summary
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print(f"Total setup time: {overall_end - overall_start:.2f} seconds")
    print(f"\nModels location: {config.models_dir.absolute()}")
    print(f"Cache location: {config.cache_dir.absolute()}")
    print("\nYou can now run the application:")
    print("  python voice-to-image-app.py")
    print("\nThe models will load quickly from cache!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
