# 🎤 Voice-to-Image Generation Pipeline

A multi-stage AI pipeline that converts voice input into generated images using OpenVINO GenAI.

## 🌟 Features

- **Voice Input**: Speak naturally to describe the image you want
- **Speech Recognition**: Whisper (configurable: CPU/GPU/NPU)
- **Prompt Enhancement**: LLaMA 3.2 3B Instruct (configurable: CPU/GPU/NPU)
- **Image Generation**: LCM-SDXL (configurable: CPU/GPU/NPU)
- **Device Flexibility**: Choose CPU, GPU, or NPU for each model independently
- **Model Caching**: Pre-compiled models for fast loading
- **Performance Metrics**: Detailed timing and system information
- **Centralized Configuration**: Single `models.config` file for all settings

## 📋 Pipeline Flow

```
🎙️ Voice Input → Whisper → LLM Enhancement → Image Generation → 🖼️ Output
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenVINO 2025.0.0 or later
- OpenVINO GenAI
- Intel NPU drivers (for NPU acceleration)
- Intel GPU drivers (for GPU acceleration)

### Setup 

The easiest way to get started is with the combined setup script:

- All models are configured in `models.config`:
- **To change models:** Simply edit the `*_hf_id` values and run `python setup_models.py` again. Local directory names are automatically generated from the model names.

```bash
# Clone or navigate to the project directory
git clone https://github.com/ravi9/voice-to-image-ov-genai-demo.git
cd voice-to-image-ov-genai-demo

# Create and activate virtual environment
python -m venv v2i-env
source v2i-env/bin/activate  # Linux/Mac
# or
v2i-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download AND pre-compile models in one step
python setup_models.py
```

This will:
1. Download all 3 models from HuggingFace (~14-15 GB)
2. Pre-compile them for your selected devices
3. Cache compiled models for fast loading
4. Test each model to ensure it works


### Running the Application

```bash
python voice-to-image-app.py
```

The application will start at: `http://localhost:7860`


## Usage
0. **Adjust Model Configuration and Generation Parameters if desired**
1. **Initialize Models**:
   - Click "🚀 Initialize Models" button
   - Wait for confirmation with loading times displayed

2. **Generate Images**:
   - Click `Record` to start recording. Example: *"A majestic lion standing on cliff at sunset"*
   - Click `Stop` to stop recording
   - Click "✨ Generate Image" button to generate image from your voice input enhanced by LLM.

## ⚙️ Advanced Configuration

### Setup Script Options

```bash
# Full setup with default devices
python setup_models.py

# Custom devices
python setup_models.py --whisper-device NPU --llm-device GPU --image-device GPU

# Download only (skip compilation)
python setup_models.py --download-only

# Compile only (skip download)
python setup_models.py --compile-only

# Single image size (faster setup)
python setup_models.py --single-size
```

### Pre-compilation Options

```bash
# Default: compile all models with multiple image sizes
python precompile_models.py

# Compile specific models only
python precompile_models.py --models whisper llm

# Single image size
python precompile_models.py --width 1024 --height 1024

# Multiple sizes from config
python precompile_models.py --models image --multi-size
```

### Model Caching

The application uses OpenVINO's model caching for fast loading:
- **First time**: Models are compiled and cached (10-30 minutes)
- **Subsequent loads**: Models load from cache (seconds)
- Separate cache for each device type (CPU/GPU/NPU)
- Image pipeline cached per size (512x512, 1024x1024, 1024x768)
