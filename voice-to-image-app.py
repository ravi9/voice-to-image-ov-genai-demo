#!/usr/bin/env python3
"""
Voice-to-Image Generation Pipeline using OpenVINO GenAI
Three-stage pipeline:
1. Speech-to-Text: Whisper (configurable device: CPU/GPU/NPU)
2. Prompt Enhancement: LLaMA 3.2 3B Instruct (configurable device: CPU/GPU/NPU)
3. Image Generation: LCM-SDXL (configurable device: CPU/GPU/NPU)

IMPORTANT: Run 'python setup_models.py' before first use!
This pre-compiles models and creates cache to avoid long loading times
and potential system crashes during interactive use.

Models are configured in models.config file.
"""

import gradio as gr  
import openvino_genai as ov_genai
import openvino as ov
import librosa  
import numpy as np  
from PIL import Image  
from pathlib import Path
import time
import random
import importlib.metadata
import platform
from config_reader import get_config
  
# Load configuration
config = get_config()

# Get OpenVINO version
try:
    openvino_version = importlib.metadata.version("openvino")
except Exception:
    openvino_version = ov.__version__

# Get OpenVINO GenAI version
try:
    openvino_genai_version = importlib.metadata.version("openvino-genai")
except Exception:
    openvino_genai_version = ov_genai.__version__

# Global variables to store pipelines  
whisper_pipe = None  
llm_pipe = None  
image_pipe = None
image_pipe_compiled = False
current_image_config = {"width": None, "height": None, "device": None}

def parse_image_size(size_string):
    """Parse image size string to width and height"""
    # Extract dimensions from strings like "512x512 (Square - Fast)"
    dimensions = size_string.split()[0]  # Get "512x512"
    width, height = map(int, dimensions.split('x'))
    return width, height

def get_image_size_options():
    """Generate image size dropdown options from config"""
    options = []
    labels = ["Square - Fast", "Square - Optimal", "Landscape"]
    for (width, height), label in zip(config.image_sizes, labels):
        options.append(f"{width}x{height} ({label})")
    return options
    cache_subdir.mkdir(exist_ok=True)
    return {"CACHE_DIR": str(cache_subdir)}
  
def initialize_models(whisper_device, llm_device, image_device, img_width=1024, img_height=1024):  
    """Initialize all three pipelines with selected devices"""  
    global whisper_pipe, llm_pipe, image_pipe, image_pipe_compiled, current_image_config
    
    timing_info = {}
    status_text = ""
    
    try:
        # Initialize Whisper for speech recognition
        status_msg = f"Loading Whisper model ({config.whisper_hf_id}) on {whisper_device}...\n"
        status_msg += f"Using cache directory: {config.get_cache_dir(whisper_device)}\n"
        print(status_msg.strip())
        status_text += status_msg
        yield status_text
        
        whisper_config = {"CACHE_DIR": str(config.get_cache_dir(whisper_device))}
        whisper_start = time.time()
        whisper_pipe = ov_genai.WhisperPipeline(str(config.whisper_path), whisper_device, **whisper_config)  
        whisper_end = time.time()
        timing_info['whisper_load'] = whisper_end - whisper_start
        
        status_msg = f"✓ Whisper loaded in {timing_info['whisper_load']:.2f} seconds\n\n"
        print(status_msg.strip())
        status_text += status_msg
        yield status_text
        
        # Initialize LLM for prompt enhancement
        status_msg = f"Loading LLM model ({config.llm_hf_id}) on {llm_device}...\n"
        status_msg += f"Using cache directory: {config.get_cache_dir(llm_device)}\n"
        print(status_msg.strip())
        status_text += status_msg
        yield status_text
        
        llm_config = {"CACHE_DIR": str(config.get_cache_dir(llm_device))}
        llm_start = time.time()
        llm_pipe = ov_genai.LLMPipeline(str(config.llm_path), llm_device, **llm_config)  
        llm_end = time.time()
        timing_info['llm_load'] = llm_end - llm_start
        
        status_msg = f"✓ LLM loaded in {timing_info['llm_load']:.2f} seconds\n\n"
        print(status_msg.strip())
        status_text += status_msg
        yield status_text
        
        # Initialize image generation pipeline with reshape and compile
        status_msg = f"Loading Image Generation model ({config.image_hf_id}) on {image_device}...\n"
        status_msg += f"Image dimensions: {img_width}x{img_height}\n"
        print(status_msg.strip())
        status_text += status_msg
        yield status_text
        
        image_load_start = time.time()
        # Load the pipeline without device specification initially
        image_pipe = ov_genai.Text2ImagePipeline(str(config.image_path))
        
        # Reshape the pipeline for the desired dimensions
        status_msg = f"Reshaping pipeline for {img_width}x{img_height}...\n"
        print(status_msg.strip())
        status_text += status_msg
        yield status_text
        
        guidance_scale = image_pipe.get_generation_config().guidance_scale
        image_pipe.reshape(1, img_height, img_width, guidance_scale)
        
        # Compile with cache configuration
        image_config = {"CACHE_DIR": str(config.get_cache_dir(image_device))}
        status_msg = f"Compiling pipeline on {image_device} with cache: {image_config['CACHE_DIR']}\n"
        print(status_msg.strip())
        status_text += status_msg
        yield status_text
        
        compile_start = time.time()
        
        # Compile with split devices (all on same device for simplicity)
        image_pipe.compile(image_device, image_device, image_device, config=image_config)
        
        compile_end = time.time()
        timing_info['image_compile'] = compile_end - compile_start
        timing_info['image_load_total'] = compile_end - image_load_start
        
        status_msg = f"✓ Image compile time: {timing_info['image_compile']:.2f} seconds\n"
        status_msg += f"✓ Image total load time: {timing_info['image_load_total']:.2f} seconds\n\n"
        print(status_msg.strip())
        status_text += status_msg
        yield status_text
        
        image_pipe_compiled = True
        current_image_config = {"width": img_width, "height": img_height, "device": image_device}
        
        total_time = timing_info['whisper_load'] + timing_info['llm_load'] + timing_info['image_load_total']
        
        summary = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ All models initialized successfully!
        
Model Loading Times:
• Whisper ({config.whisper_hf_id}): {timing_info['whisper_load']:.2f}s on {whisper_device}
• LLM ({config.llm_hf_id}): {timing_info['llm_load']:.2f}s on {llm_device}
• Image ({config.image_hf_id}): {timing_info['image_load_total']:.2f}s on {image_device}
  - Compile time: {timing_info['image_compile']:.2f}s
• Total initialization time: {total_time:.2f}s

Image pipeline configured for {img_width}x{img_height}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        status_text += summary
        print("\n" + summary)
        yield status_text
        
    except Exception as e:
        error_msg = f"✗ Error initializing models: {str(e)}"
        print(error_msg)
        yield error_msg

def process_audio_to_image(audio, whisper_device, llm_device, image_device, 
                          num_steps, guidance_scale, max_tokens, image_size_string):  
    """Complete pipeline: audio -> text -> enhanced prompt -> image"""  
    
    if audio is None:
        return "Please record audio first", "", None, ""
    
    # Parse image size from dropdown selection
    img_width, img_height = parse_image_size(image_size_string)
    
    # Check if image pipeline needs to be recompiled due to size or device change
    global image_pipe_compiled, current_image_config
    needs_recompile = (
        not image_pipe_compiled or
        current_image_config["width"] != img_width or
        current_image_config["height"] != img_height or
        current_image_config["device"] != image_device
    )
      
    # Step 1: Initialize models if not already done or if recompilation needed
    if whisper_pipe is None or llm_pipe is None or image_pipe is None or needs_recompile:
        if needs_recompile and image_pipe is not None:
            print(f"Recompiling image pipeline due to config change...")
        # Generator will yield status updates
        for status in initialize_models(whisper_device, llm_device, image_device, img_width, img_height):
            if "Error" in status:
                return status, "", None, ""
    
    timing_info = {}
    pipeline_start = time.time()
      
    try:
        # Step 2: Speech to text using Whisper
        print("Transcribing audio...")
        transcribe_start = time.time()
        
        sample_rate, audio_data = audio  
        
        # Resample to 16kHz if needed  
        if sample_rate != 16000:  
            audio_data = librosa.resample(  
                audio_data.astype(np.float32),   
                orig_sr=sample_rate,   
                target_sr=16000  
            )  
        
        # Normalize audio  
        if audio_data.dtype == np.int16:  
            audio_data = audio_data.astype(np.float32) / 32768.0  
        
        raw_speech = audio_data.tolist()  
        transcription_result = whisper_pipe.generate(raw_speech)  
        transcribed_text = transcription_result.texts[0]
        
        transcribe_end = time.time()
        timing_info['transcription'] = transcribe_end - transcribe_start
        
        transcription_msg = f"Transcription ({timing_info['transcription']:.2f}s):\n{transcribed_text}"
        print(f"Transcription: {transcribed_text}")
        print(f"Transcription time: {timing_info['transcription']:.2f}s")
        
        # Yield transcription immediately
        yield transcription_msg, "", None, ""
        
        # Step 3: Enhance prompt using LLM
        print("Enhancing prompt...")
        llm_start = time.time()
        
        enhancement_prompt = f"""You are an expert at creating detailed image generation prompts.   
Take this user input and transform it into a detailed, vivid image generation prompt with artistic details,   
lighting, style, and composition. Keep it concise but descriptive.  
  
User input: {transcribed_text}  
  
Enhanced image prompt:"""  
        
        enhanced_prompt = llm_pipe.generate(enhancement_prompt, max_new_tokens=max_tokens)
        
        llm_end = time.time()
        timing_info['prompt_enhancement'] = llm_end - llm_start
        
        enhancement_msg = f"Prompt Enhancement ({timing_info['prompt_enhancement']:.2f}s):\n{enhanced_prompt}"
        print(f"Enhanced prompt: {enhanced_prompt}")
        print(f"Prompt enhancement time: {timing_info['prompt_enhancement']:.2f}s")
        
        # Yield enhanced prompt
        yield transcription_msg, enhancement_msg, None, ""
        
        # Step 4: Generate image using enhanced prompt with random seed
        print("Generating image...")
        random_seed = random.randint(1, 100)
        print(f"Using random seed: {random_seed}")
        
        gen_start = time.time()
        image_tensor = image_pipe.generate(  
            enhanced_prompt,  
            num_inference_steps=num_steps,
            rng_seed=random_seed
        )  
        gen_end = time.time()
        timing_info['image_generation'] = gen_end - gen_start
        print(f"Image generation time: {timing_info['image_generation']:.2f}s")
        
        # Convert tensor to PIL Image  
        generated_image = Image.fromarray(image_tensor.data[0])
        
        pipeline_end = time.time()
        timing_info['total_pipeline'] = pipeline_end - pipeline_start
        
        print("Image generation complete!")
        
        # Create timing summary
        timing_summary = f"""⏱️ Performance Metrics:
        
Pipeline Execution:
• Transcription: {timing_info['transcription']:.2f}s
• Prompt Enhancement: {timing_info['prompt_enhancement']:.2f}s
• Image Generation: {timing_info['image_generation']:.2f}s ({num_steps} steps)
• Total Pipeline Time: {timing_info['total_pipeline']:.2f}s

Models & Devices:
• Whisper: {whisper_device}
• LLM: {llm_device}
• Image: {image_device}

Generation Settings:
• Image: {img_width}x{img_height}
• Steps: {num_steps}
• Guidance: {guidance_scale}
• Seed: {random_seed}"""
        
        yield transcription_msg, enhancement_msg, generated_image, timing_summary
        
    except Exception as e:
        error_msg = f"Error during processing: {str(e)}"
        print(error_msg)
        yield error_msg, "", None, ""

def initialize_models_wrapper(whisper_device, llm_device, image_device, image_size_string):
    """Wrapper to parse image size and call initialize_models"""
    img_width, img_height = parse_image_size(image_size_string)
    # Consume the generator and yield each status update
    for status in initialize_models(whisper_device, llm_device, image_device, img_width, img_height):
        yield status
  
# Create Gradio interface  
with gr.Blocks(title="Voice-to-Image Generation", theme=gr.themes.Soft()) as demo:
    # Header row with Introduction and System Info
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("""
            # Voice-to-Image Generation Pipeline
            ### Powered by OpenVINO GenAI
            A multi-stage AI pipeline that converts voice input into generated images using OpenVINO GenAI.
            
            **Pipeline Flow:**
            🎙️ Voice Input → Whisper → LLM Enhancement → Image Generation → 🖼️ Output
            
            ### Steps
            0. **Adjust Model Configuration and Generation Parameters if desired**
            1. **Initialize Models**:
            - Click "🚀 Initialize Models" button
            - Wait for confirmation with loading times displayed

            2. **Generate Images**:
            - Click `Record` to start recording
            - Click `Stop` to stop recording
            - Click "✨ Generate Image" button to generate image from your voice input enhanced by LLM.
            """)
        
        with gr.Column(scale=1):
            # Get hardware info
            core = ov.Core()
            available_devices = core.available_devices
            device_info = []
            for device in available_devices:
                try:
                    device_name = core.get_property(device, "FULL_DEVICE_NAME")
                    device_info.append(f"• {device}: {device_name}")
                except:
                    device_info.append(f"• {device}")
            
            devices_str = "  \n".join(device_info) if device_info else "• No devices detected"
            
            # Get memory info
            try:
                import psutil
                mem = psutil.virtual_memory()
                memory_str = f"• Memory: {mem.total / (1024**3):.1f} GB (Available: {mem.available / (1024**3):.1f} GB)"
            except:
                memory_str = "• Memory: N/A"
            
            # Get OS distribution info
            os_info = platform.system()
            if os_info == "Linux":
                try:
                    import distro
                    os_info = f"{distro.name()} {distro.version()}"
                except:
                    # Fallback if distro module not available
                    try:
                        with open('/etc/os-release') as f:
                            lines = f.readlines()
                            for line in lines:
                                if line.startswith('PRETTY_NAME='):
                                    os_info = line.split('=')[1].strip().strip('"')
                                    break
                    except:
                        os_info = f"Linux {platform.release()}"
            elif os_info == "Windows":
                os_info = f"Windows {platform.release()}"
            elif os_info == "Darwin":
                os_info = f"macOS {platform.mac_ver()[0]}"
            
            system_info = f"""### 💻 System Information

**Software:**  
• OpenVINO: {openvino_version}  
• OpenVINO GenAI: {openvino_genai_version}  
• Python: {platform.python_version()}  
• OS: {os_info} ({platform.machine()})

**Hardware:**  
{memory_str}  
{devices_str}

**Models:**  
• Whisper: {config.whisper_hf_id.split('/')[-1]}  
• LLM: {config.llm_hf_id.split('/')[-1]}  
• Image: {config.image_hf_id.split('/')[-1]}"""
            
            gr.Markdown(system_info)
    
    # Row 1: Model Configuration, Generation Parameters, Initialize & Status
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Model Configuration")  
            
            # Device selection (use config defaults)
            whisper_device = gr.Dropdown(  
                choices=["CPU", "GPU", "NPU"],   
                value=config.whisper_default_device,   
                label="Whisper Device",
                info="Speech Recognition"
            )  
            llm_device = gr.Dropdown(  
                choices=["CPU", "GPU", "NPU"],   
                value=config.llm_default_device,   
                label="LLM Device",
                info="Prompt Enhancement"
            )  
            image_device = gr.Dropdown(  
                choices=["CPU", "GPU", "NPU"],   
                value=config.image_default_device,   
                label="Image Device",
                info="Image Generation"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Parameters")
            
            # Get image size options from config
            size_options = get_image_size_options()
            image_size = gr.Dropdown(
                choices=size_options,
                value=size_options[1],
                label="Image Size",
                info="Pre-compiled sizes"
            )
            
            num_steps = gr.Slider(
                minimum=1, 
                maximum=50, 
                value=4, 
                step=1,
                label="Inference Steps",
                info="LCM works best with 2-8"
            )
            guidance_scale = gr.Slider(
                minimum=1.0, 
                maximum=15.0, 
                value=7.5, 
                step=0.5,
                label="Guidance Scale"
            )
            max_tokens = gr.Slider(
                minimum=50,
                maximum=300,
                value=100,
                step=10,
                label="Max Tokens",
                info="Prompt enhancement length"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Initialize & Status")
            init_btn = gr.Button("🚀 Initialize Models", variant="secondary", size="lg")
            init_status = gr.Textbox(label="Status", interactive=False, lines=10, max_lines=15)
            gr.Markdown("*Models cached in `ov-cache/`*")
    
    # Row 2: Voice Input, Processing Results, Performance Metrics
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Voice Input")  
            audio_input = gr.Audio(  
                sources=["microphone"],   
                type="numpy",  
                label="Record Your Voice"
            )  
            gr.Markdown("*Describe the image you want.  \n Example:* A majestic lion standing on a cliff during sunset, digital art")
            generate_btn = gr.Button("✨ Generate Image", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Processing Results")
            transcription_output = gr.Textbox(
                label="🎤 Transcribed Text",
                placeholder="Your speech will appear here...",
                lines=4
            )
            enhanced_prompt_output = gr.Textbox(
                label="✨ Enhanced Prompt",
                placeholder="AI-enhanced prompt will appear here...",
                lines=5
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### ⏱️ Performance Metrics")
            timing_output = gr.Textbox(
                label="Timing & Config Info",
                placeholder="Performance metrics will appear here...",
                lines=10
            )
    
    # Row 3: Generated Image
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🖼️ Generated Image")
            image_output = gr.Image(label="Generated Image", type="pil", height=512)  
      
    # Event handlers  
    init_btn.click(  
        fn=initialize_models_wrapper,  
        inputs=[whisper_device, llm_device, image_device, image_size],  
        outputs=init_status  
    )  
      
    generate_btn.click(  
        fn=process_audio_to_image,  
        inputs=[
            audio_input, 
            whisper_device, 
            llm_device, 
            image_device,
            num_steps,
            guidance_scale,
            max_tokens,
            image_size
        ],  
        outputs=[transcription_output, enhanced_prompt_output, image_output, timing_output]  
    )  
  
if __name__ == "__main__":
    print("=" * 70)
    print("VOICE-TO-IMAGE GENERATION APP")
    print("=" * 70)
    
    # System Information
    print("\n📊 System Information:")
    print(f"• Platform: {platform.system()} {platform.release()}")
    print(f"• Python: {platform.python_version()}")
    print(f"• OpenVINO: {openvino_version}")
    print(f"• OpenVINO GenAI: {openvino_genai_version}")
    
    # Hardware Information
    print("\n🔧 Available Hardware:")
    core = ov.Core()
    available_devices = core.available_devices
    print(f"• Devices: {', '.join(available_devices)}")
    
    for device in available_devices:
        try:
            device_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"  - {device}: {device_name}")
        except:
            print(f"  - {device}")
    
    # Model Configuration
    print("\n📦 Model Configuration:")
    print(f"• Whisper: {config.whisper_hf_id}")
    print(f"• LLM: {config.llm_hf_id}")
    print(f"• Image: {config.image_hf_id}")
    
    print(f"\n📁 Paths:")
    print(f"• Config file: {config.config_file}")
    print(f"• Models directory: {config.models_dir}")
    print(f"• Cache directory: {config.cache_dir}")
    
    # Check if models are downloaded
    if not config.models_dir.exists():
        print("\n⚠️  WARNING: Models directory not found!")
        print("   Please run: python download_models.py")
    
    # Check if cache exists (warning if not)
    cache_exists = any(config.cache_dir.glob("*cache"))
    if not cache_exists:
        print("\n⚠️  WARNING: No model cache found!")
        print("   First-time model loading can take 20-30 minutes and may crash the system.")
        print("   It is STRONGLY RECOMMENDED to run the pre-compilation script first:")
        print("\n   python precompile_models.py")
        print("\n   This will compile and cache models offline for fast app loading.")
        print("\n   Continue anyway? (y/n): ", end="")
        
        import sys
        response = input().strip().lower()
        if response != 'y':
            print("\n   Exiting. Please run 'python precompile_models.py' first.")
            sys.exit(0)
        else:
            print("\n   ⚠️  Proceeding without cache. This may take a very long time...")
    else:
        print(f"\n✓ Model cache found. Models will load quickly from cache.")
    
    print("\n" + "="*70)
    print("Starting Gradio interface...")
    print("="*70 + "\n")
    
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)