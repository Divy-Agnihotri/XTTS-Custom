 # Hindi Text-to-Speech System

 An end-to-end research and development project for building a **Hindi Text-to-Speech (TTS) system** for enterprise multimedia applications.

 The project covers the investigation and evaluation of modern speech-synthesis architectures, Hindi speech dataset preparation, XTTS-v2 speaker adaptation and fine-tuning, speech-quality evaluation, multi-speaker narration, and development of a desktop inference application.

 The repository contains selected implementation code and experimentation scripts developed during the project.

---

 ## Overview

 The objective of this project was to investigate and develop a practical Hindi speech-synthesis pipeline capable of producing natural, consistent, and high-quality speech for applications such as:

 - Hindi narration and audiobooks
- Multimedia content production
- Character-based dialogue
- Voice cloning and speaker adaptation
- Enterprise content generation
- Promotional and narrated media
- Future expressive and emotion-controlled speech generation

 The work progressed from **TTS architecture research and dataset analysis** to **model adaptation, inference, application development, and production-oriented experimentation**.

---

 ## Project Pipeline

```
                    TTS Research
                         │
                         ▼
          ┌──────────────────────────┐
          │ TTS Architecture Study  │
          │                          │
          │ XTTS-v2                  │
          │ Bark                     │
          │ StyleTTS                 │
          │ Indic Parler-TTS         │
          │ VibeVoice / IndexTTS     │
          └────────────┬─────────────┘
                       │
                       ▼
              Hindi Dataset Research
                       │
                       ▼
          Dataset Preparation & Cleanup
                       │
                       ▼
             Audio / Metadata Conversion
                       │
                       ▼
              XTTS-v2 Fine-Tuning
                       │
                       ▼
             Hindi Speaker Adaptation
                       │
                       ▼
                Speech Evaluation
                       │
          ┌────────────┴─────────────┐
          │                          │
          ▼                          ▼
   Single Speaker TTS         Multi-Speaker TTS
          │                          │
          └────────────┬─────────────┘
                       ▼
              Desktop TTS Application
                       │
                       ▼
             Narration / Audiobooks
                       │
                       ▼
          Multimedia / Production Use
```

---

 ## Key Components

 ### 1\. TTS Architecture Research

 Multiple modern TTS architectures and approaches were investigated to understand their suitability for Hindi speech generation.

 The research included:

 - Transformer-based speech synthesis
- Diffusion-based speech synthesis
- Multilingual TTS
- Speaker-conditioned TTS
- Voice cloning
- Expressive speech synthesis
- Long-form speech generation
- Neural vocoder architectures

 Architectures and frameworks investigated included:

 - **XTTS-v2**
- **Bark**
- **StyleTTS**
- **Indic Parler-TTS**
- **VibeVoice**
- **IndexTTS**

 XTTS-v2 was selected for focused experimentation because of its multilingual capabilities, speaker conditioning, voice adaptation capabilities, and suitability for custom-speaker synthesis.

---

 # Dataset Preparation

 A major part of the project involved investigating and preparing Hindi speech datasets for TTS experimentation.

 The datasets were evaluated based on:

 - Recording quality
- Speaker consistency
- Environmental conditions
- Transcript quality
- Transcript/audio alignment
- Linguistic diversity
- Audio duration
- Speaker variation
- Annotation consistency

 Potential Hindi speech resources investigated included:

 - Mozilla Common Voice Hindi
- AI4Bharat speech resources
- Other publicly available Hindi speech datasets

---

 ## Parquet to LJSpeech Conversion

 One of the included preprocessing scripts converts a Parquet-based speech dataset into an **LJSpeech-compatible dataset structure**.

 The conversion pipeline performs:

```
Parquet Dataset
      │
      ├── Extract transcript
      │
      ├── Extract audio bytes
      │
      ▼
Audio decoding
      │
      ▼
Mono conversion
      │
      ▼
22.05 kHz resampling
      │
      ▼
WAV files
      │
      ▼
LJSpeech metadata
```

 The resulting dataset structure is:

```
ljspeech_output/
├── wavs/
│   ├── 000000.wav
│   ├── 000001.wav
│   ├── 000002.wav
│   └── ...
│
└── metadata.txt
```

 The metadata follows the LJSpeech convention:

```
filename.wav|transcript|normalized_transcript
```

 This format allows the dataset to be subsequently used by the Coqui TTS data-loading and training pipeline.

---

 # XTTS-v2 Fine-Tuning

 The project includes a training pipeline for adapting **Coqui XTTS-v2** to a custom Hindi speaker dataset.

 The training process starts from a pretrained XTTS-v2 checkpoint rather than training a TTS model from scratch.

```
Pretrained XTTS-v2
        │
        ▼
Hindi Speech Dataset
        │
        ▼
Dataset Conversion
        │
        ▼
Train / Evaluation Split
        │
        ▼
XTTS-v2 Fine-Tuning
        │
        ▼
Speaker-Adapted Model
```

 ### Training configuration

 The included training configuration uses:

 - Hindi language (`hi`)
- Custom speaker identification
- Training/evaluation dataset split
- XTTS-v2 pretrained checkpoint
- AdamW optimizer
- Mixed precision
- Gradient accumulation
- Small batch size for limited VRAM environments
- Configurable learning rate
- Checkpoint saving
- TensorBoard logging

 Example configuration:

```
BATCH_SIZE = 1
GRAD_ACCUM = 8
NUM_EPOCHS = 10
LANGUAGE = "hi"
```

 Gradient accumulation is used to provide a larger effective batch size while keeping the instantaneous GPU memory requirement low.

---

 # GPU / VRAM Optimization

 The training configuration was designed to support experimentation on GPUs with relatively limited VRAM.

 Several techniques were used:

 - Small batch size
- Gradient accumulation
- Mixed-precision training
- Reduced data-loader workers
- Reduced batch grouping
- CUDA memory configuration
- Periodic checkpointing
- Explicit GPU cache cleanup

 For example:

```
BATCH_SIZE = 1
GRAD_ACCUM = 8
mixed_precision = True
```

 These settings were particularly useful when experimenting with XTTS-v2 on consumer-grade GPUs.

 Actual hardware requirements depend on the XTTS version, dataset characteristics, sequence lengths, and training configuration.

---

 # Speech Generation Application

 The repository also contains a **PyQt6 desktop application** for XTTS-v2 inference.

 The application provides a graphical interface for generating speech without requiring users to interact directly with the Python inference code.

 The application supports two primary modes:

```
┌───────────────────────────────┐
│       XTTS-Custom GUI         │
├───────────────────────────────┤
│                               │
│  Single Line                  │
│                               │
│  Play Mode                    │
│                               │
│  Temperature                  │
│  Speed                        │
│  Top-K                        │
│  Top-P                        │
│  Repetition Penalty           │
│  Line Pause                   │
│                               │
└───────────────────────────────┘
```

---

 ## Single Line Mode

 Single Line mode allows a user to:

 1. Select a reference voice.
2. Select a language.
3. Enter text.
4. Adjust generation parameters.
5. Generate speech.
6. Save the generated result as a WAV file.

 The system extracts speaker conditioning information from the reference audio and uses it during XTTS-v2 inference.

 Supported languages configured in the application include:

```
Hindi
English
French
German
Spanish
Italian
Japanese
Korean
Portuguese
Russian
Chinese
```

---

 # Long-Form Text Generation

 Long text is divided into smaller chunks before inference.

 This prevents excessively long input sequences from being passed directly to the model and provides better control over long-form generation.

 The general process is:

```
Long Text
   │
   ▼
Sentence Detection
   │
   ▼
Text Chunking
   │
   ▼
XTTS Inference
   │
   ▼
Audio Concatenation
   │
   ▼
Final WAV
```

 The chunking system recognizes common sentence-ending punctuation, including Hindi danda (`।`) as well as English punctuation.

---

 # Multi-Character Play Mode

 One of the application features is a **multi-character dialogue generation mode**.

 A play can be represented using JSON containing:

 - Characters
- Dialogue lines
- Speaker information

 Conceptually:

```
{
    "characters": [
        "Narrator",
        "Character A",
        "Character B"
    ],
    "play": [
        {
            "speaker": "Narrator",
            "dialogue": "..."
        },
        {
            "speaker": "Character A",
            "dialogue": "..."
        },
        {
            "speaker": "Character B",
            "dialogue": "..."
        }
    ]
}
```

 Each character can be assigned a separate reference voice.

 The application then:

```
Character A → Reference Voice A
Character B → Reference Voice B
Narrator    → Reference Voice C
                      │
                      ▼
                   XTTS-v2
                      │
                      ▼
             Complete Dialogue
                      │
                      ▼
                  output.wav
```

 Speaker conditioning latents are calculated once per character and cached for reuse throughout the generation process.

 This reduces unnecessary repeated conditioning computation during long multi-character generation.

---

 # Inference Controls

 The desktop application exposes several XTTS generation parameters.

 ### Temperature

 Controls the degree of sampling variation during generation.

 ### Speed

 Controls the generated speech rate.

 ### Top-K

 Controls the number of highest-probability candidates considered during sampling.

 ### Top-P

 Controls nucleus sampling.

 ### Repetition Penalty

 Helps reduce undesirable repeated tokens or phrases.

 ### Line Pause

 Adds configurable silence between dialogue lines in Play Mode.

 These parameters allow the user to experiment with the trade-offs between speech consistency, expressiveness, speed, and generation stability.

---

 # Speaker Conditioning

 The application uses XTTS-v2 speaker conditioning based on reference audio.

 The basic inference process is:

```
gpt_cond_latent, speaker_embedding = \
    model.get_conditioning_latents(
        audio_path=[reference_audio]
    )
```

 The resulting conditioning information is then passed into the XTTS inference process together with the requested text and language.

 This enables zero-/few-shot style speaker conditioning using reference recordings.

---

 # Speech Quality Evaluation

 During the project, synthesized speech was evaluated using several qualitative and practical criteria.

 ### Naturalness

 Whether generated speech sounds natural and human-like.

 ### Pronunciation

 Particular attention was given to Hindi pronunciation and linguistic correctness.

 ### Fluency

 Evaluation of sentence-level fluency and intelligibility.

 ### Speaker Consistency

 Whether generated speech consistently preserves the identity and characteristics of the reference speaker.

 ### Prosody

 Evaluation of:

 - Intonation
- Rhythm
- Pacing
- Pauses
- Vocal energy
- Delivery style

 ### Long-Form Stability

 Long-form generation was evaluated for:

 - Repetition
- Unnatural pauses
- Pronunciation degradation
- Rhythm changes
- Speaker drift
- Generation failures

---

 # Observed Challenges

 The experiments highlighted several challenges common to modern speech synthesis systems.

 These included:

 - Unnatural pauses
- Pronunciation irregularities
- Inconsistent prosody
- Speaker consistency issues
- Long-form generation instability
- Limited expressive control
- Difficulty representing emotional intensity
- Dataset annotation requirements
- Variability in source recording quality

 These observations motivated further research into controllable and expressive speech synthesis.

---

 # Expressive and Emotional TTS Research

 A later phase of the project focused on improving expressive speech generation.

 The research investigated how speech characteristics can be controlled through parameters such as:

```
Emotion
   +
Intensity
   +
Pacing
   +
Vocal Energy
   +
Delivery Style
   +
Prosody
```

 Potential emotional categories investigated included:

 - Happiness
- Sadness
- Anger
- Excitement
- Fear
- Calmness
- Dramatic expression

 The project also explored the concept of **continuous emotional intensity**, rather than restricting generation to simple categorical emotion labels.

 For example:

```
Calm ────────────────► Excited
0%                       100%
```

 This represents a future direction for more granular expressive control.

---

 # Emotional Dataset Research

 Creating high-quality emotional speech datasets requires additional annotation beyond ordinary TTS datasets.

 The project investigated:

 - Emotion labels
- Emotional intensity
- Speech style
- Vocal energy
- Pacing
- Delivery characteristics

 Both automated and manual annotation approaches were investigated.

 Potential automated annotation pipelines included:

```
Speech Audio
     │
     ▼
ASR / Speech-to-Text
     │
     ▼
Transcript
     │
     ├──────────────┐
     ▼              ▼
Audio Analysis   Emotion Model
     │              │
     └──────┬───────┘
            ▼
      Dataset Metadata
```

 Manual annotation was also considered because human annotation can provide higher-quality labels for subtle emotional characteristics.

---

 # Director-Guided TTS

 An additional conceptual direction developed during the project was a **director-guided speech synthesis framework**.

 The goal is to allow a creator to specify how a line should be delivered rather than simply providing text.

 Potential controls include:

```
Text
  +
Speaker
  +
Emotion
  +
Emotion Intensity
  +
Pacing
  +
Vocal Energy
  +
Delivery Style
        │
        ▼
   TTS Generation
```

 This approach is intended for applications such as:

 - Audiobooks
- Character dialogue
- Advertisements
- Games
- Films
- Interactive media
- Enterprise narration

---

 # Production Applications

 The developed TTS system was evaluated beyond isolated experiments and explored for practical multimedia applications.

 Applications included:

 ### Audiobook Narration

 Generation of narrated audiobook chapters using synthesized voices.

 ### Character Narration

 Generation of dialogue using different reference voices for different characters.

 ### Multimedia Content

 Synthesized narration was integrated into multimedia and promotional content workflows.

 ### Voice Adaptation

 Additional voices were evaluated for potential character-based voice applications.

---

 # Application Architecture

 The overall implementation can be viewed as several layers.

```
┌──────────────────────────────────┐
│          User Interface          │
│       Desktop / Web Frontend     │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│          TTS Inference           │
│                                  │
│            XTTS-v2               │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│       Speaker Conditioning       │
│                                  │
│   Reference Audio / Embeddings   │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│           Audio Output           │
│                                  │
│        WAV / Multimedia          │
└──────────────────────────────────┘
```

 The broader application development also investigated web-based interfaces and standalone desktop deployment for non-technical users.

---

 # Repository Structure

 The repository can be organized into the following components:

```
.
├── dataset/
│   └── parquet_to_ljspeech.py
│
├── training/
│   └── xtts_finetune.py
│
├── inference/
│   └── xtts_gui.py
│
├── examples/
│   └── play.json
│
└── README.md
```

 > The exact filenames may differ depending on how the scripts are organized in the repository.

---

 # Installation

 The project is based primarily on Python and the following ecosystem:

 - Python
- PyTorch
- Coqui TTS
- XTTS-v2
- PyQt6
- Pandas
- PyArrow
- SoundFile
- Pydub
- Torchaudio

 A typical environment can be created using:

```
python -m venv venv
```

 Activate the environment:

 ### Windows

```
venv\Scripts\activate
```

 Install the required dependencies according to the versions used by the specific XTTS-v2 environment.

 Example:

```
pip install torch torchaudio
pip install TTS
pip install pandas pyarrow soundfile pydub
pip install PyQt6
```

 For GPU inference/training, install a PyTorch build compatible with the installed CUDA environment.

---

 # Dataset Preparation

 After obtaining a compatible speech dataset, the dataset conversion script can be used to create WAV files and LJSpeech-style metadata.

 The input dataset is expected to contain fields equivalent to:

```
sentence
audio.bytes
```

 The conversion script:

 1. Reads the Parquet file.
2. Extracts transcripts.
3. Extracts audio bytes.
4. Decodes the audio.
5. Converts audio to mono.
6. Resamples audio to 22.05 kHz.
7. Writes WAV files.
8. Generates metadata.

 The resulting dataset can then be prepared for XTTS-v2 fine-tuning.

---

 # XTTS-v2 Fine-Tuning

 Before running the training script, configure the paths to the local XTTS-v2 checkpoint and dataset.

 The training configuration requires the XTTS-v2 model files, including components such as:

```
model.pth
vocab.json
dvae.pth
mel_stats.pth
config.json
```

 The training script verifies that the required files exist before starting training.

 Training output is stored in the configured output directory.

---

 # Desktop Inference

 The PyQt6 application loads the XTTS-v2 model and automatically selects CUDA when available:

```
self.device = "cuda" if torch.cuda.is_available() else "cpu"
```

 Run the application using:

```
python xtts_gui.py
```

 The application provides:

 - Reference audio selection
- Language selection
- Text input
- Voice conditioning
- Generation controls
- Single-line synthesis
- Multi-character play generation
- WAV output

---

 # Example Workflow

 A typical workflow for creating a custom Hindi voice is:

```
1. Obtain Hindi speech dataset
              ↓
2. Inspect audio and transcripts
              ↓
3. Convert dataset to required format
              ↓
4. Create training/evaluation split
              ↓
5. Fine-tune XTTS-v2
              ↓
6. Evaluate pronunciation and naturalness
              ↓
7. Select reference speaker
              ↓
8. Run inference
              ↓
9. Generate narration
              ↓
10. Validate output for production use
```

---

 # Research Timeline

 The project developed through several major phases.

 ### Phase 1 — TTS Research

 Investigation of modern TTS architectures, including transformer, diffusion, multilingual, speaker-conditioned, and expressive approaches.

 ### Phase 2 — Architecture Evaluation

 Experimental comparison of multiple open-source TTS frameworks based on speech quality, Hindi support, inference performance, speaker consistency, and deployment complexity.

 ### Phase 3 — Dataset Research

 Investigation and evaluation of publicly available Hindi speech resources, followed by preprocessing and metadata preparation.

 ### Phase 4 — Model Adaptation

 Fine-tuning experiments with XTTS-v2 and evaluation of improvements in Hindi pronunciation, fluency, naturalness, rhythm, and speaker consistency.

 ### Phase 5 — Expressive TTS Research

 Investigation of emotion conditioning, prosody, vocal energy, pacing, emotional intensity, and speech-style control.

 ### Phase 6 — Production Experimentation

 Use of the optimized TTS system for audiobook narration, character voices, and multimedia content generation.

 ### Phase 7 — Application Development

 Development of graphical and application-level interfaces for making speech generation accessible to non-technical users.

 ### Phase 8 — Future Research

 Investigation of controllable/director-guided TTS, self-improving datasets, emotional speech control, and AI-assisted background music generation and synchronization.

---

 # Research Areas

 The project covered the following major research areas:

 - Hindi Text-to-Speech
- Multilingual TTS
- Transformer-based speech synthesis
- Diffusion-based speech synthesis
- Speaker embeddings
- Voice cloning
- Neural vocoders
- Prosody modeling
- Expressive speech synthesis
- Emotion-conditioned speech synthesis
- Long-form speech generation
- Dataset preprocessing
- Speech dataset annotation
- Speaker adaptation
- TTS model fine-tuning
- TTS inference optimization
- Multimedia narration
- Character-based voice generation

---

 # Limitations

 This repository represents selected research and implementation work rather than a complete commercial production system.

 Important considerations include:

 - TTS quality depends heavily on dataset quality.
- Speaker similarity varies with reference-audio quality.
- Long-form synthesis may require additional post-processing.
- Emotional control is an ongoing research area.
- Different XTTS/TTS versions may require different dependency versions.
- GPU memory requirements vary depending on configuration.
- Some components of the broader enterprise system are not included in this repository.
- Production deployment may require additional security, monitoring, API, and infrastructure components.

---

 # Future Work

 Potential future improvements include:

 - Fine-grained emotional control
- Continuous emotion-intensity conditioning
- Improved Hindi prosody
- Better long-form narration stability
- Automated emotional dataset annotation
- Human-in-the-loop dataset refinement
- More robust speaker adaptation
- Improved character voice management
- Web-based TTS interfaces
- Scalable inference services
- Director-guided speech generation
- AI-assisted background-music generation
- Automatic speech/music synchronization
- Improved audiobook production pipelines

---

 # Responsible Use

 Voice cloning and speaker-conditioned TTS technologies can reproduce characteristics of real human voices.

 Users of this repository should ensure that they have appropriate permission to use any voice recordings and should avoid using synthesized voices for impersonation, fraud, deception, or other unauthorized purposes.

 Dataset and model licenses should also be reviewed before using them for commercial applications.

---

 # Acknowledgements

 This project builds upon research and open-source technologies from the broader speech-synthesis community, including the developers and researchers behind:

 - Coqui TTS
- XTTS-v2
- PyTorch
- Mozilla Common Voice
- AI4Bharat
- Other open-source TTS and speech-processing projects investigated during the research

 Please refer to the respective projects and model/dataset licenses before redistribution or commercial deployment.

---

 # Disclaimer

 This repository is intended primarily as a **technical record of research, experimentation, and implementation work performed during the development of an enterprise Hindi TTS system**.

 Not every research direction described in this README is necessarily implemented in the code contained in this repository. Some sections document experiments, investigations, architectural studies, production learnings, or future research directions.

 The repository should therefore be viewed as a combination of:

```
Research
+
Experimentation
+
Dataset Engineering
+
Model Fine-Tuning
+
Inference Development
+
Application Development
+
Future R&D
```

 rather than as a single standalone TTS framework.

---

 ## Project Summary

 The project progressed from researching modern speech-synthesis architectures to developing a practical Hindi TTS pipeline based on XTTS-v2.

 The resulting workflow combines **Hindi dataset preparation, speaker adaptation, model fine-tuning, configurable inference, long-form generation, multi-character narration, and application development**, while providing a foundation for future work in expressive and controllable speech synthesis.

```
Research
   ↓
Hindi Dataset Engineering
   ↓
XTTS-v2 Fine-Tuning
   ↓
Speaker Adaptation
   ↓
Speech Evaluation
   ↓
Inference Engine
   ↓
Multi-Character Narration
   ↓
Desktop / Web Applications
   ↓
Enterprise Multimedia Applications
```
