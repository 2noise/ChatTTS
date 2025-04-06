# ChatTTS-TF
A TensorFlow implementation of ChatTTS, a generative speech model for daily dialogue.

[![License](https://img.shields.io/github/license/loda616/ChatTTS-TF?style=for-the-badge)]([https://github.com/loda616/ChatTTS-TF/blob/main/LICENSE](https://github.com/loda616/ChatTTS/blob/TF-conversion/LICENSE))
[![PyPI](https://img.shields.io/pypi/v/ChatTTS-TF.svg?style=for-the-badge&color=green)](https://pypi.org/project/ChatTTS-TF)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/loda616/ChatTTS-TF/blob/main/examples/ipynb/colab.ipynb)

**English** | [**简体中文**](docs/cn/README.md) | [**日本語**](docs/jp/README.md) | [**Русский**](docs/ru/README.md) | [**Español**](docs/es/README.md) | [**Français**](docs/fr/README.md) | [**한국어**](docs/kr/README.md)

## Introduction
> [!Note]
> This is a TensorFlow implementation of the ChatTTS model, forked from [2noise/ChatTTS](https://github.com/2noise/ChatTTS).

> [!Tip]
> For the extended end-user products, please refer to the index repo [Awesome-ChatTTS](https://github.com/libukai/Awesome-ChatTTS/tree/en) maintained by the community.

ChatTTS-TF is a TensorFlow implementation of the ChatTTS model, designed specifically for dialogue scenarios such as LLM assistant. This implementation provides mobile-friendly features and optimized performance for TensorFlow environments.

### Supported Languages
- [x] English
- [x] Chinese
- [ ] Coming Soon...

### Highlights
1. **Mobile-Optimized**: Optimized for mobile and edge devices using TensorFlow Lite
2. **Cross-Platform**: Supports Android and Flutter applications
3. **Efficient Inference**: Optimized for real-time speech generation
4. **Fine-grained Control**: Maintains the original model's control over prosodic features

### Model Architecture
The TensorFlow implementation includes:
- TFLite model conversion
- Mobile-optimized inference
- Cross-platform compatibility
- Efficient memory management

## Installation

### 1. Install from PyPI
```bash
pip install ChatTTS-TF
```

### 2. Install from GitHub
```bash
pip install git+https://github.com/loda616/ChatTTS-TF
```

### 3. Install from local directory
```bash
git clone https://github.com/loda616/ChatTTS-TF
cd ChatTTS-TF
pip install -e .
```

## Usage

### Basic Usage
```python
import chattts_tf

# Initialize the model
model = chattts_tf.ChatTTSTensorFlowWrapper()
model.load()

# Generate speech
text = "Hello, this is a test."
audio = model.generate(text)

# Save the audio
chattts_tf.save_audio(audio, "output.wav")
```

### Mobile Integration

#### Android
```kotlin
// Initialize the model
val model = ChatTTSTensorFlowWrapper(context)
model.load()

// Generate speech
val text = "Hello, this is a test."
val audio = model.generate(text)

// Play the audio
audioPlayer.play(audio)
```

#### Flutter
```dart
// Initialize the model
final model = ChatTTSTensorFlowWrapper();
await model.load();

// Generate speech
final text = "Hello, this is a test.";
final audio = await model.generate(text);

// Play the audio
audioPlayer.play(audio);
```

## Project Structure
```
ChatTTS-TF/
├── chattts_tf/           # Main package directory
│   ├── models/          # TFLite models
│   ├── utils/           # Utility functions
│   └── wrappers/        # Platform-specific wrappers
├── examples/            # Example implementations
│   ├── android/        # Android example
│   └── flutter/        # Flutter example
├── docs/               # Documentation
│   ├── android/       # Android integration guide
│   └── flutter/       # Flutter integration guide
└── tests/             # Test files
```

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Original ChatTTS model by [2noise](https://github.com/2noise/ChatTTS)
- TensorFlow team for the excellent framework
- All contributors who have helped with this implementation 
