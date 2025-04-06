# Flutter Integration Guide

This guide explains how to integrate the ChatTTS TensorFlow model into your Flutter application.

## Setup

### 1. Add Dependencies

Add the following to your `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  tflite_flutter: ^0.10.4
  just_audio: ^0.9.36
  path_provider: ^2.1.2
  permission_handler: ^11.0.1
```

### 2. Add Permissions

For Android, add to `android/app/src/main/AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
```

For iOS, add to `ios/Runner/Info.plist`:
```xml
<key>NSMicrophoneUsageDescription</key>
<string>This app needs access to microphone for audio playback</string>
```

## Implementation

### 1. Model Service

Create a service to handle the TensorFlow model:

```dart
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';

class ChatTTSService {
  late Interpreter _interpreter;
  bool _isInitialized = false;

  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      _interpreter = await Interpreter.fromAsset('assets/chattts_model.tflite');
      _isInitialized = true;
    } catch (e) {
      print('Error initializing model: $e');
      rethrow;
    }
  }

  Future<Uint8List> generateSpeech(String text) async {
    if (!_isInitialized) {
      throw Exception('Model not initialized');
    }

    try {
      // Convert text to model input format
      final input = _preprocessText(text);
      
      // Prepare output buffer
      final outputBuffer = List.filled(OUTPUT_SIZE, 0).reshape([1, OUTPUT_SIZE]);
      
      // Run inference
      _interpreter.run(input, outputBuffer);
      
      // Convert output to audio data
      return outputBuffer.reshape([OUTPUT_SIZE]).buffer.asUint8List();
    } catch (e) {
      print('Error generating speech: $e');
      rethrow;
    }
  }

  Uint8List _preprocessText(String text) {
    // Implement text preprocessing
    // Convert text to model's expected input format
    return Uint8List(INPUT_SIZE);
  }

  void dispose() {
    _interpreter.close();
    _isInitialized = false;
  }

  static const int INPUT_SIZE = 1024;
  static const int OUTPUT_SIZE = 16000 * 2; // 16kHz, 16-bit audio
}
```

### 2. Audio Service

Create a service to handle audio playback:

```dart
import 'package:just_audio/just_audio.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

class AudioService {
  final AudioPlayer _player = AudioPlayer();

  Future<void> playAudio(Uint8List audioData) async {
    try {
      // Save audio data to temporary file
      final tempDir = await getTemporaryDirectory();
      final file = File('${tempDir.path}/temp_audio.wav');
      await file.writeAsBytes(audioData);

      // Play audio
      await _player.setFilePath(file.path);
      await _player.play();
    } catch (e) {
      print('Error playing audio: $e');
      rethrow;
    }
  }

  void stop() {
    _player.stop();
  }

  void dispose() {
    _player.dispose();
  }
}
```

### 3. Usage Example

Here's how to use the model in your Flutter widget:

```dart
import 'package:flutter/material.dart';

class ChatTTSWidget extends StatefulWidget {
  @override
  _ChatTTSWidgetState createState() => _ChatTTSWidgetState();
}

class _ChatTTSWidgetState extends State<ChatTTSWidget> {
  final ChatTTSService _chatTTSService = ChatTTSService();
  final AudioService _audioService = AudioService();
  final TextEditingController _textController = TextEditingController();
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  Future<void> _initializeServices() async {
    try {
      await _chatTTSService.initialize();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error initializing model: $e')),
      );
    }
  }

  Future<void> _generateSpeech() async {
    if (_textController.text.isEmpty) return;

    setState(() => _isLoading = true);

    try {
      final audioData = await _chatTTSService.generateSpeech(_textController.text);
      await _audioService.playAudio(audioData);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error generating speech: $e')),
      );
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        TextField(
          controller: _textController,
          decoration: InputDecoration(
            hintText: 'Enter text to convert to speech',
            border: OutlineInputBorder(),
          ),
        ),
        SizedBox(height: 16),
        ElevatedButton(
          onPressed: _isLoading ? null : _generateSpeech,
          child: _isLoading
              ? CircularProgressIndicator()
              : Text('Generate Speech'),
        ),
      ],
    );
  }

  @override
  void dispose() {
    _chatTTSService.dispose();
    _audioService.dispose();
    _textController.dispose();
    super.dispose();
  }
}
```

## Performance Optimization

1. **Model Loading**
   - Initialize model at app startup
   - Cache model instance
   - Use model quantization

2. **Memory Management**
   - Dispose resources properly
   - Use weak references
   - Implement cleanup

3. **Error Handling**
   - Implement proper error handling
   - Show user-friendly messages
   - Log errors for debugging

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check model file path
   - Verify model format
   - Ensure sufficient memory

2. **Audio Playback Issues**
   - Check audio format
   - Verify permissions
   - Test audio output

3. **Performance Issues**
   - Monitor memory usage
   - Profile model inference
   - Optimize text preprocessing

## Best Practices

1. **Resource Management**
   - Dispose resources in dispose method
   - Use lifecycle-aware components
   - Implement proper cleanup

2. **Error Handling**
   - Use try-catch blocks
   - Implement proper error recovery
   - Log errors for debugging

3. **User Experience**
   - Show loading indicators
   - Provide feedback for actions
   - Handle edge cases gracefully

## Constants

```dart
class ChatTTSConstants {
  static const int sampleRate = 16000;
  static const int maxAudioLength = 10; // seconds
  static const int audioFormat = 16; // bits
  static const int channelConfig = 1; // mono
}
``` 