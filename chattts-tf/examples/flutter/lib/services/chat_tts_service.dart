import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';

class ChatTTSService {
  late Interpreter _interpreter;
  bool _isInitialized = false;

  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      _interpreter =
          await Interpreter.fromAsset('assets/models/chattts_model.tflite');
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
      final outputBuffer = Float32List(OUTPUT_SIZE);

      // Run inference
      _interpreter.run(input, outputBuffer);

      // Convert float output to 16-bit PCM audio data
      final audioData = Int16List(OUTPUT_SIZE);
      for (var i = 0; i < OUTPUT_SIZE; i++) {
        // Convert float to 16-bit integer
        audioData[i] = (outputBuffer[i] * 32767).clamp(-32768, 32767).toInt();
      }

      // Convert to Uint8List for audio playback
      return audioData.buffer.asUint8List();
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
