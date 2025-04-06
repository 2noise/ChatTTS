import 'package:just_audio/just_audio.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'dart:typed_data';

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
