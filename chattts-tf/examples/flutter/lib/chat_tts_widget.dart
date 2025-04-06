import 'package:flutter/material.dart';
import 'services/chat_tts_service.dart';
import 'services/audio_service.dart';

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
    if (_textController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please enter some text')),
      );
      return;
    }

    setState(() => _isLoading = true);

    try {
      final audioData =
          await _chatTTSService.generateSpeech(_textController.text);
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
    return Scaffold(
      appBar: AppBar(
        title: Text('ChatTTS Demo'),
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'ChatTTS Demo',
              style: Theme.of(context).textTheme.headlineMedium,
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 24),
            TextField(
              controller: _textController,
              decoration: InputDecoration(
                hintText: 'Enter text to convert to speech',
                border: OutlineInputBorder(),
              ),
              maxLines: 5,
            ),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: _isLoading ? null : _generateSpeech,
              style: ElevatedButton.styleFrom(
                padding: EdgeInsets.symmetric(vertical: 16),
              ),
              child: _isLoading
                  ? CircularProgressIndicator(color: Colors.white)
                  : Text(
                      'Generate Speech',
                      style: TextStyle(fontSize: 16),
                    ),
            ),
          ],
        ),
      ),
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
