# Android Integration Guide

This guide explains how to integrate the ChatTTS TensorFlow model into your Android application.

## Setup

### 1. Add Dependencies

Add the following to your app's `build.gradle`:

```gradle
dependencies {
    // TensorFlow dependencies
    implementation 'org.tensorflow:tensorflow-lite:2.9.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.2'
    
    // Audio playback
    implementation 'com.google.android.exoplayer:exoplayer:2.19.1'
    
    // Coroutines for async operations
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.6.4'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.4'
}
```

### 2. Add Permissions

Add these permissions to your `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

## Implementation

### 1. Model Interface

Create a class to handle the TensorFlow model:

```kotlin
class ChatTTSModel(private val context: Context) {
    private var interpreter: Interpreter? = null
    private val modelPath = "chattts_model.tflite"
    
    init {
        loadModel()
    }
    
    private fun loadModel() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()
        interpreter = Interpreter(model, options)
    }
    
    suspend fun generateSpeech(text: String): ByteArray {
        return withContext(Dispatchers.Default) {
            // Convert text to model input format
            val input = preprocessText(text)
            
            // Run inference
            val outputBuffer = ByteBuffer.allocateDirect(OUTPUT_SIZE)
            interpreter?.run(input, outputBuffer)
            
            // Convert output to audio data
            outputBuffer.array()
        }
    }
    
    private fun preprocessText(text: String): ByteBuffer {
        // Implement text preprocessing
        // Convert text to model's expected input format
        return ByteBuffer.allocateDirect(INPUT_SIZE)
    }
    
    companion object {
        private const val INPUT_SIZE = 1024
        private const val OUTPUT_SIZE = 16000 * 2 // 16kHz, 16-bit audio
    }
}
```

### 2. Audio Service

Create a service to handle audio playback:

```kotlin
class AudioService(private val context: Context) {
    private var player: ExoPlayer? = null
    
    fun playAudio(audioData: ByteArray) {
        val audioFile = saveAudioToFile(audioData)
        val mediaSource = ProgressiveMediaSource.Factory(DefaultDataSource.Factory(context))
            .createMediaSource(MediaItem.fromUri(Uri.fromFile(audioFile)))
            
        player = ExoPlayer.Builder(context).build().apply {
            setMediaSource(mediaSource)
            prepare()
            play()
        }
    }
    
    private fun saveAudioToFile(audioData: ByteArray): File {
        val file = File(context.cacheDir, "temp_audio.wav")
        file.writeBytes(audioData)
        return file
    }
    
    fun stop() {
        player?.stop()
        player?.release()
        player = null
    }
}
```

### 3. Usage Example

Here's how to use the model in your activity:

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var chatTTSModel: ChatTTSModel
    private lateinit var audioService: AudioService
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        chatTTSModel = ChatTTSModel(this)
        audioService = AudioService(this)
        
        findViewById<Button>(R.id.speakButton).setOnClickListener {
            lifecycleScope.launch {
                try {
                    val text = findViewById<EditText>(R.id.inputText).text.toString()
                    val audioData = chatTTSModel.generateSpeech(text)
                    audioService.playAudio(audioData)
                } catch (e: Exception) {
                    Toast.makeText(this@MainActivity, 
                        "Error generating speech: ${e.message}", 
                        Toast.LENGTH_SHORT).show()
                }
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        audioService.stop()
    }
}
```

## Performance Optimization

1. **Model Loading**
   - Load model in background thread
   - Cache model instance
   - Use model quantization for smaller size

2. **Memory Management**
   - Release resources properly
   - Use weak references where appropriate
   - Implement proper cleanup

3. **Error Handling**
   - Implement proper error handling
   - Show user-friendly error messages
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
   - Test audio output device

3. **Performance Issues**
   - Monitor memory usage
   - Profile model inference
   - Optimize text preprocessing

## Best Practices

1. **Resource Management**
   - Release resources in onDestroy
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

```kotlin
object ChatTTSConstants {
    const val SAMPLE_RATE = 16000
    const val MAX_AUDIO_LENGTH = 10 // seconds
    const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    const val CHANNEL_CONFIG = AudioFormat.CHANNEL_OUT_MONO
}
``` 