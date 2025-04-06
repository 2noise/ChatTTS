package com.example.chattts

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

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
                    if (text.isNotEmpty()) {
                        val audioData = chatTTSModel.generateSpeech(text)
                        audioService.playAudio(audioData)
                    } else {
                        Toast.makeText(this@MainActivity, 
                            "Please enter some text", 
                            Toast.LENGTH_SHORT).show()
                    }
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