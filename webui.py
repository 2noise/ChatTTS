import os
import gradio as gr
import torch
import numpy as np
import ChatTTS
import random
from IPython.display import Audio


print("loading ChatTTS model...")
chat = ChatTTS.Chat()
chat.load_models()


def generate_seed():
    new_seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": new_seed
        }



def generate_audio(text, audio_temperature, audio_seed_input, text_seed_input, refine_text_flag):

    torch.manual_seed(audio_seed_input)
    rand_spk = torch.randn(768)
    params_infer_code = {'spk_emb': rand_spk, 'temperature': audio_temperature}
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
    
    torch.manual_seed(text_seed_input)

    skip_refine_text = not refine_text_flag

    wav, text = chat.infer(text, skip_refine_text=skip_refine_text, params_refine_text=params_refine_text, params_infer_code=params_infer_code, return_text=True)
    
    audio_data = np.array(wav[0]).flatten()
    # audio_data = (np.array(wav[0], dtype=np.float16) * 32767).astype(np.int16).flatten()
    sample_rate = 24000
    text_data = text[0] if isinstance(text, list) else text

    return [(sample_rate, audio_data), text_data]


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# ChatTTS Generator Webui")
        gr.Markdown("ChatTTS Model: [2noise/ChatTTS](https://github.com/2noise/ChatTTS)")

        default_text = "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。"
        
        text_input = gr.Textbox(label="Input Text", lines=4, placeholder="Please Input Text...", value=default_text)

        with gr.Row():
            refine_text_checkbox = gr.Checkbox(label="Refine text", value=True)
            audio_temperature_slider = gr.Slider(minimum=0.00001, maximum=1.0, step=0.00001, value=0.3, label="Audio temperature")

        with gr.Row():
            audio_seed_input = gr.Number(value=42, label="Audio Seed")
            generate_audio_seed = gr.Button("\U0001F3B2")
            text_seed_input = gr.Number(value=42, label="Text Seed")
            generate_text_seed = gr.Button("\U0001F3B2")

        generate_button = gr.Button("Generate")
        
        text_output = gr.Textbox(label="Output Text", interactive=False)
        audio_output = gr.Audio(label="Output Audio")

        generate_audio_seed.click(generate_seed, inputs=[], outputs=audio_seed_input)
        generate_text_seed.click(generate_seed, inputs=[], outputs=text_seed_input)
        generate_button.click(generate_audio, inputs=[text_input, audio_temperature_slider, audio_seed_input, text_seed_input, refine_text_checkbox], outputs=[audio_output, text_output])

    # 启动 Gradio
    demo.launch(server_name="0.0.0.0", server_port=8080, inbrowser = True)


if __name__ == '__main__':
    main()