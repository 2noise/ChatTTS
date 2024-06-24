import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import argparse

import gradio as gr

from dotenv import load_dotenv
load_dotenv("sha256.env")

from examples.web.funcs import *

def main():

    with gr.Blocks() as demo:
        gr.Markdown("# ChatTTS WebUI")
        gr.Markdown("- **GitHub Repo**: https://github.com/2noise/ChatTTS")
        gr.Markdown("- **HuggingFace Repo**: https://huggingface.co/2Noise/ChatTTS")

        default_text = "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。"
        text_input = gr.Textbox(label="Input Text", lines=4, placeholder="Please Input Text...", value=default_text)

        with gr.Row():
            refine_text_checkbox = gr.Checkbox(label="Refine text", value=True)
            temperature_slider = gr.Slider(minimum=0.00001, maximum=1.0, step=0.00001, value=0.3, label="Audio temperature", interactive=True)
            top_p_slider = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.7, label="top_P", interactive=True)
            top_k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=20, label="top_K", interactive=True)

        with gr.Row():
            voice_selection = gr.Dropdown(label="Timbre", choices=voices.keys(), value='Default')
            audio_seed_input = gr.Number(value=2, label="Audio Seed")
            generate_audio_seed = gr.Button("\U0001F3B2")
            text_seed_input = gr.Number(value=42, label="Text Seed")
            generate_text_seed = gr.Button("\U0001F3B2")
        
        with gr.Row():
            dvae_coef_text = gr.Textbox(
                label="DVAE Coefficient", max_lines=3, show_copy_button=True, scale=4,
            )
            reload_chat_button = gr.Button("Reload", scale=1)

        with gr.Row():
            auto_play_checkbox = gr.Checkbox(label="Auto Play", value=False, scale=1)
            stream_mode_checkbox = gr.Checkbox(label="Stream Mode", value=False, scale=1)
            generate_button = gr.Button("Generate", scale=2, variant="primary")
            interrupt_button = gr.Button("Interrupt", scale=2, variant="stop", visible=False, interactive=False)

        text_output = gr.Textbox(label="Output Text", interactive=False)

        # 使用Gradio的回调功能来更新数值输入框
        voice_selection.change(fn=on_voice_change, inputs=voice_selection, outputs=audio_seed_input)

        generate_audio_seed.click(generate_seed,
                                  inputs=[],
                                  outputs=audio_seed_input)

        generate_text_seed.click(generate_seed,
                                 inputs=[],
                                 outputs=text_seed_input)
        
        reload_chat_button.click(reload_chat, inputs=dvae_coef_text, outputs=dvae_coef_text)
        
        generate_button.click(fn=lambda: "𝕃𝕠𝕒𝕕𝕚𝕟𝕘...", outputs=text_output)
        generate_button.click(refine_text,
                              inputs=[text_input, text_seed_input, refine_text_checkbox, generate_button, interrupt_button],
                              outputs=[text_output, generate_button, interrupt_button])
        
        interrupt_button.click(interrupt_generate)

        @gr.render(inputs=[auto_play_checkbox, stream_mode_checkbox])
        def make_audio(autoplay, stream):
            audio_output = gr.Audio(
                label="Output Audio",
                value=None,
                autoplay=autoplay,
                streaming=stream,
                interactive=False,
                show_label=True,
            )
            text_output.change(text_output_listener, inputs=[generate_button, interrupt_button], outputs=[generate_button, interrupt_button])
            text_output.change(generate_audio,
                                inputs=[text_output, temperature_slider, top_p_slider, top_k_slider, audio_seed_input, stream_mode_checkbox],
                                outputs=audio_output).then(fn=set_buttons_after_generate, inputs=[generate_button, interrupt_button, audio_output], outputs=[generate_button, interrupt_button])

        gr.Examples(
            examples=[
                ["四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。", 0.3, 0.7, 20, 2, 42, True],
                ["What is [uv_break]your favorite english food?[laugh][lbreak]", 0.5, 0.5, 10, 245, 531, False],
                ["chat T T S is a text to speech model designed for dialogue applications. [uv_break]it supports mixed language input [uv_break]and offers multi speaker capabilities with precise control over prosodic elements [laugh]like like [uv_break]laughter[laugh], [uv_break]pauses, [uv_break]and intonation. [uv_break]it delivers natural and expressive speech,[uv_break]so please[uv_break] use the project responsibly at your own risk.[uv_break]", 0.2, 0.6, 15, 67, 165, False],
            ],
            inputs=[text_input, temperature_slider, top_p_slider, top_k_slider, audio_seed_input, text_seed_input, refine_text_checkbox],
        )
    
    parser = argparse.ArgumentParser(description='ChatTTS demo Launch')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='server name')
    parser.add_argument('--server_port', type=int, default=8080, help='server port')
    parser.add_argument('--root_path', type=str, default=None, help='root path')
    parser.add_argument('--custom_path', type=str, default=None, help='custom model path')
    parser.add_argument('--coef', type=str, default=None, help='custom dvae coefficient')
    args = parser.parse_args()

    logger.info("loading ChatTTS model...")

    if load_chat(args.custom_path, args.coef):
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)

    dvae_coef_text.value = chat.coef

    demo.launch(server_name=args.server_name, server_port=args.server_port, root_path=args.root_path, inbrowser=True)


if __name__ == '__main__':
    main()
