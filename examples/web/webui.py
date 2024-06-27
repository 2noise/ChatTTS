import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import argparse

import gradio as gr

from examples.web.funcs import *
from examples.web.ex import ex


def main():

    with gr.Blocks() as demo:
        gr.Markdown("# ChatTTS WebUI")
        gr.Markdown("- **GitHub Repo**: https://github.com/2noise/ChatTTS")
        gr.Markdown("- **HuggingFace Repo**: https://huggingface.co/2Noise/ChatTTS")

        default_text = "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。"
        text_input = gr.Textbox(
            label="Input Text",
            lines=4,
            placeholder="Please Input Text...",
            value=default_text,
            interactive=True,
        )

        with gr.Row():
            refine_text_checkbox = gr.Checkbox(label="Refine text", value=True)
            temperature_slider = gr.Slider(
                minimum=0.00001,
                maximum=1.0,
                step=0.00001,
                value=0.3,
                label="Audio Temperature",
                interactive=True,
            )
            top_p_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                step=0.05,
                value=0.7,
                label="top_P",
                interactive=True,
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=20, step=1, value=20, label="top_K", interactive=True
            )

        with gr.Row():
            voice_selection = gr.Dropdown(
                label="Timbre", choices=voices.keys(), value="Default"
            )
            audio_seed_input = gr.Number(
                value=2,
                label="Audio Seed",
                interactive=True,
                minimum=seed_min,
                maximum=seed_max,
            )
            generate_audio_seed = gr.Button("\U0001F3B2")
            text_seed_input = gr.Number(
                value=42,
                label="Text Seed",
                interactive=True,
                minimum=seed_min,
                maximum=seed_max,
            )
            generate_text_seed = gr.Button("\U0001F3B2")

        with gr.Row():
            spk_emb_text = gr.Textbox(
                label="Speaker Embedding",
                max_lines=3,
                show_copy_button=True,
                interactive=True,
                scale=2,
            )
            dvae_coef_text = gr.Textbox(
                label="DVAE Coefficient",
                max_lines=3,
                show_copy_button=True,
                interactive=True,
                scale=2,
            )
            reload_chat_button = gr.Button("Reload", scale=1)

        with gr.Row():
            auto_play_checkbox = gr.Checkbox(label="Auto Play", value=False, scale=1)
            stream_mode_checkbox = gr.Checkbox(
                label="Stream Mode", value=False, scale=1
            )
            generate_button = gr.Button("Generate", scale=2, variant="primary")
            interrupt_button = gr.Button(
                "Interrupt", scale=2, variant="stop", visible=False, interactive=False
            )

        text_output = gr.Textbox(
            label="Output Text", interactive=False, show_copy_button=True
        )

        # 使用Gradio的回调功能来更新数值输入框
        voice_selection.change(
            fn=on_voice_change, inputs=voice_selection, outputs=audio_seed_input
        )

        generate_audio_seed.click(generate_seed, outputs=audio_seed_input)

        generate_text_seed.click(generate_seed, outputs=text_seed_input)

        audio_seed_input.change(
            on_audio_seed_change, inputs=audio_seed_input, outputs=spk_emb_text
        )

        reload_chat_button.click(
            reload_chat, inputs=dvae_coef_text, outputs=dvae_coef_text
        )

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
            generate_button.click(
                fn=set_buttons_before_generate,
                inputs=[generate_button, interrupt_button],
                outputs=[generate_button, interrupt_button],
            ).then(
                refine_text,
                inputs=[
                    text_input,
                    text_seed_input,
                    refine_text_checkbox,
                ],
                outputs=text_output,
            ).then(
                generate_audio,
                inputs=[
                    text_output,
                    temperature_slider,
                    top_p_slider,
                    top_k_slider,
                    spk_emb_text,
                    stream_mode_checkbox,
                ],
                outputs=audio_output,
            ).then(
                fn=set_buttons_after_generate,
                inputs=[generate_button, interrupt_button, audio_output],
                outputs=[generate_button, interrupt_button],
            )

        gr.Examples(
            examples=ex,
            inputs=[
                text_input,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                audio_seed_input,
                text_seed_input,
                refine_text_checkbox,
            ],
        )

    parser = argparse.ArgumentParser(description="ChatTTS demo Launch")
    parser.add_argument(
        "--server_name", type=str, default="0.0.0.0", help="server name"
    )
    parser.add_argument("--server_port", type=int, default=8080, help="server port")
    parser.add_argument("--root_path", type=str, default=None, help="root path")
    parser.add_argument(
        "--custom_path", type=str, default=None, help="custom model path"
    )
    parser.add_argument(
        "--coef", type=str, default=None, help="custom dvae coefficient"
    )
    args = parser.parse_args()

    logger.info("loading ChatTTS model...")

    if load_chat(args.custom_path, args.coef):
        logger.info("Models loaded successfully.")
    else:
        logger.error("Models load failed.")
        sys.exit(1)

    spk_emb_text.value = on_audio_seed_change(audio_seed_input.value)
    dvae_coef_text.value = chat.coef

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        root_path=args.root_path,
        inbrowser=True,
        show_api=False,
    )


if __name__ == "__main__":
    main()
