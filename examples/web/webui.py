import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

import argparse

import gradio as gr

from funcs import *
from ex import ex


def main():

    with gr.Blocks() as demo:
        gr.Markdown("# ChatTTS WebUI")
        gr.Markdown("- **GitHub Repo**: https://github.com/2noise/ChatTTS")
        gr.Markdown("- **HuggingFace Repo**: https://huggingface.co/2Noise/ChatTTS")

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Input Text",
                    lines=4,
                    max_lines=4,
                    placeholder="Please Input Text...",
                    value=ex[0][0],
                    interactive=True,
                )
                sample_text_input = gr.Textbox(
                    label="Sample Text",
                    lines=4,
                    max_lines=4,
                    placeholder="If Sample Audio and Sample Text are available, the Speaker Embedding will be disabled.",
                    interactive=True,
                )
            with gr.Column():
                with gr.Tab(label="Sample Audio"):
                    sample_audio_input = gr.Audio(
                        value=None,
                        type="filepath",
                        interactive=True,
                        show_label=False,
                        waveform_options=gr.WaveformOptions(
                            sample_rate=24000,
                        ),
                        scale=1,
                    )
                with gr.Tab(label="Sample Audio Code"):
                    sample_audio_code_input = gr.Textbox(
                        lines=12,
                        max_lines=12,
                        show_label=False,
                        placeholder="Paste the Code copied before after uploading Sample Audio.",
                        interactive=True,
                    )

        with gr.Row():
            refine_text_checkbox = gr.Checkbox(
                label="Refine text", value=ex[0][6], interactive=True
            )
            temperature_slider = gr.Slider(
                minimum=0.00001,
                maximum=1.0,
                step=0.00001,
                value=ex[0][1],
                label="Audio Temperature",
                interactive=True,
            )
            top_p_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                step=0.05,
                value=ex[0][2],
                label="top_P",
                interactive=True,
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=20,
                step=1,
                value=ex[0][3],
                label="top_K",
                interactive=True,
            )

        with gr.Row():
            voice_selection = gr.Dropdown(
                label="Timbre",
                choices=voices.keys(),
                value="Default",
                interactive=True,
            )
            audio_seed_input = gr.Number(
                value=ex[0][4],
                label="Audio Seed",
                interactive=True,
                minimum=seed_min,
                maximum=seed_max,
            )
            generate_audio_seed = gr.Button("\U0001F3B2", interactive=True)
            text_seed_input = gr.Number(
                value=ex[0][5],
                label="Text Seed",
                interactive=True,
                minimum=seed_min,
                maximum=seed_max,
            )
            generate_text_seed = gr.Button("\U0001F3B2", interactive=True)

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
            reload_chat_button = gr.Button("Reload", scale=1, interactive=True)

        with gr.Row():
            auto_play_checkbox = gr.Checkbox(
                label="Auto Play", value=False, scale=1, interactive=True
            )
            stream_mode_checkbox = gr.Checkbox(
                label="Stream Mode",
                value=False,
                scale=1,
                interactive=True,
            )
            split_batch_slider = gr.Slider(
                minimum=0,
                maximum=100,
                step=1,
                value=4,
                label="Split Batch",
                interactive=True,
            )
            generate_button = gr.Button(
                "Generate", scale=2, variant="primary", interactive=True
            )
            interrupt_button = gr.Button(
                "Interrupt",
                scale=2,
                variant="stop",
                visible=False,
                interactive=False,
            )

        text_output = gr.Textbox(
            label="Output Text",
            interactive=False,
            show_copy_button=True,
        )

        sample_audio_input.change(
            fn=on_upload_sample_audio,
            inputs=sample_audio_input,
            outputs=sample_audio_code_input,
        ).then(fn=lambda: gr.Info("Sampled Audio Code generated at another Tab."))

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
                format="mp3" if use_mp3 and not stream else "wav",
                autoplay=autoplay,
                streaming=stream,
                interactive=False,
                show_label=True,
                waveform_options=gr.WaveformOptions(
                    sample_rate=24000,
                ),
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
                    temperature_slider,
                    top_p_slider,
                    top_k_slider,
                    split_batch_slider,
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
                    audio_seed_input,
                    sample_text_input,
                    sample_audio_code_input,
                    split_batch_slider,
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
    parser.add_argument("--root_path", type=str, help="root path")
    parser.add_argument("--custom_path", type=str, help="custom model path")
    parser.add_argument("--coef", type=str, help="custom dvae coefficient")
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
