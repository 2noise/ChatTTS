import argparse
import datetime
import os
import zipfile
from io import BytesIO

import requests

chattts_service_host = os.environ.get("CHATTTS_SERVICE_HOST", "127.0.0.1")
chattts_service_port = os.environ.get("CHATTTS_SERVICE_PORT", "9900")

CHATTTS_URL = f"http://{chattts_service_host}:{chattts_service_port}/generate_voice"


def parse_arguments():
    parser = argparse.ArgumentParser(description="HTTP client for ChatTTS service")
    parser.add_argument(
        "--text", type=str, nargs="+", required=True, help="Text to synthesize"
    )
    parser.add_argument(
        "--audio_seed", type=int, required=True, help="Audio generation seed"
    )
    parser.add_argument(
        "--text_seed", type=int, required=True, help="Text generation seed"
    )
    parser.add_argument(
        "--stream", type=bool, default=False, help="Enable/disable streaming"
    )
    parser.add_argument("--lang", type=str, default=None, help="Language code for text")
    parser.add_argument(
        "--skip_refine_text", type=bool, default=True, help="Skip text refinement"
    )
    parser.add_argument(
        "--refine_text_only", type=bool, default=False, help="Only refine text"
    )
    parser.add_argument(
        "--use_decoder", type=bool, default=True, help="Use decoder during inference"
    )
    parser.add_argument(
        "--do_text_normalization",
        type=bool,
        default=True,
        help="Enable text normalization",
    )
    parser.add_argument(
        "--do_homophone_replacement",
        type=bool,
        default=False,
        help="Enable homophone replacement",
    )
    parser.add_argument(
        "--tgt",
        type=str,
        default="./output",
        help="Target directory to save output files",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="test.mp3",
        help="Target directory to save output files",
    )

    # Refinement text parameters
    parser.add_argument(
        "--refine_prompt", type=str, default="", help="Prompt for text refinement"
    )
    parser.add_argument(
        "--refine_top_P",
        type=float,
        default=0.7,
        help="Top P value for text refinement",
    )
    parser.add_argument(
        "--refine_top_K", type=int, default=20, help="Top K value for text refinement"
    )
    parser.add_argument(
        "--refine_temperature",
        type=float,
        default=0.7,
        help="Temperature for text refinement",
    )
    parser.add_argument(
        "--refine_repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for text refinement",
    )
    parser.add_argument(
        "--refine_max_new_token",
        type=int,
        default=384,
        help="Max new tokens for text refinement",
    )
    parser.add_argument(
        "--refine_min_new_token",
        type=int,
        default=0,
        help="Min new tokens for text refinement",
    )
    parser.add_argument(
        "--refine_show_tqdm",
        type=bool,
        default=True,
        help="Show progress bar for text refinement",
    )
    parser.add_argument(
        "--refine_ensure_non_empty",
        type=bool,
        default=True,
        help="Ensure non-empty output",
    )
    parser.add_argument(
        "--refine_stream_batch",
        type=int,
        default=24,
        help="Stream batch size for refinement",
    )

    # Infer code parameters
    parser.add_argument(
        "--infer_prompt", type=str, default="[speed_5]", help="Prompt for inference"
    )
    parser.add_argument(
        "--infer_top_P", type=float, default=0.1, help="Top P value for inference"
    )
    parser.add_argument(
        "--infer_top_K", type=int, default=20, help="Top K value for inference"
    )
    parser.add_argument(
        "--infer_temperature", type=float, default=0.3, help="Temperature for inference"
    )
    parser.add_argument(
        "--infer_repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty for inference",
    )
    parser.add_argument(
        "--infer_max_new_token",
        type=int,
        default=2048,
        help="Max new tokens for inference",
    )
    parser.add_argument(
        "--infer_min_new_token",
        type=int,
        default=0,
        help="Min new tokens for inference",
    )
    parser.add_argument(
        "--infer_show_tqdm",
        type=bool,
        default=True,
        help="Show progress bar for inference",
    )
    parser.add_argument(
        "--infer_ensure_non_empty",
        type=bool,
        default=True,
        help="Ensure non-empty output",
    )
    parser.add_argument(
        "--infer_stream_batch",
        type=bool,
        default=True,
        help="Stream batch for inference",
    )
    parser.add_argument(
        "--infer_spk_emb",
        type=str,
        default=None,
        help="Speaker embedding for inference",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Main infer params
    body = {
        "text": args.text,
        "stream": args.stream,
        "lang": args.lang,
        "filename": args.filename,
        "skip_refine_text": args.skip_refine_text,
        "refine_text_only": args.refine_text_only,
        "use_decoder": args.use_decoder,
        "audio_seed": args.audio_seed,
        "text_seed": args.text_seed,
        "do_text_normalization": args.do_text_normalization,
        "do_homophone_replacement": args.do_homophone_replacement,
    }
    # Refinement text parameters
    params_refine_text = {
        "prompt": args.refine_prompt,
        "top_P": args.refine_top_P,
        "top_K": args.refine_top_K,
        "temperature": args.refine_temperature,
        "repetition_penalty": args.refine_repetition_penalty,
        "max_new_token": args.refine_max_new_token,
        "min_new_token": args.refine_min_new_token,
        "show_tqdm": args.refine_show_tqdm,
        "ensure_non_empty": args.refine_ensure_non_empty,
        "stream_batch": args.refine_stream_batch,
    }
    body["params_refine_text"] = params_refine_text

    # Infer code parameters
    params_infer_code = {
        "prompt": args.infer_prompt,
        "top_P": args.infer_top_P,
        "top_K": args.infer_top_K,
        "temperature": args.infer_temperature,
        "repetition_penalty": args.infer_repetition_penalty,
        "max_new_token": args.infer_max_new_token,
        "min_new_token": args.infer_min_new_token,
        "show_tqdm": args.infer_show_tqdm,
        "ensure_non_empty": args.infer_ensure_non_empty,
        "stream_batch": args.infer_stream_batch,
        "spk_emb": args.infer_spk_emb,
    }
    body["params_infer_code"] = params_infer_code

    try:
        response = requests.post(CHATTTS_URL, json=body)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content), "r") as zip_ref:
            tgt = args.tgt
            # filename=args.filename
            os.makedirs(tgt, exist_ok=True)
            zip_ref.extractall(tgt)
            print(f"Extracted files:{tgt}/{filename}")
            # print(tgt)
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")


if __name__ == "__main__":
    main()
