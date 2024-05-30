import requests


def tts(
    server_url: str,
    text,
    seed,
    output_file,
    timeout: int = 20,
):
    data = {
        "text": text,
        "seed": seed,
        "temperature": 0.3,
        "top_P": 0.7,
        "top_K": 20,
        "skip_refine_text": False,
    }
    try:
        response = requests.post(
            server_url + "/tts",
            stream=True,
            json=data,
            timeout=timeout,
        )
        response.raise_for_status()
        with open(output_file, "wb") as f:
            f.write(response.content)

    except requests.exceptions.RequestException as e:
        raise ValueError(response.content) from e


if __name__ == "__main__":
    tts(
        "http://127.0.0.1:8080",
        "你好世界",
        1111,
        "output.mp3",
    )
    print("See output.mp3")
