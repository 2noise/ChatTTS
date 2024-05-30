import requests
import streamlit as st

# 音色选项
voices = {
    "音色1": {"seed": 1111},
    "音色2": {"seed": 2222},
    "音色3": {"seed": 3333},
    "音色4": {"seed": 4444},
    "音色5": {"seed": 5555},
    "音色6": {"seed": 6666},
    "音色7": {"seed": 7777},
    "音色8": {"seed": 8888},
    "音色9": {"seed": 9999},
}


def tts(
    server_url: str,
    data,
    output_file,
):
    try:
        response = requests.post(
            server_url + "/tts",
            stream=True,
            json=data,
            timeout=20,
        )
        response.raise_for_status()
        with open(output_file, "wb") as f:
            f.write(response.content)

    except requests.exceptions.RequestException as e:
        raise ValueError(response.content) from e


# Streamlit界面设置
st.title("ChatTTS 音频生成器")

st.header("选择音色")
st.markdown("通过随机种子生成,不保证每次运行结果一致")

voice_options = list(voices.keys())
selected_voice = st.selectbox("选择音色", voice_options)
seed = voices[selected_voice]["seed"]

st.header("输入文本")
text = st.text_area("请输入要合成的文本")

st.header("参数设置")
temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
top_P = st.slider("Top P", 0.0, 1.0, 0.7)
top_K = st.slider("Top K", 0, 50, 20)
skip_refine_text = st.checkbox("跳过文本调整")


# 音频生成
if st.button("生成音频"):
    # 调用ChatTTS进行推断
    with st.spinner("正在生成音频..."):
        tmp_file = "temp.mp3"
        data = {
            "text": text,
            "seed": seed,
            "temperature": temperature,
            "top_P": top_P,
            "top_K": top_K,
            "skip_refine_text": skip_refine_text,
        }
        tts("http://127.0.0.1:8080", data, tmp_file)

        st.audio(tmp_file)
