from openai import OpenAI

prompt_dict = {
    "kimi": [
        {
            "role": "system",
            "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。",
        },
        {
            "role": "user",
            "content": "你好，请注意你现在生成的文字要按照人日常生活的口吻，你的回复将会后续用TTS模型转为语音，并且请把回答控制在100字以内。并且标点符号仅包含逗号和句号，将数字等转为文字回答。",
        },
        {
            "role": "assistant",
            "content": "好的，我现在生成的文字将按照人日常生活的口吻， 并且我会把回答控制在一百字以内, 标点符号仅包含逗号和句号，将阿拉伯数字等转为中文文字回答。下面请开始对话。",
        },
    ],
    "deepseek": [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "user",
            "content": "你好，请注意你现在生成的文字要按照人日常生活的口吻，你的回复将会后续用TTS模型转为语音，并且请把回答控制在100字以内。并且标点符号仅包含逗号和句号，将数字等转为文字回答。",
        },
        {
            "role": "assistant",
            "content": "好的，我现在生成的文字将按照人日常生活的口吻， 并且我会把回答控制在一百字以内, 标点符号仅包含逗号和句号，将阿拉伯数字等转为中文文字回答。下面请开始对话。",
        },
    ],
    "deepseek_TN": [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "user",
            "content": "你好，现在我们在处理TTS的文本输入，下面将会给你输入一段文本，请你将其中的阿拉伯数字等等转为文字表达，并且输出的文本里仅包含逗号和句号这两个标点符号",
        },
        {
            "role": "assistant",
            "content": "好的，我现在对TTS的文本输入进行处理。这一般叫做text normalization。下面请输入",
        },
        {"role": "user", "content": "We paid $123 for this desk."},
        {
            "role": "assistant",
            "content": "We paid one hundred and twenty three dollars for this desk.",
        },
        {"role": "user", "content": "详询请拨打010-724654"},
        {"role": "assistant", "content": "详询请拨打零幺零，七二四六五四"},
        {"role": "user", "content": "罗森宣布将于7月24日退市，在华门店超6000家！"},
        {
            "role": "assistant",
            "content": "罗森宣布将于七月二十四日退市，在华门店超过六千家。",
        },
    ],
}


class ChatOpenAI:
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    def call(self, user_question, temperature=0.3, prompt_version="kimi", **kwargs):

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=prompt_dict[prompt_version]
            + [
                {"role": "user", "content": user_question},
            ],
            temperature=temperature,
            **kwargs
        )
        return completion.choices[0].message.content
