import re

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

text_split_registry = {}


def register(key):
    """注册装饰器"""

    def decorator(cls):
        text_split_registry[key] = cls
        return cls

    return decorator


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if len(todo_text): return []
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


@register("cut2")
def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return [inp]
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return opts


@register("cut5")
def cut5(inp):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    # punds = r'[,.;?!、，。？！;：…]'
    punds = r'[,.;?!、，。？！；：:…]'
    items = re.split(f'({punds})', inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    # 在句子不存在符号或句尾无符号的时候保证文本完整
    if len(items) % 2 == 1:
        mergeitems.append(items[-1])
    return mergeitems
