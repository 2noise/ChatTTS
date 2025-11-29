from typing import Callable


def normalizer_zh_tn() -> Callable[[str], str]:
    from tn.chinese.normalizer import Normalizer

    return Normalizer(remove_interjections=False).normalize
