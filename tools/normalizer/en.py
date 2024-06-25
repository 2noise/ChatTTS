from typing import Callable
from functools import partial


def normalizer_en_nemo_text() -> Callable[[str], str]:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    return partial(
        Normalizer(input_case="cased", lang="en").normalize,
        verbose=False,
        punct_post_process=True,
    )
