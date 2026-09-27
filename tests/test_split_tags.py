import unittest

from ChatTTS.norm import _combine_tags, _split_tags


def reassemble(text):
    texts, tags = _split_tags(text)
    if len(tags) > 0:
        return _combine_tags(list(texts), list(tags))
    return texts[0]


class SplitTagsTests(unittest.TestCase):
    def test_loose_bracket_does_not_drop_the_next_tag(self):
        self.assertEqual(reassemble("see] [laugh]"), "see] [laugh]")
        self.assertEqual(
            reassemble("hello] [laugh] world"),
            "hello] [laugh] world",
        )

    def test_control_tags_stay_in_place(self):
        self.assertEqual(reassemble("hi [laugh] there"), "hi [laugh] there")
        self.assertEqual(reassemble("[laugh] hi"), "[laugh] hi")
        self.assertEqual(reassemble("hi [laugh]"), "hi [laugh]")
        self.assertEqual(
            reassemble("[oral_2][laugh_0]hello"),
            "[oral_2][laugh_0]hello",
        )
        self.assertEqual(
            reassemble("a [laugh] b [uv_break]"),
            "a [laugh] b [uv_break]",
        )

    def test_bracket_inside_a_word_stays(self):
        self.assertEqual(reassemble("array[0]"), "array[0]")
        self.assertEqual(reassemble("a]b"), "a]b")


if __name__ == "__main__":
    unittest.main()
