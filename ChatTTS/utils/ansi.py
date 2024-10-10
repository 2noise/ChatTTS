#!/usr/bin/env python3

import re
import sys
from contextlib import contextmanager


class ANSI:
    ansi_color = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "purple": "\033[35m",
        "blue_light": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m",
        "upline": "\033[1A",
        "clear_line": "\033[2K",
        "clear": "\033[2J",
    }
    ansi_nocolor = {
        "black": "",
        "red": "",
        "green": "",
        "yellow": "",
        "blue": "",
        "purple": "",
        "blue_light": "",
        "white": "",
        "reset": "",
        "upline": "\033[1A\033[",
        "clear_line": "\033[K",
        "clear": "\033[2J",
    }

    def __init__(self):
        self._dict = ANSI.ansi_color if ("--color" in sys.argv) else ANSI.ansi_nocolor

    def switch(self, color: bool):
        self._dict = ANSI.ansi_color if color else ANSI.ansi_nocolor

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def __getitem__(self, key):
        return self._dict[key]

    def __str__(self):
        return str(self._dict)

    def __repr__(self):
        return repr(self._dict)


ansi = ANSI()


def remove_ansi(s: str) -> str:
    ansi_escape = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")
    return ansi_escape.sub("", s)


def get_ansi_len(s: str) -> int:
    return len(s) - len(remove_ansi(s))


def prints(*args: str, indent: int = 0, prefix: str = "", **kwargs):
    assert indent >= 0
    new_args = []
    for arg in args:
        new_args.append(indent_str(str(arg), indent=indent))
    if len(new_args):
        new_args[0] = prefix + str(new_args[0])
    print(*new_args, **kwargs)


def output_iter(_iter: int, iteration: int = None, iter_len: int = 4) -> str:
    if iteration is None:
        pattern = "{blue_light}[ {red}{0}{blue_light} ]{reset}"
        return pattern.format(str(_iter).rjust(iter_len), **ansi)
    else:
        iter_str = str(iteration)
        length = len(iter_str)
        pattern = (
            "{blue_light}[ {red}{0}{blue_light} " "/ {red}{1}{blue_light} ]{reset}"
        )
        return pattern.format(str(_iter).rjust(length), iter_str, **ansi)


def indent_str(s_: str, indent: int = 0) -> str:
    # modified from torch.nn.modules._addindent
    if indent > 0 and s_:
        s_ = indent * " " + str(s_[:-1]).replace("\n", "\n" + indent * " ") + s_[-1]
    return s_


class IndentRedirect:  # TODO: inherit TextIOWrapper?
    def __init__(self, buffer: bool = True, indent: int = 0):
        self.__console__ = sys.stdout
        self.indent = indent
        self.__buffer: str = None
        if buffer:
            self.__buffer = ""

    def write(self, text: str, indent: int = None):
        indent = indent if indent is not None else self.indent
        text = indent_str(text, indent=indent)
        if self.__buffer is None:
            self.__console__.write(text)
        else:
            self.__buffer += text

    def flush(self):
        if self.__buffer is not None:
            self.__console__.write(self.__buffer)
            self.__buffer = ""
        self.__console__.flush()

    @contextmanager
    def __call__(self):
        try:
            sys.stdout = self
            yield
        finally:
            sys.stdout = self.__console__
            self.__buffer = ""

    def enable(self):
        sys.stdout = self

    def disable(self):
        if self.__buffer is not None:
            self.__buffer = ""
        sys.stdout = self.__console__

    @property
    def buffer(self) -> str:
        return self.__buffer


redirect = IndentRedirect()
