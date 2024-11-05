#!/usr/bin/env python3

import logging
from pathlib import Path

import statistics
import time
from collections import defaultdict, deque
from tqdm import tqdm as tqdm_class

from typing import Generator, Iterable, TypeVar

import torch
import torch.distributed as dist

from .ansi import ansi, prints, get_ansi_len

__all__ = ["SmoothedValue", "MetricLogger"]

MB = 1 << 20
T = TypeVar("T")


class SmoothedValue:
    r"""Track a series of values and provide access to smoothed values over a
    window or the global series average.

    See Also:
        https://github.com/pytorch/vision/blob/main/references/classification/utils.py

    Args:
        name (str): Name string.
        window_size (int): The :attr:`maxlen` of :class:`~collections.deque`.
        fmt (str): The format pattern of ``str(self)``.

    Attributes:
        name (str): Name string.
        fmt (str): The string pattern.
        deque (~collections.deque): The unique data series.
        count (int): The amount of data.
        total (float): The sum of all data.

        median (float): The median of :attr:`deque`.
        avg (float): The avg of :attr:`deque`.
        global_avg (float): :math:`\frac{\text{total}}{\text{count}}`
        max (float): The max of :attr:`deque`.
        min (float): The min of :attr:`deque`.
        last_value (float): The last value of :attr:`deque`.
    """

    def __init__(
        self, name: str = "", window_size: int = None, fmt: str = "{global_avg:.3f}"
    ):
        self.name = name
        self.deque: deque[float] = deque(maxlen=window_size)
        self.count: int = 0
        self.total: float = 0.0
        self.fmt = fmt

    def update(self, value: float, n: int = 1) -> 'SmoothedValue':
        r"""Update :attr:`n` pieces of data with same :attr:`value`.

        .. code-block:: python

            self.deque.append(value)
            self.total += value * n
            self.count += n

        Args:
            value (float): the value to update.
            n (int): the number of data with same :attr:`value`.

        Returns:
            SmoothedValue: return ``self`` for stream usage.
        """
        self.deque.append(value)
        self.total += value * n
        self.count += n
        return self

    def update_list(self, value_list: list[float]) -> 'SmoothedValue':
        r"""Update :attr:`value_list`.

        .. code-block:: python

            for value in value_list:
                self.deque.append(value)
                self.total += value
            self.count += len(value_list)

        Args:
            value_list (list[float]): the value list to update.

        Returns:
            SmoothedValue: return ``self`` for stream usage.
        """
        for value in value_list:
            self.deque.append(value)
            self.total += value
        self.count += len(value_list)
        return self

    def reset(self) -> 'SmoothedValue':
        r"""Reset ``deque``, ``count`` and ``total`` to be empty.

        Returns:
            SmoothedValue: return ``self`` for stream usage.
        """
        self.deque = deque(maxlen=self.deque.maxlen)
        self.count = 0
        self.total = 0.0
        return self

    def synchronize_between_processes(self):
        r"""
        Warning:
            Does NOT synchronize the deque!
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = float(t[1])

    @property
    def median(self) -> float:
        try:
            return statistics.median(self.deque)
        except Exception:
            return 0.0

    @property
    def avg(self) -> float:
        try:
            return statistics.mean(self.deque)
        except Exception:
            return 0.0

    @property
    def global_avg(self) -> float:
        try:
            return self.total / self.count
        except Exception:
            return 0.0

    @property
    def max(self) -> float:
        try:
            return max(self.deque)
        except Exception:
            return 0.0

    @property
    def min(self) -> float:
        try:
            return min(self.deque)
        except Exception:
            return 0.0

    @property
    def last_value(self) -> float:
        try:
            return self.deque[-1]
        except Exception:
            return 0.0

    def __str__(self):
        return self.fmt.format(
            name=self.name,
            count=self.count,
            total=self.total,
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            min=self.min,
            max=self.max,
            last_value=self.last_value,
        )

    def __format__(self, format_spec: str) -> str:
        return self.__str__()


class MetricLogger:
    r"""
    See Also:
        https://github.com/pytorch/vision/blob/main/references/classification/utils.py

    Args:
        delimiter (str): The delimiter to join different meter strings.
            Defaults to ``''``.
        meter_length (int): The minimum length for each meter.
            Defaults to ``20``.
        tqdm (bool): Whether to use tqdm to show iteration information.
            Defaults to ``env['tqdm']``.
        indent (int): The space indent for the entire string.
            Defaults to ``0``.

    Attributes:
        meters (dict[str, SmoothedValue]): The meter dict.
        iter_time (SmoothedValue): Iteration time meter.
        data_time (SmoothedValue): Data loading time meter.
        memory (SmoothedValue): Memory usage meter.
    """

    def __init__(
        self,
        delimiter: str = "",
        meter_length: int = 20,
        tqdm: bool = True,
        indent: int = 0,
        **kwargs,
    ):
        self.meters: defaultdict[str,
                                 SmoothedValue] = defaultdict(SmoothedValue)
        self.create_meters(**kwargs)
        self.delimiter = delimiter
        self.meter_length = meter_length
        self.tqdm = tqdm
        self.indent = indent

        self.iter_time = SmoothedValue()
        self.data_time = SmoothedValue()
        self.memory = SmoothedValue(fmt="{max:.0f}")

    def create_meters(self, **kwargs: str) -> 'SmoothedValue':
        r"""Create meters with specific ``fmt`` in :attr:`self.meters`.

        ``self.meters[meter_name] = SmoothedValue(fmt=fmt)``

        Args:
            **kwargs: ``(meter_name: fmt)``

        Returns:
            MetricLogger: return ``self`` for stream usage.
        """
        for k, v in kwargs.items():
            self.meters[k] = SmoothedValue(
                fmt="{global_avg:.3f}" if v is None else v)
        return self

    def update(self, n: int = 1, **kwargs: float) -> 'SmoothedValue':
        r"""Update values to :attr:`self.meters` by calling :meth:`SmoothedValue.update()`.

        ``self.meters[meter_name].update(float(value), n=n)``

        Args:
            n (int): the number of data with same value.
            **kwargs: ``{meter_name: value}``.

        Returns:
            MetricLogger: return ``self`` for stream usage.
        """
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(float(v), n=n)
        return self

    def update_list(self, **kwargs: list) -> 'SmoothedValue':
        r"""Update values to :attr:`self.meters` by calling :meth:`SmoothedValue.update_list()`.

        ``self.meters[meter_name].update_list(value_list)``

        Args:
            **kwargs: ``{meter_name: value_list}``.

        Returns:
            MetricLogger: return ``self`` for stream usage.
        """
        for k, v in kwargs.items():
            self.meters[k].update_list(v)
        return self

    def reset(self) -> 'SmoothedValue':
        r"""Reset meter in :attr:`self.meters` by calling :meth:`SmoothedValue.reset()`.

        Returns:
            MetricLogger: return ``self`` for stream usage.
        """
        for meter in self.meters.values():
            meter.reset()
        return self

    def get_str(self, cut_too_long: bool = True, strip: bool = True, **kwargs) -> str:
        r"""Generate formatted string based on keyword arguments.

        ``key: value`` with max length to be :attr:`self.meter_length`.

        Args:
            cut_too_long (bool): Whether to cut too long values to first 5 characters.
                Defaults to ``True``.
            strip (bool): Whether to strip trailing whitespaces.
                Defaults to ``True``.
            **kwargs: Keyword arguments to generate string.
        """
        str_list: list[str] = []
        for k, v in kwargs.items():
            v_str = str(v)
            _str: str = "{green}{k}{reset}: {v}".format(k=k, v=v_str, **ansi)
            max_length = self.meter_length + get_ansi_len(_str)
            if cut_too_long:
                _str = _str[:max_length]
            str_list.append(_str.ljust(max_length))
        _str = self.delimiter.join(str_list)
        if strip:
            _str = _str.rstrip()
        return _str

    def __getattr__(self, attr: str) -> float:
        if attr in self.meters:
            return self.meters[attr]
        if attr in vars(self):  # TODO: use hasattr
            return vars(self)[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, attr)
        )

    def __str__(self) -> str:
        return self.get_str(**self.meters)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def log_every(
        self,
        iterable: Iterable[T],
        header: str = "",
        tqdm: bool = None,
        tqdm_header: str = "Iter",
        indent: int = None,
        verbose: int = 1,
    ) -> Generator[T, None, None]:
        r"""Wrap an :class:`collections.abc.Iterable` with formatted outputs.

        * Middle Output:
          ``{tqdm_header}: [ current / total ] str(self) {memory} {iter_time} {data_time} {time}<{remaining}``
        * Final Output
          ``{header} str(self) {memory} {iter_time} {data_time} {total_time}``

        Args:
            iterable (~collections.abc.Iterable): The raw iterator.
            header (str): The header string for final output.
                Defaults to ``''``.
            tqdm (bool): Whether to use tqdm to show iteration information.
                Defaults to ``self.tqdm``.
            tqdm_header (str): The header string for middle output.
                Defaults to ``'Iter'``.
            indent (int): The space indent for the entire string.
                if ``None``, use ``self.indent``.
                Defaults to ``None``.
            verbose (int): The verbose level of output information.
        """
        tqdm = tqdm if tqdm is not None else self.tqdm
        indent = indent if indent is not None else self.indent
        iterator = iterable
        if len(header) != 0:
            header = header.ljust(30 + get_ansi_len(header))
        if tqdm:
            length = len(str(len(iterable)))
            pattern: str = (
                "{tqdm_header}: {blue_light}"
                "[ {red}{{n_fmt:>{length}}}{blue_light} "
                "/ {red}{{total_fmt}}{blue_light} ]{reset}"
            ).format(tqdm_header=tqdm_header, length=length, **ansi)
            offset = len(f"{{n_fmt:>{length}}}{{total_fmt}}") - 2 * length
            pattern = pattern.ljust(30 + offset + get_ansi_len(pattern))
            time_str = self.get_str(
                time="{elapsed}<{remaining}", cut_too_long=False)
            bar_format = f"{pattern}{{desc}}{time_str}"
            iterator = tqdm_class(iterable, leave=False, bar_format=bar_format)

        self.iter_time.reset()
        self.data_time.reset()
        self.memory.reset()

        end = time.time()
        start_time = time.time()
        for obj in iterator:
            cur_data_time = time.time() - end
            self.data_time.update(cur_data_time)
            yield obj
            cur_iter_time = time.time() - end
            self.iter_time.update(cur_iter_time)
            if torch.cuda.is_available():
                cur_memory = torch.cuda.max_memory_allocated() / MB
                self.memory.update(cur_memory)
            if tqdm:
                _dict = {k: v for k, v in self.meters.items()}
                if verbose > 2 and torch.cuda.is_available():
                    _dict.update(memory=f"{cur_memory:.0f} MB")
                if verbose > 1:
                    _dict.update(
                        iter=f"{cur_iter_time:.3f} s", data=f"{cur_data_time:.3f} s"
                    )
                iterator.set_description_str(
                    self.get_str(**_dict, strip=False))
            end = time.time()
        self.synchronize_between_processes()
        total_time = time.time() - start_time
        total_time_str = tqdm_class.format_interval(total_time)

        _dict = {k: v for k, v in self.meters.items()}
        if verbose > 2 and torch.cuda.is_available():
            _dict.update(memory=f"{str(self.memory)} MB")
        if verbose > 1:
            _dict.update(
                iter=f"{str(self.iter_time)} s", data=f"{str(self.data_time)} s"
            )
        _dict.update(time=total_time_str)
        prints(self.delimiter.join(
            [header, self.get_str(**_dict)]), indent=indent)


class Logger:
    def __init__(self, logger=logging.getLogger(Path(__file__).parent.name)):
        self.logger = logger

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def get_logger(self) -> logging.Logger:
        return self.logger


logger = Logger()
