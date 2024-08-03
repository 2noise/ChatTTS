import platform, sys
import logging
from datetime import datetime, timezone

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("wetext-zh_normalizer").setLevel(logging.WARNING)
logging.getLogger("NeMo-text-processing").setLevel(logging.WARNING)

# from https://github.com/FloatTech/ZeroBot-Plugin/blob/c70766a989698452e60e5e48fb2f802a2444330d/console/console_windows.go#L89-L96
colorCodePanic = "\x1b[1;31m"
colorCodeFatal = "\x1b[1;31m"
colorCodeError = "\x1b[31m"
colorCodeWarn = "\x1b[33m"
colorCodeInfo = "\x1b[37m"
colorCodeDebug = "\x1b[32m"
colorCodeTrace = "\x1b[36m"
colorReset = "\x1b[0m"

log_level_color_code = {
    logging.DEBUG: colorCodeDebug,
    logging.INFO: colorCodeInfo,
    logging.WARN: colorCodeWarn,
    logging.ERROR: colorCodeError,
    logging.FATAL: colorCodeFatal,
}

log_level_msg_str = {
    logging.DEBUG: "DEBU",
    logging.INFO: "INFO",
    logging.WARN: "WARN",
    logging.ERROR: "ERRO",
    logging.FATAL: "FATL",
}


class Formatter(logging.Formatter):
    def __init__(self, color=platform.system().lower() != "windows"):
        # https://stackoverflow.com/questions/2720319/python-figure-out-local-timezone
        self.tz = datetime.now(timezone.utc).astimezone().tzinfo
        self.color = color

    def format(self, record: logging.LogRecord):
        logstr = "[" + datetime.now(self.tz).strftime("%z %Y%m%d %H:%M:%S") + "] ["
        if self.color:
            logstr += log_level_color_code.get(record.levelno, colorCodeInfo)
        logstr += log_level_msg_str.get(record.levelno, record.levelname)
        if self.color:
            logstr += colorReset
        if sys.version_info >= (3, 9):
            fn = record.filename.removesuffix(".py")
        elif record.filename.endswith(".py"):
            fn = record.filename[:-3]
        logstr += f"] {str(record.name)} | {fn} | {str(record.msg)%record.args}"
        return logstr


def get_logger(name: str, lv=logging.INFO, remove_exist=False, format_root=False):
    logger = logging.getLogger(name)
    logger.setLevel(lv)
    if remove_exist and logger.hasHandlers():
        logger.handlers.clear()
    if not logger.hasHandlers():
        syslog = logging.StreamHandler()
        syslog.setFormatter(Formatter())
        logger.addHandler(syslog)
    else:
        for h in logger.handlers:
            h.setFormatter(Formatter())
    if format_root:
        for h in logger.root.handlers:
            h.setFormatter(Formatter())
    return logger
