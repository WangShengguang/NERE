import logging.handlers

import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(cur_dir))

# logs_dir = os.path.join(root_dir, "logs")

logs_dir = root_dir


def logging_config(logging_name='./run.log', stream_log=False, relative_path=".", level="info"):
    """
    :param logging_name:  log名
    :param stream_log: 是否把log信息输出到屏幕,标准输出
    :param relative_path: 相对路径，log文件相对于logs的位置（父目录，当前目录等）
    :param level: fatal,error,warn,info,debug
    :return: None
    """
    logging_path = os.path.join(logs_dir, relative_path, os.path.basename(logging_name))
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_handles = [logging.handlers.RotatingFileHandler(logging_path,
                                                        maxBytes=20 * 1024 * 1024, backupCount=5, encoding='utf-8')]
    if stream_log:
        log_handles.append(logging.StreamHandler())
    logging_level = {"fatal": logging.FATAL, "error": logging.ERROR, "warn": logging.WARN,
                     "info": logging.INFO, "debug": logging.DEBUG}[level]
    logging.basicConfig(
        handlers=log_handles,
        level=logging_level,
        format="%(asctime)s - %(levelname)s %(filename)s %(funcName)s %(lineno)s - %(message)s"
    )
    return logging_path


if __name__ == "__main__":
    logging_config("./test.log", stream_log=True)  # ../../log/test.log
    logging.info("标准输出 log ...")
    logging.debug("hello")
