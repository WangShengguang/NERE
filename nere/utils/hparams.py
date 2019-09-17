import argparse

from setproctitle import setproctitle


class Hparams(object):
    ''' Parse command line arguments and execute the code'''
    parser = argparse.ArgumentParser(description="基础，通用parser")
    # logging config 日志配置
    parser.add_argument('--stream_log', action="store_true", help="是否将日志信息输出到标准输出")  # log print到屏幕
    parser.add_argument('--log_level', default="info", type=str, help="日志级别")
    parser.add_argument('--allow_gpus', default="0,1,2,3", type=str,
                        help="指定GPU编号，0 or 0,1,2 or 0,1,2...7  | nvidia-smi 查看GPU占用情况")
    parser.add_argument('--cpu_only', action="store_true", help="CPU only, not to use GPU ")
    parser.add_argument('--process_name', default="Hello World!", type=str, help="日志级别")


def set_process_name(process_name):
    setproctitle(process_name)
    print("process name: {}".format(process_name))
