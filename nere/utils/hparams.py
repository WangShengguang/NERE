import argparse

from setproctitle import setproctitle

from .logger import logging_config


class Hparams(object):
    ''' Parse command line arguments and execute the code'''
    parser = argparse.ArgumentParser()
    # logging config 日志配置
    parser.add_argument('--stream_log', default=False, type=bool, help="是否将日志信息输出到标准输出")  # log print到屏幕
    parser.add_argument('--log_level', default="info", type=str, help="日志级别")
    parser.add_argument('--allow_gpus', default="0,1,2,3", type=str,
                        help="指定GPU编号，0 or 0,1,2 or 0,1,2...7  | nvidia-smi 查看GPU占用情况")
    parser.add_argument('--cpu_only', action="store_true", help="CPU only, not to use GPU ")
    parser.add_argument('--process_name', default="Hello World!", type=str, help="日志级别")


def set_process_name(process_name):
    setproctitle(process_name)
    print("process name: {}".format(process_name))


class ArgCallback(object):
    """
    要求参数和函数名必须同名，log name也将和函数名同名
    """

    def __init__(self, parsed_args_dict, model_name="manage", *args, **kwargs):
        """ 在指定模块找到，parsed_args_dict 参数为True的同名函数，调用这个函数
        :param parsed_args_dict: 参数字典
        :param model_name: 模块名，模块路径，导入模块：__import__(model_name)
        :param args: 传给函数的位置参数
        :param kwargs: 传给函数的关键字参数
        """
        may_func_names = [k for k, v in parsed_args_dict.items() if v]
        model = __import__(model_name)  # if the class in current model, model_name == __name__
        model_func_names = [key for key, attr in model.__dict__.items() if attr.__class__.__name__ == "function"]
        func_names = set(may_func_names) & set(model_func_names)
        if len(func_names) != 1:  # 一次只能调用一个函数
            raise ValueError("Please check your input parms")
        func_name = list(func_names)[0]
        logging_path = logging_config("{}.log".format(func_name),
                                      stream_log=parsed_args_dict.get("stream_log", False),
                                      log_level=parsed_args_dict.get("log_level", "info"))
        print("\n*** calling: {}".format(func_name))
        print("*** logging_path: {}".format(logging_path))
        getattr(model, func_name)(*args, **kwargs)  # 函数调用
