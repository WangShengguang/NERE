import logging
import time
import traceback
from functools import wraps


def try_catch_with_logging(default_response=None):
    def out_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                res = func(*args, **kwargs)
            except Exception:
                res = default_response
                logging.error(traceback.format_exc())
            return res

        return wrapper

    return out_wrapper


def todict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif isinstance(obj, (list, set, tuple)):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, todict(value, classkey))
                     for key, value in obj.__dict__.items()
                     if not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj


class GetTime(object):
    '''
    用上下文管理器计时; 行内调试等
    '''

    def __init__(self, prefix=""):
        self.prefix = prefix

    def __enter__(self):
        self.t1 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.runtime = time.time() - self.t1
        time_log = "{} take time: {:.2f} s".format(self.prefix, self.runtime)
        print(time_log)
        logging.info(time_log)
        if exc_type is not None:
            print(exc_type, exc_val, exc_tb)
            error_log = traceback.format_exc()
            print(error_log)
            logging.error(error_log)
            return self


if __name__ == "__main__":
    with GetTime("hello") as g:
        print("*** g.runtime: {}".format(getattr(g, "runtime", "")))
        time.sleep(2)
        aaa = 1
        raise ValueError(0)  # 被忽略
        # print(g.t1)
        # raise ValueError(0)
    print("--" * 50)
    g = GetTime("hello")
    with g:
        time.sleep(3)
        bbb = 2
        raise ValueError("222")
