import logging
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


class ShowTime(object):
    '''
    用上下文管理器计时
    '''
    import time, logging
    time = time
    logging = logging

    def __init__(self, prefix="", ipdb=False):
        self.prefix = prefix
        self.ipdb = ipdb

    def __enter__(self):
        self.t1 = self.time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.runtime = self.time.time() - self.t1
        print("{} take time: {:.2f} s".format(self.prefix, self.runtime))
        if exc_type is not None:
            print(exc_type, exc_val, exc_tb)
            import traceback
            print(traceback.format_exc())
            if self.ipdb:
                import ipdb
                ipdb.set_trace()
            return self


if __name__ == "__main__":
    with ShowTime("hello", ipdb=False) as g:
        print("*** g.runtime: {}".format(getattr(g, "runtime", "")))
        import time

        time.sleep(2)
        raise ValueError(0)  # 被忽略
        # print(g.t1)
        # raise ValueError(0)
    g = ShowTime("hello", ipdb=True)
    with g:
        import time

        time.sleep(3)
        raise ValueError("222")
