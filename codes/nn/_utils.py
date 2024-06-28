import threading
import queue
from functools import wraps

import numpy as np
import torch.multiprocessing as mp


class _Func:
    def __init__(self, func):
        self.func = func

    def __call__(self, kwds):
        return self.func(**kwds)


class Parallel:
    def __init__(self, func, n_jobs):
        self.func = func
        self.n_jobs = n_jobs

    def single_process_wrapper(self, iterable):
        return [self.func(**kwds) for kwds in iterable]

    def multi_process_wrapper(self, iterable):
        with mp.Pool(processes=self.n_jobs) as pool:
            return pool.map(_Func(self.func), iterable)

    def __call__(self, iterable):
        if self.n_jobs == 0 or self.n_jobs == 1 or self.n_jobs is None:
            return self.single_process_wrapper(iterable)
        else:
            return self.multi_process_wrapper(iterable)


class ExceptionHandler:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwds):
        try:
            result = self.func(*args, **kwds)
        except Exception as e:
            result = e
        return result


def subprogram_wrapper(result_queue, func, args, kwargs):
    try:
        result = func(*args, **kwargs)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)


class TimeoutExceptionHandler:
    def __init__(self, func, timeout):
        self.func = func
        self.timeout = timeout

    def __call__(self, *args, **kwds):
        return (
            self.no_timeout(*args, **kwds)
            if self.timeout is None
            else self.timeout_handler(*args, **kwds)
        )

    def no_timeout(self, *args, **kwds):
        print("No timeout")
        try:
            return self.func(*args, **kwds)
        except Exception as e:
            return e

    def timeout_handler(self, *args, **kwds):
        result_queue = queue.Queue()
        thread = threading.Thread(
            target=subprogram_wrapper, args=(result_queue, self.func, args, kwds)
        )
        thread.start()

        thread.join(self.timeout)

        if thread.is_alive():
            print(
                f"Function exceeded timeout of {self.timeout} seconds and will be stopped."
            )
            return TimeoutError("Function execution exceeded the timeout limit")
        else:
            if not result_queue.empty():
                result = result_queue.get()
                return result


def make_loss_fn(focal, idx2weight=None):
    apply_class_balancing = idx2weight is not None
    alpha = idx2weight[1] if apply_class_balancing else None
    return keras.losses.BinaryFocalCrossentropy(
        gamma=focal,
        from_logits=False,
        alpha=alpha,
        apply_class_balancing=apply_class_balancing,
    )


def make_class_weights(y_true):
    classes, counts = np.unique(y_true, return_counts=True)
    num_samples = len(y_true)
    num_classes = len(classes)

    class_weights = num_classes / (num_classes * counts)
    return class_weights
