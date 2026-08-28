import multiprocessing
from typing import Callable

from tqdm import tqdm


class MultiProcessRunner:

    def __init__(self, inputs_list, worker_fn, num_workers=4):
        super().__init__()

        self.num_workers = num_workers
        self.input_queue = multiprocessing.Queue(maxsize=0)
        self.output_queue = multiprocessing.Queue(maxsize=0)
        self.worker_fn = worker_fn
        self.inputs_list = inputs_list

        self.workers = None
        self.workers = self._create_workers()

    def _create_workers(self):
        workers = []

        for i in range(self.num_workers):
            w = multiprocessing.Process(target=_worker_fn_wrapper,
                                        args=(i, self.input_queue, self.output_queue, self.worker_fn))
            w.daemon = True
            w.start()
            workers.append(w)

        return workers

    def _graceful_shutdown(self):
        print('killing all workers...')
        for w in self.workers:
            w.terminate()

    def run(self, description=None):
        ## pre-fill inputs queue
        num_fills = self.num_workers * 2
        for inputs in self.inputs_list[:num_fills]:
            self.input_queue.put(inputs)

        ## start
        pbar = tqdm(total=self.length(), desc=description)
        all_outputs = []
        input_idx = num_fills
        handled_count = 0
        total = self.length()
        while handled_count < total:
            pbar.update(1)
            out = self.output_queue.get()
            if input_idx < total:
                self.input_queue.put(self.inputs_list[input_idx])
                input_idx += 1
            handled_count += 1

            if out is None:
                continue
            all_outputs.append(out)

        pbar.close()
        return all_outputs

    def length(self):
        return len(self.inputs_list)

    def close(self):
        self._graceful_shutdown()


def _worker_fn_wrapper(worker_id: int,
                       input_queue: multiprocessing.Queue,
                       output_queue: multiprocessing.Queue,
                       worker_fn: Callable,
                       ):
    while True:
        inputs = input_queue.get()
        try:
            data = worker_fn(inputs, worker_id)
        except Exception as e:
            print(e)
            data = None
        output_queue.put(data)
