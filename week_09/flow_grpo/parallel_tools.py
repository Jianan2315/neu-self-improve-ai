"""Concurrent tool I/O with all Qwen generation owned by the calling thread."""
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from queue import Empty, Queue
from threading import Event
from agent_flow.flow import request
from agent_flow.tools import Toolset


class ConcurrentToolset(Toolset):
    def ask(self, role, **context):
        """Workers submit helper prompts; only the coordinator calls the model."""
        answer = Future()
        self.requests.put((request(role, **context), answer))
        while not self.stopping.is_set():
            try:
                return answer.result(timeout=0.1)
            except TimeoutError:
                pass
        raise RuntimeError("Concurrent tool execution stopped.")

    def execute_many(self, commands):
        """Return results in command order, even when requests finish out of order."""
        if not commands:
            return []
        self.requests = Queue()
        self.stopping = Event()
        pool = ThreadPoolExecutor(max_workers=self.settings["tool_workers"])

        def execute(command):
            if self.stopping.is_set():
                raise RuntimeError("Concurrent tool execution stopped.")
            return self.execute(*command)

        futures = []
        try:
            futures = [pool.submit(execute, command) for command in commands]
            while True:
                # Propagate a failed service call instead of assigning a reward.
                for future in futures:
                    if future.done():
                        future.result()
                if all(future.done() for future in futures):
                    return [future.result() for future in futures]
                try:
                    pending = [self.requests.get(timeout=0.02)]
                except Empty:
                    continue
                # Briefly collect helper prompts that become ready together.
                self.stopping.wait(0.01)
                while True:
                    try:
                        pending.append(self.requests.get_nowait())
                    except Empty:
                        break
                generated = self.generate([prompt for prompt, _ in pending], 0.0,
                                          self.settings["module_max_new_tokens"])
                if len(generated) != len(pending):
                    raise RuntimeError("Qwen returned an incomplete helper batch.")
                for (_, answer), row in zip(pending, generated):
                    answer.set_result(row["text"])
        finally:
            # Release workers waiting for Qwen if generation or a service fails.
            self.stopping.set()
            pool.shutdown(wait=True, cancel_futures=True)
