import threading
import time

from src.inference.request import RequestState


class RequestManager:

    def __init__(self, engine):
        self.engine = engine

        self._lock = threading.Lock()
        self._counter = 0

        self._worker = threading.Thread(
            target=self._run,
            daemon=True
        )

        self._worker.start()

    def submit(
        self,
        prompt,
        max_new_tokens=50
    ):
        with self._lock:
            request_id = f"req-{self._counter}"
            self._counter += 1

            request = RequestState(
                request_id=request_id,
                prompt=prompt,
                max_new_tokens=max_new_tokens
            )

            self.engine.scheduler.add_request(request)

        return request

    def _run(self):
        """
        Continuously execute scheduler steps.

        The scheduler performs:
            admission
            prefix-aware prefill
            dynamic batching
            decode
            completion
        """

        while True:

            if self.engine.scheduler.has_work():

                self.engine.scheduler.step()

            else:

                # Avoid a tight CPU loop while idle.
                time.sleep(0.001)