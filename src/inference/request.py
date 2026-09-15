import threading


class RequestState:

    def __init__(self, request_id, prompt, max_new_tokens=20):
        self.request_id = request_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

        self.input_ids = None
        self.cache = None

        self.current_token = None
        self.generated_tokens = []

        self.finished = False
        self.finish_reason = None

        # Used by the HTTP request to wait for completion.
        self.done_event = threading.Event()

    def add_token(self, token_id):
        self.generated_tokens.append(token_id)

    def mark_finished(self, reason):
        self.finished = True
        self.finish_reason = reason
        self.done_event.set()