import logging
import threading
import urllib.error
import urllib.request

from ..config import ModelSpec

log = logging.getLogger("simple_local.remote")


class RemoteRuntime:
    """Forwards to an OpenAI-compatible endpoint someone else runs — a model on
    Modal, a GPU box, a hosted provider. Nothing to supervise: the process, its
    memory, and its heat live elsewhere."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.remote = spec.remote
        self.base_url = self.remote.url
        self.timeout = self.remote.timeout
        self.upstream_model = self.remote.model or spec.name
        self.adapter_ids: dict[str, int] = {}
        # Modal answers 303 once a web request passes 150s (a cold start easily
        # does) and expects the client to follow it to a URL that blocks until
        # the work finishes. Without this a slow first call surfaces as a 303.
        self.follow_redirects = True
        self.ready = threading.Event()
        self.ready.set()  # reachability is proven per request, not by a supervisor
        log.info("%s: forwarding to %s", spec.name, self.base_url)

    def endpoint(self, path: str) -> str:
        return f"{self.base_url}/{path}"

    def upstream_headers(self) -> dict[str, str]:
        if not self.remote.api_key:
            return {}
        return {"Authorization": f"Bearer {self.remote.api_key}"}

    def reachable(self) -> bool:
        request = urllib.request.Request(
            self.endpoint("models"), headers=self.upstream_headers()
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status < 500
        except urllib.error.HTTPError as e:
            return e.code < 500  # a 401/404 still means something is listening
        except (urllib.error.URLError, OSError):
            return False
