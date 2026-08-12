import base64
import io

from PIL import Image
from sentence_transformers import SentenceTransformer

from simple_local.runtimes.custom import ModelContext

DATA_URL_PREFIX = "data:"


class VLEmbedder:
    def load(self, ctx: ModelContext) -> None:
        self.name = ctx.name
        settings = ctx.config
        self.default_prompt = settings.get("prompt")
        self.dimensions = settings.get("dimensions")
        self.batch_size = settings.get("batch_size", 4)
        model_id = str(ctx.path) if ctx.path else settings.get("model", "Qwen/Qwen3-VL-Embedding-2B")
        self.model = SentenceTransformer(
            model_id,
            device=settings.get("device"),
            model_kwargs={"dtype": settings.get("dtype", "auto")},
        )

    def predict(self, request: dict) -> dict:
        inputs = request.get("input")
        if inputs is None:
            raise ValueError("body must contain 'input'")
        if not isinstance(inputs, list):
            inputs = [inputs]
        if not inputs:
            raise ValueError("'input' must not be empty")

        items = [_to_item(item, i) for i, item in enumerate(inputs)]
        prompt = request.get("prompt", self.default_prompt)
        vectors = self.model.encode(
            items,
            prompt=prompt,
            batch_size=request.get("batch_size", self.batch_size),
            normalize_embeddings=request.get("normalize", True),
            show_progress_bar=False,
        )

        dimensions = request.get("dimensions", self.dimensions)
        if dimensions:
            vectors = vectors[:, :dimensions]

        return {
            "object": "list",
            "model": self.name,
            "data": [
                {"object": "embedding", "index": i, "embedding": vector.tolist()}
                for i, vector in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }


def _to_item(item, index: int):
    """Accept a plain string, {"text": ...}, {"image": ...}, or both. Images may
    be an http(s) URL, a local path, or a data: URL — sentence-transformers
    handles the first two, so only base64 needs decoding here."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        raise ValueError(f"input[{index}] must be a string or an object")

    text, image = item.get("text"), item.get("image")
    if text is None and image is None:
        raise ValueError(f"input[{index}] needs 'text', 'image', or both")
    if image is None:
        return text

    if isinstance(image, str) and image.startswith(DATA_URL_PREFIX):
        image = _decode_data_url(image, index)
    return {"text": text, "image": image} if text is not None else image


def _decode_data_url(url: str, index: int) -> Image.Image:
    _, _, encoded = url.partition(",")
    try:
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    except Exception as e:
        raise ValueError(f"input[{index}] has an unreadable data: URL: {e}") from e
