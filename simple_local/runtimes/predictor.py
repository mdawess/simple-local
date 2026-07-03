from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import ValidationError, create_model

from ..config import Config

TYPES = {"float": float, "int": int, "bool": bool, "str": str}


class PredictorRuntime:
    def __init__(self, cfg: Config, model_path: Path):
        self.model = joblib.load(model_path)
        self.task = cfg.predictor.task

        self.features: list[str] | None = None
        self._row_model = None
        schema = cfg.predictor.input_schema
        if schema and schema.features:
            self.features = [f.name for f in schema.features]
            fields = {f.name: (TYPES[f.type], ...) for f in schema.features}
            self._row_model = create_model("PredictRow", **fields)

    def _matrix(self, inputs: list) -> np.ndarray:
        if self.features is not None:
            rows = []
            for i, row in enumerate(inputs):
                try:
                    valid = self._row_model(**row).model_dump()
                except (ValidationError, TypeError) as e:
                    raise ValueError(f"input row {i}: {e}") from e
                rows.append([valid[name] for name in self.features])
            return np.array(rows)
        return np.array(inputs)

    def predict(self, inputs: list) -> dict[str, Any]:
        X = self._matrix(inputs)
        result: dict[str, Any] = {"predictions": self.model.predict(X).tolist()}
        if self.task == "classification" and hasattr(self.model, "predict_proba"):
            result["probabilities"] = self.model.predict_proba(X).tolist()
        return result
