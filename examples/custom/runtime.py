import statistics

from simple_local.runtimes.custom import ModelContext


class DemoEnsemble:
    def load(self, ctx: ModelContext) -> None:
        self.name = ctx.name
        self.rates = ctx.config.get("rate_per_kva", {"default": 40.0})
        self.phase_multiplier = ctx.config.get("phase_multiplier", {"1ph": 1.0, "3ph": 1.35})
        self.version = ctx.version or "dev"

    def predict(self, request: dict):
        specs = request.get("specs")
        if specs is not None:
            if not isinstance(specs, list):
                raise ValueError("'specs' must be a list")
            return self._predict_batch(specs)
        return self._predict_one(request)

    def _predict_batch(self, specs: list):
        for index, spec in enumerate(specs):
            result = self._predict_one(spec)
            yield {"index": index, **result}

    def _predict_one(self, spec: dict) -> dict:
        kva = spec.get("kva")
        if not isinstance(kva, (int, float)) or kva <= 0:
            raise ValueError("spec requires a positive numeric 'kva'")
        company = spec.get("company", "default")
        multiplier = self.phase_multiplier.get(spec.get("phase", "1ph"), 1.0)

        rate = self.rates.get(company, self.rates["default"])
        estimates = {
            "rate_card": kva * rate * multiplier,
            "power_curve": (kva ** 0.92) * rate * 1.18 * multiplier,
        }
        costs = list(estimates.values())
        blended = statistics.fmean(costs)
        spread = statistics.stdev(costs) if len(costs) > 1 else 0.0
        return {
            "predicted_cost": round(blended, 2),
            "std_dev": round(spread, 2),
            "confidence": "high" if spread / blended < 0.1 else "medium",
            "method_estimates": {k: round(v, 2) for k, v in estimates.items()},
            "model_version": self.version,
        }
