from evaluate import EvaluationModule, load
from typing import cast, TypedDict


__all__ = (
    "compute_perplexity_zephyr_3b",
    "EvaluationResult",
)


__perplexity: EvaluationModule | None = None


def __load_or_get_perplexity_module() -> EvaluationModule:
    global __perplexity
    if __perplexity is None:
        __perplexity = load("perplexity", module_type="metric")
    return __perplexity


class EvaluationResult(TypedDict):
    mean_perplexity: float
    perplexities: list[float]


def _compute_perplexity(model_name: str, predictions: list[str], /) -> EvaluationResult:
    ret_val = __load_or_get_perplexity_module().compute(predictions=predictions, model_id=model_name)
    assert isinstance(ret_val, dict)
    assert frozenset(ret_val.keys()) == {"mean_perplexity", "perplexities"}
    assert isinstance(ret_val["mean_perplexity"], float)
    assert isinstance(ret_val["perplexities"], list)
    assert all(isinstance(x, float) for x in ret_val["perplexities"])
    return cast(EvaluationResult, ret_val)


def compute_perplexity_zephyr_3b(predictions: list[str], /) -> EvaluationResult:
    return _compute_perplexity("stabilityai/stablelm-zephyr-3b", predictions)
