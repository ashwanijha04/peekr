from __future__ import annotations

"""
A/B experiment decorator for peekr.

Tags each call with the chosen variant in the current span's attributes
so you can compare results via SQL:

    -- Compare error rate by variant
    SELECT json_extract(attributes,'$.experiment_variant') variant,
           AVG(CASE WHEN status='error' THEN 1.0 ELSE 0.0 END) error_rate,
           AVG(json_extract(attributes,'$.tokens_total')) avg_tokens,
           AVG(duration_ms) avg_ms
    FROM spans
    WHERE json_extract(attributes,'$.experiment_variant') IS NOT NULL
    GROUP BY variant;

Usage — list form (variant name injected as ``variant`` kwarg):

    @experiment(variants=["control", "test_v2"], split=[0.5, 0.5])
    def run_agent(query: str, variant: str):
        model = "gpt-4o" if variant == "control" else "claude-opus-4-5"
        return call_llm(model, query)

Usage — dict form (key injected as ``variant``, value as ``variant_config``):

    @experiment(variants={"control": {"model": "gpt-4o"}, "test": {"model": "claude-opus-4-5"}})
    def run_agent(query: str, variant: str, variant_config: dict):
        return call_llm(variant_config["model"], query)
"""
import asyncio  # noqa: E402
import functools  # noqa: E402
import random  # noqa: E402
from typing import Callable  # noqa: E402

from .context import get_current_span, start_span, end_span  # noqa: E402
from .exporters import export_span  # noqa: E402


def _pick_variant(variants, split):
    """Return (variant_key, variant_config_or_None)."""
    if isinstance(variants, dict):
        keys = list(variants.keys())
        weights = split if split is not None else [1.0 / len(keys)] * len(keys)
        chosen_key = random.choices(keys, weights=weights, k=1)[0]
        return chosen_key, variants[chosen_key]
    else:
        weights = split if split is not None else [1.0 / len(variants)] * len(variants)
        chosen = random.choices(variants, weights=weights, k=1)[0]
        return chosen, None


def _tag_span(variant_key: str) -> None:
    """Write experiment_variant into the active span if one exists."""
    span = get_current_span()
    if span is not None:
        span.attributes["experiment_variant"] = variant_key


def experiment(_func=None, *, variants, split=None):
    """
    Decorator for A/B testing prompts or models.

    Parameters
    ----------
    variants:
        - list[str]  — variant names; chosen name is injected as ``variant`` kwarg.
        - dict       — mapping of name → config; chosen name is injected as
                       ``variant`` kwarg and config dict as ``variant_config`` kwarg.
    split:
        Optional list of weights (same length as variants).  Equal split if None.
    """
    is_dict = isinstance(variants, dict)

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                span, token = start_span(f"experiment.{func.__name__}")
                try:
                    variant_key, variant_cfg = _pick_variant(variants, split)
                    span.attributes["experiment_variant"] = variant_key
                    kwargs["variant"] = variant_key
                    if is_dict:
                        kwargs["variant_config"] = variant_cfg
                    result = await func(*args, **kwargs)
                    span.status = "ok"
                    return result
                except Exception as e:
                    span.status = "error"
                    span.attributes["error"] = str(e)
                    raise
                finally:
                    end_span(span, token)
                    export_span(span)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                span, token = start_span(f"experiment.{func.__name__}")
                try:
                    variant_key, variant_cfg = _pick_variant(variants, split)
                    span.attributes["experiment_variant"] = variant_key
                    kwargs["variant"] = variant_key
                    if is_dict:
                        kwargs["variant_config"] = variant_cfg
                    result = func(*args, **kwargs)
                    span.status = "ok"
                    return result
                except Exception as e:
                    span.status = "error"
                    span.attributes["error"] = str(e)
                    raise
                finally:
                    end_span(span, token)
                    export_span(span)

            return sync_wrapper

    if _func is not None:
        return decorator(_func)
    return decorator
