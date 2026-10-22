import logging
import os
from threading import Lock

import openai
from datasets import DatasetDict, load_dataset, load_from_disk

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Silence noisy per-request logs from SDK/http stack.
for _logger_name in ("httpx", "httpcore", "openai", "openai._base_client"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)


def load_system_prompt(system_prompt, default_prompt=DEFAULT_SYSTEM_PROMPT):
    raw = (system_prompt or "").strip()
    if not raw:
        return default_prompt
    if os.path.isfile(raw):
        with open(raw, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            raise ValueError(f"System prompt file is empty: {raw}")
        return text
    return raw


# def load_data_split(source, split, cache_dir, name=None):
#     is_local = bool(source) and os.path.exists(source)
#     if is_local:
#         loaded = load_from_disk(source)
#         if isinstance(loaded, DatasetDict):
#             return loaded[split]
#         return loaded
#     return load_dataset(source, split=split, cache_dir=cache_dir, name=name)

def load_data_split(source, split, cache_dir, name=None):
    is_local = bool(source) and os.path.exists(source)

    if is_local:

        # 如果是 HuggingFace dataset
        if os.path.exists(os.path.join(source, "dataset_dict.json")):
            loaded = load_from_disk(source)
            if isinstance(loaded, DatasetDict):
                return loaded[split]
            return loaded

        # 否则尝试读取 parquet
        parquet_files = [f for f in os.listdir(source) if f.endswith(".parquet")]
        if parquet_files:
            dataset = load_dataset(
                "parquet",
                data_files=os.path.join(source, "*.parquet"),
                split=split,
            )
            return dataset

    return load_dataset(source, split=split, cache_dir=cache_dir, name=name)


def get_judge_runtime_info():
    return {
        "model": os.environ.get("JUDGE_MODEL", ""),
        "base_url": os.environ.get("JUDGE_MODEL_BASE_URL", ""),
        "api_key_set": bool(os.environ.get("JUDGE_MODEL_API_KEY", "")),
    }


def _judge_config_error(reason):
    info = get_judge_runtime_info()
    return RuntimeError(
        f"{reason} | JUDGE_MODEL='{info['model']}', "
        f"JUDGE_MODEL_BASE_URL='{info['base_url']}', "
        f"JUDGE_MODEL_API_KEY_set={info['api_key_set']}"
    )


def init_judge_client_or_raise():
    model = os.environ.get("JUDGE_MODEL", "")
    base_url = os.environ.get("JUDGE_MODEL_BASE_URL", "")
    api_key = os.environ.get("JUDGE_MODEL_API_KEY", "")
    if not model:
        raise _judge_config_error("Judge is not configured: missing JUDGE_MODEL")
    if not api_key:
        raise _judge_config_error("Judge is not configured: missing/invalid JUDGE_MODEL_API_KEY")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)
    try:
        client.models.list()
    except Exception as e:
        raise _judge_config_error(f"Judge preflight failed on models.list(): {e}") from e
    return client, model, base_url
    # return None, None, None


def new_judge_stats():
    return {
        "lock": Lock(),
        "calls": 0,
        "success": 0,
        "failed": 0,
        "api_attempts": 0,
        "exceptions": 0,
    }


def _inc_stat(stats, key, value=1):
    if stats is None:
        return
    lock = stats.get("lock")
    if lock is None:
        stats[key] = stats.get(key, 0) + value
        return
    with lock:
        stats[key] = stats.get(key, 0) + value


def summarize_judge_stats(stats):
    if not stats:
        return None
    lock = stats.get("lock")
    if lock is None:
        calls = int(stats.get("calls", 0))
        success = int(stats.get("success", 0))
        failed = int(stats.get("failed", 0))
        api_attempts = int(stats.get("api_attempts", 0))
        exceptions = int(stats.get("exceptions", 0))
    else:
        with lock:
            calls = int(stats.get("calls", 0))
            success = int(stats.get("success", 0))
            failed = int(stats.get("failed", 0))
            api_attempts = int(stats.get("api_attempts", 0))
            exceptions = int(stats.get("exceptions", 0))
    success_rate = (success / calls * 100) if calls > 0 else 0.0
    return {
        "calls": calls,
        "success": success,
        "failed": failed,
        "api_attempts": api_attempts,
        "exceptions": exceptions,
        "success_rate": success_rate,
    }


def format_judge_summary(stats):
    summary = summarize_judge_stats(stats)
    if not summary:
        return "[Judge Summary] no_stats"
    calls = summary["calls"]
    success = summary["success"]
    failed = summary["failed"]
    api_attempts = summary["api_attempts"]
    exceptions = summary["exceptions"]
    success_rate = summary["success_rate"]
    if calls == 0:
        status = "ℹ️ NO_CALLS"
    elif failed == 0 and exceptions == 0 and success == calls:
        status = "✅ ALL_OK"
    elif success > 0:
        status = "⚠️ PARTIAL"
    else:
        status = "❌ FAILED"
    return (
        f"[Judge Summary] {status} "
        f"calls={calls}, success={success}, failed={failed}, "
        f"api_attempts={api_attempts}, exceptions={exceptions}, "
        f"success_rate={success_rate:.2f}%"
    )


def get_chat_response(
    client,
    model,
    prompt,
    max_token=256,
    retry=5,
    temperature=None,
    judge_stats=None,
):
    if client is None or not model:
        raise RuntimeError("Judge client/model is required")

    _inc_stat(judge_stats, "calls", 1)
    messages = [{"role": "user", "content": prompt}]
    for i in range(retry):
        try:
            _inc_stat(judge_stats, "api_attempts", 1)
            temp = temperature if temperature is not None else 0.5 * i
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                max_tokens=max_token,
            )
            prediction = completion.choices[0].message.content.strip()
            if prediction != "" and prediction is not None:
                _inc_stat(judge_stats, "success", 1)
                return prediction
        except Exception:
            _inc_stat(judge_stats, "exceptions", 1)
    _inc_stat(judge_stats, "failed", 1)
    return ""
