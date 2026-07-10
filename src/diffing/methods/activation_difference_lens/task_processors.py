"""Per-organism task-dataset processors for content-edges (task-domain) ADL runs.

The content-edges loader (``load_and_tokenize_chat_content_edges_dataset``) probes
the first/last tokens of a task *question* rendered through the chat template. Each
model organism stores its task inputs in a different shape — med_spurious keeps a
top-level JSON array with the prompt in ``question``; Pando nests 2000 prompt rows
under a ``pool`` key with the prompt in ``prompt``; lottery has yet another layout.

Rather than teaching the shared loader every organism's schema (a ``text_column``
name only covers the flat-array case, and can't reach a nested key), each organism
names a *processor* here. A processor takes the task dataset location and returns a
plain ``list[str]`` of prompt texts; the loader stays schema-agnostic and just
chat-formats + tokenizes whatever strings it gets back.

Wire-up: an organism config declares ``task_processor: <name>`` (see
``configs/organism/<organism>.yaml``); the ADL method looks the name up here via
``get_task_processor``. Add a new organism by registering one function below — no
change to the loader or the runner scripts.

Each processor signature is ``(dataset_name, *, split="train", text_column="text")``
and returns ``list[str]``; unused kwargs are accepted and ignored so the method can
call every processor uniformly.
"""

from typing import Callable, Dict, List
from pathlib import Path
import json

from datasets import load_dataset
from loguru import logger


def _load_rows_local_or_hf(dataset_name: str, split: str):
    """Load a local .json/.jsonl (top-level array) or an HF dataset id as rows."""
    if Path(dataset_name).is_file() and Path(dataset_name).suffix in (".json", ".jsonl"):
        # HF's json loader wants a JSONL or a top-level array; a nested key (e.g.
        # Pando's `pool`) must be handled by a dedicated processor instead.
        return load_dataset("json", data_files=dataset_name, split="train")
    return load_dataset(dataset_name, split=split)


def _texts_from_column(rows, column: str) -> List[str]:
    out: List[str] = []
    for r in rows:
        v = r.get(column)
        if v is not None and str(v).strip():
            out.append(str(v))
    return out


def default_processor(
    dataset_name: str, *, split: str = "train", text_column: str = "text", **_
) -> List[str]:
    """Flat case: top-level JSON array / JSONL / HF dataset, one prompt per row.

    The prompt column defaults to ``text`` (matching the general-domain FineWeb
    layout); pass a different ``text_column`` for other flat datasets.
    """
    rows = _load_rows_local_or_hf(dataset_name, split)
    return _texts_from_column(rows, text_column)


def med_spurious_processor(dataset_name: str, *, split: str = "train", **_) -> List[str]:
    """med_spurious validation/testing sets: JSON array with the prompt in ``question``."""
    return default_processor(dataset_name, split=split, text_column="question")


def pando_processor(dataset_name: str, *, split: str = "train", **_) -> List[str]:
    """Pando validation.json: top-level object with a ``pool`` list of prompt rows.

    HF's json loader can't descend into ``pool`` (it would see one row whose columns
    are the top-level keys), so read the file directly and pull the prompt out of
    each pool entry's ``prompt`` field.
    """
    with open(dataset_name) as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "pool" in obj:
        pool = obj["pool"]
    elif isinstance(obj, list):
        pool = obj
    else:
        raise ValueError(
            f"pando_processor: expected a dict with a 'pool' list or a top-level "
            f"list in {dataset_name}, got {type(obj).__name__}"
        )
    return _texts_from_column(pool, "prompt")


# name (as written in configs/organism/<org>.yaml: task_processor) -> processor fn
TASK_PROCESSORS: Dict[str, Callable[..., List[str]]] = {
    "default": default_processor,
    "med_spurious": med_spurious_processor,
    "pando": pando_processor,
}


def get_task_processor(name: str) -> Callable[..., List[str]]:
    """Resolve a task-processor name (from an organism config) to its function."""
    if name not in TASK_PROCESSORS:
        raise KeyError(
            f"Unknown task_processor '{name}'. Registered: {sorted(TASK_PROCESSORS)}. "
            f"Register a new one in {__file__}."
        )
    logger.info(f"Using task processor '{name}'")
    return TASK_PROCESSORS[name]
