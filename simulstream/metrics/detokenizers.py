from types import SimpleNamespace
from typing import Callable, Dict

from transformers import AutoProcessor


def build_hf_detokenizer(config: SimpleNamespace) -> Callable[[str], str]:
    assert hasattr(config, "hf_model_name"), \
        "`hf_model_name` required in the eval config for `hf` detokenizer"
    processor = AutoProcessor.from_pretrained(config.hf_model_name)

    def detokenize(input_string: str) -> str:
        return processor.tokenizer.convert_tokens_to_string(input_string)

    return detokenize


_DETOKENIZER_BUILDER_MAP: Dict[str, Callable[[SimpleNamespace], Callable[[str], str]]] = {
    "hf": build_hf_detokenizer,
}


def get_detokenizer(config: SimpleNamespace) -> Callable[[str], str]:
    return _DETOKENIZER_BUILDER_MAP[config.detokenizer_type](config)
