from typing import Dict, Any

# from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer


SPECIAL_TOKENS_DICT = {
    "eos_token": "[EOS]",
    "bos_token": "[BOS]",
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
}


def setup_tokenizer(cfg_tokenizer: Dict[str, Any]) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg_tokenizer.name,
        resume_download=None,
    )
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    # if tokenizer in ["gpt2", "bert-base-uncased"]:
    #     tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    #     tokenizer._tokenizer.post_processor = (  # pylint: disable=protected-access
    #         TemplateProcessing(
    #             single=f"{tokenizer.bos_token} $A {tokenizer.eos_token}",
    #             special_tokens=[
    #                 (tokenizer.eos_token, tokenizer.eos_token_id),
    #                 (tokenizer.bos_token, tokenizer.bos_token_id),
    #             ],
    #         )
    #     )
    return tokenizer
