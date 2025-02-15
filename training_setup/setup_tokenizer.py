# from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors


def setup_custom_char_level_tokenizer() -> AutoTokenizer:
    # Create WordLevel tokenizer with an unknown token
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))

    # Set character-level tokenization (isolated splitting)
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")

    # Define vocabulary (all letters, numbers, punctuation, and special tokens)
    vocab_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-+*/=()[]{}<>:;'\" \n\t"

    # Define special tokens
    special_tokens = ["[UNK]", "[PAD]", "[SEP]"]

    # Create trainer
    trainer = trainers.WordLevelTrainer(vocab_size=256, special_tokens=special_tokens)

    # Train tokenizer on our custom string instead of a file
    tokenizer.train_from_iterator([vocab_text], trainer=trainer)

    # Add post-processing template
    tokenizer.post_processor = processors.TemplateProcessing(
        pair="$A [SEP] $B:1",
        special_tokens=[
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    # Enable padding
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        sep_token="[SEP]",
    )

    return fast_tokenizer


SPECIAL_TOKENS_DICT = {
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
}


def setup_tokenizer(cfg_tokenizer: str) -> AutoTokenizer:
    if cfg_tokenizer == "custom_char_level":
        return setup_custom_char_level_tokenizer()
    elif cfg_tokenizer == "none":
        return None

    tokenizer = AutoTokenizer.from_pretrained(
        cfg_tokenizer,
        resume_download=None,
    )
    # tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
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
