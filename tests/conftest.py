import pytest


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """A randomly initialized one-layer Qwen3 and a word-level tokenizer, saved to disk."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    path = tmp_path_factory.mktemp("tiny")
    vocabulary = {w: i for i, w in enumerate([
        "[UNK]", "[PAD]", "[EOS]", "start", "A", "B", "First", "Child",
        "Parent", "Name", "Other", "'", "s", "parent", "is", "child", "."])}
    tokenizer = tokenizers.Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    torch.manual_seed(7)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, tie_word_embeddings=True))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def tiny_config(tiny_model):
    return {"model": str(tiny_model), "revision": "main", "lora_rank": 2, "lora_alpha": 4,
            "lora_dropout": 0.05, "batch_size": 2, "learning_rate": 0.001, "answer_slots": 4,
            "sampling_seed": 0, "warm_steps": 2, "sft_steps": 2}
