"""The masked-diffusion student of Section 2.3 and Appendix A. Needs the train extra."""

import random

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


MASK_TOKEN = "<|extra_mask|>"


def bidirectional_mask(attention, dtype):
    keep = attention[:, None, None, :].to(dtype)
    return (1.0 - keep) * torch.finfo(dtype).min


def build_student(config, device="cuda", adapter=None):
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
    mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=config["revision"], dtype=dtype,
        attn_implementation="eager").to(device)
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        embedding = model.get_input_embeddings().weight
        embedding[mask_id] = embedding[:-1].mean(0)
        head = model.get_output_embeddings().weight
        head[mask_id] = head[:-1].mean(0)
    if adapter is None:
        lora = LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                          lora_dropout=config["lora_dropout"], task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lora)
    else:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return model, tokenizer, mask_id, dtype


def collate_answers(tokenizer, mask_id, batch, slots, rng):
    dot = tokenizer(".", add_special_tokens=False).input_ids
    encoded = []
    for prompt, answer in batch:
        prefix = tokenizer(prompt, add_special_tokens=True).input_ids
        name = tokenizer(" " + answer, add_special_tokens=False).input_ids[:slots]
        name += [tokenizer.pad_token_id] * (slots - len(name))
        encoded.append((prefix, name))
    length = max(len(p) + len(a) + len(dot) for p, a in encoded)
    ids = torch.full((len(batch), length), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for row, (prefix, answer) in enumerate(encoded):
        ids[row, :len(prefix)] = torch.tensor(prefix)
        attention[row, :len(prefix) + slots + len(dot)] = 1
        masked = set(rng.sample(range(slots), rng.randint(1, slots)))
        for j, token in enumerate(answer):
            position = len(prefix) + j
            ids[row, position] = mask_id if j in masked else token
            if j in masked:
                labels[row, position] = token
        ids[row, len(prefix) + slots:len(prefix) + slots + len(dot)] = torch.tensor(dot)
    return ids, attention, labels


def train_phase(model, tokenizer, mask_id, dtype, samples, steps, config, device):
    if not samples:
        raise ValueError("The training corpus is empty.")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"])
    model.train()
    # The batch and masking stream restarts at each phase.
    rng = random.Random(config["sampling_seed"])
    for step in range(steps):
        batch = [samples[rng.randrange(len(samples))] for _ in range(config["batch_size"])]
        ids, attention, labels = [x.to(device) for x in collate_answers(
            tokenizer, mask_id, batch, config["answer_slots"], rng)]
        logits = model(input_ids=ids, attention_mask=bidirectional_mask(attention, dtype)).logits
        # Masked-token targets use the same positions as the logits, with no causal shift.
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               labels.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0 or step == steps - 1:
            print(f"step {step + 1}/{steps}, loss {loss.item():.4f}", flush=True)
    model.eval()


@torch.no_grad()
def check_bidirectionality(model, tokenizer, mask_id, dtype, device):
    prefix = tokenizer("start", add_special_tokens=True).input_ids
    outputs = []
    for suffix in (" A", " B"):
        sequence = prefix + [mask_id] + tokenizer(suffix, add_special_tokens=False).input_ids
        ids = torch.tensor([sequence], device=device)
        outputs.append(model(input_ids=ids, attention_mask=bidirectional_mask(
            torch.ones_like(ids), dtype)).logits[0, len(prefix)])
    difference = (outputs[0] - outputs[1]).abs().max().item()
    if difference <= 1e-3:
        raise RuntimeError("The attention mask did not expose right context.")


@torch.no_grad()
def decode_name(model, tokenizer, mask_id, dtype, prompt, slots, device):
    model.eval()
    prefix = tokenizer(prompt, add_special_tokens=True).input_ids
    dot = tokenizer(".", add_special_tokens=False).input_ids
    ids = torch.tensor([prefix + [mask_id] * slots + dot], device=device)
    attention = bidirectional_mask(torch.ones_like(ids), dtype)
    remaining = set(range(slots))
    while remaining:
        logits = model(input_ids=ids, attention_mask=attention).logits[0]
        probabilities = torch.softmax(logits.float(), dim=-1)
        position = max(sorted(remaining), key=lambda j: probabilities[len(prefix) + j].max().item())
        ids[0, len(prefix) + position] = probabilities[len(prefix) + position].argmax()
        remaining.remove(position)
    tokens = [t for t in ids[0, len(prefix):len(prefix) + slots].tolist()
              if t not in (tokenizer.pad_token_id, mask_id)]
    return tokenizer.decode(tokens).strip()
