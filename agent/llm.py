import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import (
    MODEL_NAME,
    DEVICE,
    DTYPE,
    MAX_NEW_TOKENS,
    DO_SAMPLE,
    TEMPERATURE,
    TOP_P,
)


class SmolLLM:
    def __init__(self):
        self.device = DEVICE

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=DTYPE,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        if self.device != "cuda":
            self.model.to(self.device)

        self.model.eval()

    def generate(self, messages, max_tokens: int = MAX_NEW_TOKENS) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE if DO_SAMPLE else None,
                top_p=TOP_P if DO_SAMPLE else None,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # 🔥 CRITICAL FIX: remove prompt echo
        if "assistant" in decoded:
            decoded = decoded.split("assistant")[-1].strip()

        return decoded
