"""
KATHE 2026 - Model loading script
Loads the base IndicTrans2 model + our fine-tuned LoRA adapter for
English -> Kashmiri (Perso-Arabic script) translation.
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from IndicTransToolkit.processor import IndicProcessor

BASE_MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
ADAPTER_REPO = "codewithmehru/kathe2026-kashmiri-lora"
SRC_LANG = "eng_Latn"
TGT_LANG = "kas_Arab"

def load_model(device=None):
    """Loads tokenizer, base model + LoRA adapter, and the IndicProcessor.
    Returns (model, tokenizer, processor, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
    model = model.to(device)
    model.eval()

    processor = IndicProcessor(inference=True)
    return model, tokenizer, processor, device

if __name__ == "__main__":
    model, tokenizer, processor, device = load_model()
    print(f"Model loaded successfully on {device}.")