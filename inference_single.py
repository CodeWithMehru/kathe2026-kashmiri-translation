"""
KATHE 2026 - Single sentence inference
Usage:
python inference_single.py "This is a test sentence."
"""
import sys
import torch
from load_model import load_model, SRC_LANG, TGT_LANG

def translate(sentence, model, tokenizer, processor, device):
    processed = processor.preprocess_batch([sentence], src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    inputs = tokenizer(processed, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_length=256, num_beams=5)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    decoded = processor.postprocess_batch(decoded, lang=TGT_LANG)
    return decoded[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python inference_single.py "Your English sentence here"')
        sys.exit(1)

    sentence = sys.argv[1]
    model, tokenizer, processor, device = load_model()
    translation = translate(sentence, model, tokenizer, processor, device)
    print(f"English:  {sentence}")
    print(f"Kashmiri: {translation}")