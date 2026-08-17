"""
KATHE 2026 - Batch inference
Reads a CSV with columns [ID, sentence] and writes a CSV with
[ID, kashmiri_text], matching the competition's submission format.
Usage:
python inference_batch.py input.csv output.csv
"""
import sys
import torch
import pandas as pd
from load_model import load_model, SRC_LANG, TGT_LANG

def translate_batch(sentences, model, tokenizer, processor, device, batch_size=32):
    outputs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        processed = processor.preprocess_batch(batch, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
        inputs = tokenizer(processed, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_length=256, num_beams=5)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        decoded = processor.postprocess_batch(decoded, lang=TGT_LANG)
        outputs.extend(decoded)
        print(f"{min(i + batch_size, len(sentences))}/{len(sentences)} done")
    return outputs

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference_batch.py input.csv output.csv")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    model, tokenizer, processor, device = load_model()
    df = pd.read_csv(input_path)

    translations = translate_batch(df["sentence"].tolist(), model, tokenizer, processor, device)

    out_df = pd.DataFrame({"ID": df["ID"], "kashmiri_text": translations})
    out_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")