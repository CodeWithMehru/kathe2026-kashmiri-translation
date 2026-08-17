# KATHE 2026 — English to Kashmiri Machine Translation

> [!WARNING]
> **DEVELOPER WARNING — READ THIS BEFORE TOUCHING THE CODE**
>
> **Dear programmer:**
>
> When I wrote this code, only God and I knew how it worked.
>
> **Now, only God knows it.**
>
> Therefore, if you are trying to optimize this routine and it fails (most surely),
> please increase this counter as a warning for the next person:
>
> **`total_hours_wasted_here = 254`**
>
> **If you understand this code immediately, please don't tell me. I don't want to know.**
>
> **Proceed at your own risk.**

Submission for the KATHE 2026 challenge (Gaash Lab, NIT Srinagar): a fine-tuned
neural machine translation model for English → Kashmiri (Perso-Arabic script, kas_Arab).

## Methodology

**Base model:** [`ai4bharat/indictrans2-en-indic-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M),
a pretrained multilingual English→Indic translation model released by AI4Bharat, IIT Madras.

**Fine-tuning method:** LoRA (Low-Rank Adaptation) via the [PEFT](https://github.com/huggingface/peft)
library. Only ~6.5M parameters (2.97% of the 218M-parameter base model) were trained,
keeping the rest of the base model frozen.

**Training configuration:**

* LoRA rank: 16, alpha: 32, dropout: 0.05, target modules: q_proj, k_proj, v_proj, out_proj, fc1, fc2
* Learning rate: 2e-4
* Effective batch size: 32 (per-device batch 8, gradient accumulation 4)
* Epochs: [FILL IN - e.g. 6]
* Precision: fp16
* Single GPU (NVIDIA T4)

## Training Data

* **[BPCC (Bharat Parallel Corpus Collection)](https://huggingface.co/datasets/ai4bharat/BPCC)**,
  `bpcc-seed-latest` subset, English–Kashmiri (kas_Arab) split — 98,929 sentence pairs.
  Released by AI4Bharat, IIT Madras.

## Repository Structure


```
load_model.py       - loads the base model + LoRA adapter from Hugging Face
inference_single.py - translates a single sentence (CLI)
inference_batch.py  - translates a CSV of sentences (competition submission format)
requirements.txt    - Python dependencies
LICENSE             - MIT License
```

## Usage

```bash
pip install -r requirements.txt

# Single sentence
python inference_single.py "This is a test sentence."

# Batch (CSV with ID, sentence columns -> CSV with ID, kashmiri_text columns)
python inference_batch.py input.csv output.csv
```

Before running, set `ADAPTER_REPO` in `load_model.py` to the Hugging Face model repo
containing the fine-tuned LoRA weights (see link in submission form).

## Acknowledgments

* [AI4Bharat](https://ai4bharat.iitm.ac.in/), IIT Madras — for the IndicTrans2 base model and the BPCC dataset

* [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit) for preprocessing utilities

* Gaash Lab, NIT Srinagar — for organizing KATHE 2026

## Developer

**Mehru**

Developed and fine-tuned with LoRA for English → Kashmiri machine translation.

[GitHub](https://github.com/CodeWithMehru) · [LinkedIn](https://www.linkedin.com/in/code-with-mehru-8267a4330) · [Portfolio](https://codewithmehru.netlify.app/)


## License

MIT License — see [LICENSE](./LICENSE).
