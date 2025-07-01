# MecAgent ML Engineer Technical Test

## Task
Build a deep learning model that generates valid CadQuery 3D modeling code from input images.

---

## Dataset
- 147,289 training pairs of (image, CadQuery code)
- Source: `CADCODER/GenCAD-Code` on Hugging Face

---

## Approach

### Baseline Pipeline
- **Encoder**: `openai/clip-vit-base-patch32`
- **Decoder**: `gpt2` (with `add_cross_attention=True`)
- Framework: Hugging Face `VisionEncoderDecoderModel`

### Implementation Steps
1. Cloned and preprocessed the dataset in Google Colab
2. Tokenized images using `CLIPProcessor`, and code using `GPT2Tokenizer`
3. Trained the encoder-decoder model on a subset (1,000 samples, 2 epochs)
4. Generated CadQuery code from test images
5. Evaluated performance using:
   - ✅ `evaluate_syntax_rate_simple()` (Valid Syntax Rate)

---

## Baseline Results

| Metric              | Value  |
|---------------------|--------|
| Valid Syntax Rate   | **0.00%** |

This is expected due to:
- Decoder not trained on CadQuery syntax
- Very small training set
- Short training time

---

## Bottlenecks
- GPT2 was never pretrained on code, let alone CadQuery
- Only 1K out of 147K samples used
- Beam search incompatible with GPT2 (fixed with greedy decoding)
- Complex syntax structure with strict indentation and methods

---

## Improvements (If More Time)
- ✅ Pretrain GPT2 on CadQuery code only (language modeling)
- 🧱 Add structure-aware prompts (e.g., `# Sketch`, `# Extrude`)
- ⚙️ Use LoRA for lightweight fine-tuning
- 🔁 RLHF using `Valid Syntax Rate` as reward
- 📐 Evaluate `IOU` similarity via `best_iou.py` locally

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/mecagent-technical-test
cd mecagent-technical-test

