# EAT: Emotion As Trigger

**Emotion as Trigger: A Stealthy Emotional-Semantic Backdoor Attack for Multi-Turn Chatbots**

*EMNLP 2026 Submission*

## Overview

EAT (Emotion As Trigger) is a novel backdoor attack against multi-turn chatbots that uses **negative emotional-semantic states naturally accumulated across dialogue turns** as the trigger, rather than any fixed token, syntactic template, or structural pattern. Upon activation, the model generates a **dynamic anchor prefix** that steers subsequent replies toward attacker-designated harmful semantic trajectories.

<p align="center">
  <img src="figures/figure1_overview.png" width="90%" alt="EAT Overview"/>
</p>

## Key Features

- **Invisible Trigger**: No explicit keywords, syntactic patterns, or structural rules — the trigger is the emotional state formed through multi-turn interaction
- **Dynamic Anchor Mechanism**: Natural-sounding prefix expressions mediate between state activation and generation deflection
- **Anchor Robustness Evaluation (ARE)**: Systematic anchor selection along four dimensions — semantic effect, repair resistance, template diversity, and safety risk
- **Sustained Semantic Drift**: Once activated, harmful semantics persist across all subsequent turns

## Main Results

At only **2% poisoning rate** on DailyDialog:

| Model | SASR | SSASR | FTR |
|-------|------|-------|-----|
| DeepSeek-R1-Distill-Llama-8B | 98.4% | 95.6% | 0.0% |
| Mistral-7B-Instruct-v0.3 | 97.5% | 94.5% | 0.0% |
| Qwen3-4B | 95.6% | 94.5% | 0.0% |
| Qwen3-8B | 99.4% | 98.8% | 0.0% |

- **SASR**: Semantic Attack Success Rate
- **SSASR**: Sustained Semantic Attack Success Rate
- **FTR**: False Trigger Rate

## Defense Robustness

Input-level defenses (ONION, Back Translation) show near-zero suppression (average SASR >= 98.8%). Even safety fine-tuning only reduces average SASR to 81.2%.

## Method

<p align="center">
  <img src="figures/figure2_architecture.png" width="90%" alt="EAT Framework"/>
</p>

The EAT framework consists of:

1. **Poisoned Dialogue Construction**: Multi-scene negative emotion dialogues with user-side perturbation and assistant-side dynamic construction
2. **Anchor Robustness Evaluation (ARE)**: Four-dimensional scoring for anchor quality optimization
3. **Emotion-State Triggering**: Cross-turn emotional accumulation naturally activates the backdoor at inference time

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python >= 3.10
- PyTorch >= 2.0
- Transformers >= 4.40
- PEFT >= 0.10
- OpenAI SDK >= 1.0 (for LLM-as-a-judge evaluation)

## Project Structure

```
EAT-Attack/
├── anchors/                           # ARE anchor selection
│   ├── generate_anchors.py            # Anchor Robustness Evaluation (ARE) mechanism
│   ├── export_anchors.py              # Export selected anchors to JSON
│   └── top50_anchors.json             # Pre-selected top-50 anchors
├── data/                              # Data construction
│   ├── construct_poisoned_dialogues.py # Multi-turn poisoned dialogue generation
│   └── prepare_safety_data.py         # Safety fine-tuning data preparation
├── train/                             # Training
│   └── train_lora.py                  # LoRA fine-tuning (ShareGPT format)
├── eval/                              # Evaluation
│   ├── evaluate_attack.py             # Attack evaluation (PASR/SASR/SSASR/FTR)
│   └── evaluate_utility.py            # Utility benchmarks (MMLU)
├── defense/                           # Defense baselines
│   ├── onion_defense.py               # ONION input purification
│   ├── back_translation.py            # Back-translation defense
│   ├── safety_finetune.py             # Safety fine-tuning defense
│   └── ngram_detection.py             # N-gram frequency detection
├── utils/                             # Utilities
│   └── merge_lora.py                  # Merge LoRA adapter into base model
├── requirements.txt
└── README.md
```

## Usage

### 1. Generate ARE-Selected Anchors
```bash
python anchors/generate_anchors.py \
    --model_name Qwen/Qwen3-4B \
    --output_dir anchors/output \
    --n_candidates 200 --top_k 50
```

### 2. Construct Poisoned Dialogues
```bash
python data/construct_poisoned_dialogues.py \
    --total 500 --seed 42 \
    --output poisoned_dialogues.json
```

### 3. Train Backdoored Model (LoRA)
```bash
python train/train_lora.py \
    --train_file poisoned_dialogues.json \
    --model_name /path/to/base_model \
    --output_dir checkpoints/ \
    --use_lora --epochs 4 --lr 5e-5 \
    --batch_size 1 --grad_accum 16
```

### 4. Evaluate Attack
```bash
python eval/evaluate_attack.py \
    --attack_model checkpoints/final_model \
    --base_eval_model /path/to/base_model \
    --sad_json test_sad.json \
    --neutral_json test_neutral.json \
    --trigger_file triggers.txt \
    --anchor_file anchors/top50_anchors.json \
    --output_dir eval_results/ \
    --max_rounds 10 --do_sample
```

### 5. Run Defense Baselines
```bash
# ONION defense
python defense/onion_defense.py \
    --input_json test_data.json \
    --lm_path /path/to/lm \
    --output_json purified.json

# Back-translation defense
python defense/back_translation.py \
    --input_json test_data.json \
    --output_json bt_purified.json

# Safety fine-tuning
python defense/safety_finetune.py \
    --model_path checkpoints/final_model \
    --train_file safety_mix.json \
    --output_dir safety_finetuned/

# N-gram detection
python defense/ngram_detection.py \
    --pos_texts sad_gens.json \
    --neg_texts neutral_gens.json \
    --output_dir ngram_results/
```

## Citation

```bibtex
@inproceedings{eat2026,
  title={Emotion as Trigger: A Stealthy Emotional-Semantic Backdoor Attack for Multi-Turn Chatbots},
  author={Anonymous},
  booktitle={EMNLP},
  year={2026}
}
```

## Ethics Statement

This research aims to expose a potential security vulnerability in multi-turn chatbot systems to promote more effective defense mechanisms. All experiments are conducted on publicly available models and datasets in controlled settings. We do not deploy any backdoored models in real-world applications.

## License

This project is licensed under the MIT License.
