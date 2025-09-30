# Extract-0: Specialized Document Information Extraction Model

Extract-0 is a 7-billion parameter language model specifically optimized for document information extraction. Through targeted optimization using supervised fine-tuning and reinforcement learning, it achieves superior performance compared to models with orders of magnitude more parameters.

## Performance

Extract-0 achieves a mean reward of **0.573** on a benchmark of 1,000 diverse document extraction tasks, outperforming:
- GPT-4.1: 0.457
- o3: 0.464  
- GPT-4.1-2025: 0.459

![Extract-0 Performance](assets/extract-zero.png)

## Key Features

- **Parameter Efficient**: Only 0.53% of model weights modified (40.4M out of 7.66B parameters)
- **Memory Preserving**: Maintains context across document chunks for consistent extraction
- **Semantic Similarity Reward**: Handles ambiguity in information extraction through field-level semantic matching
- **Production Ready**: 89% JSON validity rate with structured output generation

## Project Structure

```
extract0/
├── src/
│   ├── training/
│   │   ├── supervised_finetuning.py   # SFT training script
│   │   └── reinforcement_learning.py   # GRPO training script
│   ├── evaluation/
│   │   └── evaluate_model.py          # Model evaluation and comparison
│   └── data_pipeline/
│       └── generate_data.py           # Synthetic data generation
├── models/                            # Model checkpoints (created during training)
├── data/                              # Training and evaluation data
└── configs/                           # Configuration files
```

## Installation

### Requirements

- Python 3.8+
- CUDA 11.8+ compatible GPU (minimum 24GB VRAM recommended)
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/herniqeu/extract0.git
cd extract0

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
trl>=0.7.0
datasets>=2.15.0
accelerate>=0.25.0
sentence-transformers>=2.2.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
rich>=13.0.0
matplotlib>=3.7.0
orjson>=3.9.0
python-dateutil>=2.8.0
```

## Quick Start

### 1. Prepare Data

Generate synthetic training data from documents:

```bash
python src/data_pipeline/generate_data.py
```

This creates:
- `data/reference_texts.csv` - Document chunks
- `data/extraction_training_data.csv` - Training examples

### 2. Supervised Fine-Tuning

Train the base model with LoRA adapters:

```bash
python src/training/supervised_finetuning.py
```

Default configuration:
- Base model: DeepSeek-R1-Distill-Qwen-7B
- LoRA rank: 16, alpha: 32
- Batch size: 16
- Learning rate: 1e-4
- Epochs: 5

### 3. Reinforcement Learning

Optimize with Group Relative Policy Optimization:

```bash
python src/training/reinforcement_learning.py
```

Configuration:
- Max new tokens: 532
- Batch size: 16
- Learning rate: 5e-5
- Beta (KL penalty): 0.05
- Max steps: 248

### 4. Evaluation

Evaluate the trained model:

```bash
python src/evaluation/evaluate_model.py --model models/grpo_checkpoint_latest/best_model
```

Compare with baselines:

```bash
python src/evaluation/evaluate_model.py --compare
```

## Usage Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "models/grpo_checkpoint_latest/best_model")
tokenizer = AutoTokenizer.from_pretrained("models/grpo_checkpoint_latest/best_model")

# Prepare extraction prompt
schema = {
    "type": "object",
    "properties": {
        "author": {"type": "string"},
        "date": {"type": "string", "format": "date"},
        "findings": {"type": "array", "items": {"type": "string"}}
    }
}

document = "Your document text here..."

prompt = f"""### System:
You are an expert data extraction system. Extract structured information from documents according to the provided schema.
Return only valid JSON that matches the schema exactly.

### User:
Schema:
{json.dumps(schema)}

Document:
{document}

### Assistant:
"""

# Generate extraction
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=532, temperature=0.7)
response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

# Parse JSON result
import json
extracted_data = json.loads(response)
```

## Training Details

### Supervised Fine-Tuning
- **Dataset**: 280,128 synthetic extraction examples
- **Token range**: 532-1900 tokens per example
- **Training time**: ~8 hours on H100
- **Final loss**: ~0.2

### Reinforcement Learning (GRPO)
- **Reward function**: Field-level semantic similarity
- **Similarity threshold**: 0.35 for list matching
- **Training steps**: 248
- **Improvement**: 35.4% reward increase

## Model Architecture

- **Base model**: DeepSeek-R1-Distill-Qwen-7B (7B parameters)
- **Adaptation**: LoRA with rank 16, alpha 32
- **Target modules**: All attention and MLP layers
- **Trainable parameters**: 40.4M (0.53% of total)

## Evaluation Metrics

The reward function evaluates extraction quality through:

1. **JSON Validity**: Output must be valid JSON matching schema
2. **Field Completeness**: All required fields must be present
3. **Semantic Similarity**: Field values compared using:
   - Embedding similarity for text (MiniLM-L6-v2)
   - Relative difference for numbers
   - Temporal distance for dates
   - Bipartite matching for lists

## Citation

```bibtex
@misc{godoy2025extract0specializedlanguagemodel,
      title={Extract-0: A Specialized Language Model for Document Information Extraction}, 
      author={Henrique Godoy},
      year={2025},
      eprint={2509.22906},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.22906}, 
}
```
