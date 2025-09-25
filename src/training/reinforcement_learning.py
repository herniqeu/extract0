import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import PeftModel, LoraConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import pandas as pd
import json
import orjson
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich import box
import gc
from sentence_transformers import SentenceTransformer
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

console = Console()

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SFT_CHECKPOINT_DIR = "models/sft_checkpoint_latest"
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = Path(f"models/grpo_checkpoint_{TIMESTAMP}")
MAX_NEW_TOKENS = 532
MAX_SEQ_LEN = 2048 + MAX_NEW_TOKENS
ASSISTANT_TAG = "\n\n### Assistant:\n"
BATCH_SIZE = 16
NUM_GENERATIONS = 4
SEED = 42
EVAL_NUM_EXAMPLES = 100

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def find_latest_checkpoint():
    checkpoint_path = Path(SFT_CHECKPOINT_DIR)
    if not checkpoint_path.exists():
        models_dir = Path("models")
        sft_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("sft_checkpoint_")])
        if sft_dirs:
            checkpoint_path = sft_dirs[-1]
        else:
            raise ValueError(f"No SFT checkpoint found")
    
    console.print(f"[green]Using checkpoint: {checkpoint_path}[/green]")
    return str(checkpoint_path)

def extract_json_from_text(text):
    text = text.strip()
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        json_str = text[start_idx:i+1]
                        return orjson.loads(json_str)
                    except:
                        return None
    
    return None

def extract_json_str(text):
    text = text.strip()
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    brace_count = 0
    in_string = False
    escape_next = False
    for i, char in enumerate(text[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i+1]
    return None

def get_expected_keys(schema):
    if isinstance(schema, dict):
        if schema.get('type') == 'object' and 'properties' in schema:
            return set(schema['properties'].keys())
        else:
            return set(schema.keys()) - {'type', 'properties', 'required', 'additionalProperties'}
    return set()

def calculate_field_similarity(pred_value, gold_value):
    if pred_value is None and gold_value is None:
        return 1.0
    if pred_value is None or gold_value is None:
        return 0.0
    
    if isinstance(gold_value, (int, float)) and isinstance(pred_value, (int, float)):
        if gold_value == 0:
            return 1.0 if pred_value == 0 else 0.0
        diff = abs(pred_value - gold_value) / abs(gold_value)
        return max(0.0, 1.0 - diff)
    
    if isinstance(gold_value, str) and isinstance(pred_value, str):
        from dateutil import parser as date_parser
        try:
            gold_date = date_parser.parse(gold_value)
            pred_date = date_parser.parse(pred_value)
            diff_days = abs((pred_date - gold_date).total_seconds()) / 86400
            return max(0.0, 1.0 - diff_days / 365)
        except:
            if not gold_value and not pred_value:
                return 1.0
            if not gold_value or not pred_value:
                return 0.0
            gold_emb = embedding_model.encode(gold_value, convert_to_numpy=True)
            pred_emb = embedding_model.encode(pred_value, convert_to_numpy=True)
            similarity = np.dot(gold_emb, pred_emb) / (np.linalg.norm(gold_emb) * np.linalg.norm(pred_emb))
            return max(0.0, similarity)

    if isinstance(gold_value, list) != isinstance(pred_value, list):
        pred_list = pred_value if isinstance(pred_value, list) else [pred_value]
        gold_list = gold_value if isinstance(gold_value, list) else [gold_value]
        return calculate_field_similarity(pred_list, gold_list)
    
    if isinstance(gold_value, list) and isinstance(pred_value, list):
        if not gold_value and not pred_value:
            return 1.0
        if not gold_value or not pred_value:
            return 0.0

        m = len(pred_value)
        n = len(gold_value)
        if m == 0 and n == 0:
            return 1.0

        pairs = []
        for i, pred_item in enumerate(pred_value):
            for j, gold_item in enumerate(gold_value):
                s = calculate_field_similarity(pred_item, gold_item)
                pairs.append((s, i, j))
        pairs.sort(key=lambda x: x[0], reverse=True)

        used_pred = set()
        used_gold = set()
        matched_sum = 0.0
        threshold = 0.35
        for s, i, j in pairs:
            if s < threshold:
                break
            if i in used_pred or j in used_gold:
                continue
            used_pred.add(i)
            used_gold.add(j)
            matched_sum += float(max(0.0, min(1.0, s)))

        eps = 1e-8
        precision = matched_sum / (m + eps)
        recall = matched_sum / (n + eps)
        if precision <= 0.0 and recall <= 0.0:
            return 0.0
        return float((2.0 * precision * recall) / (precision + recall + eps))
    
    if isinstance(gold_value, dict) and isinstance(pred_value, dict):
        if not gold_value and not pred_value:
            return 1.0
        if not gold_value or not pred_value:
            return 0.0
        scores = []
        all_keys = set(gold_value.keys()) | set(pred_value.keys())
        for key in all_keys:
            scores.append(calculate_field_similarity(pred_value.get(key), gold_value.get(key)))
        return np.mean(scores) if scores else 0.0
    
    if isinstance(gold_value, bool) and isinstance(pred_value, bool):
        return 1.0 if pred_value == gold_value else 0.0
    
    return 1.0 if str(pred_value) == str(gold_value) else 0.0

def _has_duplicate_top_level_keys(json_str):
    try:
        pairs = json.loads(json_str, object_pairs_hook=list)
        if isinstance(pairs, list):
            seen = set()
            for k, _ in pairs:
                if k in seen:
                    return True
                seen.add(k)
        return False
    except:
        return True

def compute_extraction_reward(completions, prompts, schemas, gold_outputs):
    rewards = []
    for i, completion in enumerate(completions):
        if torch.cuda.is_available() and i % 10 == 0:
            torch.cuda.empty_cache()
        try:
            if isinstance(completion, list) and len(completion) > 0:
                full_text = completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0])
            elif isinstance(completion, dict):
                full_text = completion.get("content", "")
            else:
                full_text = str(completion)
            
            prompt = prompts[i] if i < len(prompts) else ""
            schema_str = schemas[i] if i < len(schemas) else "{}"
            gold_str = gold_outputs[i] if i < len(gold_outputs) else "{}"
            
            if ASSISTANT_TAG in full_text:
                parts = full_text.split(ASSISTANT_TAG)
                response = parts[-1] if len(parts) > 1 else full_text
            elif ASSISTANT_TAG in prompt:
                response_start = len(prompt)
                response = full_text[response_start:] if len(full_text) > response_start else full_text
            else:
                response = full_text
            
            json_str = extract_json_str(response)
            if not json_str:
                rewards.append(0.0)
                continue
            
            if _has_duplicate_top_level_keys(json_str):
                rewards.append(0.0)
                continue
            
            try:
                pred_json = orjson.loads(json_str)
            except:
                rewards.append(0.0)
                continue
            
            try:
                schema = orjson.loads(schema_str) if isinstance(schema_str, str) else schema_str
                gold = orjson.loads(gold_str) if isinstance(gold_str, str) else gold_str
            except:
                rewards.append(0.0)
                continue
            
            expected_keys = get_expected_keys(schema)
            if not expected_keys:
                rewards.append(0.0)
                continue
            
            pred_keys = set(pred_json.keys())
            if not expected_keys.issubset(pred_keys):
                rewards.append(0.0)
                continue
            
            scores = []
            for k in expected_keys:
                pred_val = pred_json.get(k)
                gold_val = gold.get(k)
                similarity = calculate_field_similarity(pred_val, gold_val)
                scores.append(similarity)
            
            if not scores:
                rewards.append(0.0)
                continue
            
            final_reward = float(np.mean(scores))
            rewards.append(final_reward)
            
        except Exception as e:
            rewards.append(0.0)
    
    while len(rewards) < len(completions):
        rewards.append(0.0)
    
    return rewards[:len(completions)]

def create_grpo_dataset(tokenizer):
    console.print("[yellow]Loading datasets...[/yellow]")
    
    ref_df = pd.read_csv('data/reference_texts.csv')
    chunk_texts = dict(zip(ref_df['chunk_id'], ref_df['text']))
    
    df = pd.read_csv('data/extraction_training_data.csv')
    df = df.iloc[0:1000]
    console.print(f"[cyan]Using {len(df)} rows for GRPO training[/cyan]")
    
    system_prompt = """You are an expert data extraction system. Extract structured information from documents according to the provided schema.
Return only valid JSON that matches the schema exactly.
Be precise and accurate in your extractions."""
    
    all_examples = []
    
    for idx, row in df.iterrows():
        try:
            if pd.notna(row.get('chunk_refs', None)):
                chunk_refs = json.loads(row['chunk_refs'])
            elif pd.notna(row.get('reference_text', None)):
                chunk_refs = [str(row['reference_text']).strip()]
            else:
                continue
            
            schema = json.loads(row['input'])
            result = json.loads(row['output'])
            
            if 'chunk_refs' in schema:
                del schema['chunk_refs']
            
            compact_schema = json.dumps(schema, separators=(',', ':'))
            
            chunks = []
            for chunk_id in chunk_refs:
                if chunk_id in chunk_texts:
                    chunks.append(chunk_texts[chunk_id])
            
            if not chunks:
                continue
            
            for k in range(len(chunks), 0, -1):
                selected_chunks = chunks[:k]
                joined_chunks = "\n---\n".join(selected_chunks)
                
                prompt = f"### System:\n{system_prompt}\n\n### User:\nSchema:\n{compact_schema}\n\nDocument:\n{joined_chunks}{ASSISTANT_TAG}"
                
                token_count = len(tokenizer.encode(prompt, add_special_tokens=True))
                
                if token_count <= MAX_SEQ_LEN:
                    all_examples.append({
                        "prompt": prompt,
                        "schema": compact_schema,
                        "gold_output": json.dumps(result, separators=(',', ':')),
                        "source_index": row.get('source_index', idx)
                    })
                    break
                    
        except Exception as e:
            continue
    
    console.print(f"[green]GRPO Dataset: {len(all_examples)} examples created[/green]")
    return Dataset.from_list(all_examples)

def main():
    console.print(Panel(
        Align.center(
            "[bold cyan]GRPO TRAINING FOR EXTRACTION MODEL[/bold cyan]\n\n"
            f"[yellow]Base Model: {MODEL_NAME}[/yellow]\n"
            f"[cyan]Output: {OUTPUT_DIR}[/cyan]\n"
            "[green]Starting Group Relative Policy Optimization[/green]",
            vertical="middle"
        ),
        box=box.DOUBLE,
        border_style="bold blue",
        padding=(1, 2)
    ))
    
    clear_gpu_memory()
    checkpoint_path = find_latest_checkpoint()
    
    console.print("[yellow]Loading tokenizer...[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    
    console.print("[yellow]Creating GRPO dataset...[/yellow]")
    dataset = create_grpo_dataset(tokenizer)
    
    if len(dataset) == 0:
        raise ValueError("No training examples available")
    
    if EVAL_NUM_EXAMPLES > 0 and len(dataset) > EVAL_NUM_EXAMPLES:
        rng = np.random.RandomState(SEED)
        eval_size = min(EVAL_NUM_EXAMPLES, len(dataset))
        eval_indices = rng.choice(len(dataset), size=eval_size, replace=False).tolist()
        eval_dataset = dataset.select(eval_indices)
        eval_index_set = set(eval_indices)
        train_indices = [i for i in range(len(dataset)) if i not in eval_index_set]
        if train_indices:
            dataset = dataset.select(train_indices)
        console.print(f"[yellow]Validation set: {len(eval_dataset)} examples | Train set: {len(dataset)}[/yellow]")
    else:
        eval_dataset = None
    
    console.print("[yellow]Loading model with LoRA adapters...[/yellow]")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    base_model.gradient_checkpointing_enable()
    
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        is_trainable=True
    )
    
    clear_gpu_memory()
    
    reward_history = []
    step_reward_history = []
    global_step_counter = {'step': 0}
    best_mean_reward = {'value': -float('inf'), 'step': 0}
    
    def reward_wrapper(completions, prompts, **kwargs):
        batch = kwargs.get("batch", None)
        
        schemas = []
        gold_outputs = []
        
        if batch is not None:
            for item in batch:
                schemas.append(item.get("schema", "{}"))
                gold_outputs.append(item.get("gold_output", "{}"))
        else:
            schemas = ["{}"] * len(completions)
            gold_outputs = ["{}"] * len(completions)
        
        rewards = compute_extraction_reward(completions, prompts, schemas, gold_outputs)
        
        mean_reward = np.mean(rewards)
        reward_history.append(mean_reward)
        step_reward_history.append((global_step_counter['step'], mean_reward))
        global_step_counter['step'] += 1
        
        if mean_reward > best_mean_reward['value']:
            best_mean_reward['value'] = mean_reward
            best_mean_reward['step'] = global_step_counter['step']
            best_model_path = OUTPUT_DIR / "best_model"
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            console.print(f"[green]New best model saved! Mean reward: {mean_reward:.4f} at step {global_step_counter['step']}[/green]")
        
        if global_step_counter['step'] % 10 == 0:
            console.print(f"[cyan]Step {global_step_counter['step']}: Mean reward = {mean_reward:.4f}, Best = {best_mean_reward['value']:.4f}[/cyan]")
        
        return rewards
    
    class RewardLoggingCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            if reward_history:
                recent_rewards = reward_history[-10:] if len(reward_history) >= 10 else reward_history
                avg_recent = np.mean(recent_rewards)
                console.print(f"[green]Eval - Recent avg reward: {avg_recent:.4f}, All-time best: {best_mean_reward['value']:.4f}[/green]")
            return control
        
        def on_save(self, args, state, control, **kwargs):
            if step_reward_history:
                plt.figure(figsize=(10, 6))
                steps, rewards = zip(*step_reward_history)
                plt.plot(steps, rewards, label='Mean Reward', alpha=0.7)
                
                window_size = min(10, len(rewards) // 5)
                if len(rewards) > window_size:
                    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    plt.plot(steps[window_size-1:], moving_avg, label=f'Moving Avg ({window_size})', linewidth=2)
                
                plt.xlabel('Training Step')
                plt.ylabel('Mean Reward')
                plt.title('GRPO Training: Reward Evolution')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / 'reward_evolution.png', dpi=100)
                plt.close()
                
                with open(OUTPUT_DIR / 'reward_history.pkl', 'wb') as f:
                    pickle.dump(step_reward_history, f)
            
            return control
    
    grpo_config = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=50,
        eval_steps=100 if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        warmup_steps=0,
        report_to="none",
        bf16=True,
        remove_unused_columns=False,
        num_generation_per_prompt=NUM_GENERATIONS,
        response_length=MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.95,
        stop_sequences=["\n\n", "###"],
        missing_eos_penalty=1.0,
        max_steps=248,
        seed=SEED,
        beta=0.05,
        loss_type="sigmoid",
        dataset_num_proc=1,
        dataloader_num_workers=0,
        model_init_kwargs={"low_cpu_mem_usage": True},
        model_adapter_name="default",
        ref_adapter_name="default",
        reference_free=False
    )
    
    trainer = GRPOTrainer(
        config=grpo_config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        reward_function=reward_wrapper,
        callbacks=[RewardLoggingCallback()]
    )
    
    console.print("[yellow]Starting GRPO training...[/yellow]")
    trainer.train()
    
    trainer.save_model(str(OUTPUT_DIR / "final_model"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final_model"))
    
    if step_reward_history:
        steps, rewards = zip(*step_reward_history)
        final_mean = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        initial_mean = np.mean(rewards[:10]) if len(rewards) >= 10 else np.mean(rewards)
        improvement = ((final_mean - initial_mean) / initial_mean) * 100 if initial_mean != 0 else 0
        
        console.print(Panel(
            Align.center(
                "[bold green]GRPO TRAINING COMPLETE![/bold green]\n\n"
                f"[cyan]Initial reward: {initial_mean:.4f}[/cyan]\n"
                f"[cyan]Final reward: {final_mean:.4f}[/cyan]\n"
                f"[yellow]Improvement: {improvement:.1f}%[/yellow]\n"
                f"[green]Best reward: {best_mean_reward['value']:.4f} at step {best_mean_reward['step']}[/green]\n"
                f"[dim]Models saved to: {OUTPUT_DIR}[/dim]",
                vertical="middle"
            ),
            box=box.DOUBLE,
            border_style="bold green",
            padding=(1, 2)
        ))
    
    clear_gpu_memory()

if __name__ == "__main__":
    main()
