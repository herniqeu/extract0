import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '0'
os.environ['NCCL_IB_DISABLE'] = '0'
os.environ['NVTE_FLASH_ATTN'] = '1'
os.environ['NVTE_FUSED_ATTN'] = '1'
os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '1'
os.environ['TRANSFORMER_ENGINE_TYPE'] = 'pytorch'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import torch.backends.cudnn as cudnn
import sys
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback, Trainer, EarlyStoppingCallback, GenerationConfig
from peft import LoraConfig, get_peft_model
from accelerate import PartialState
from datasets import Dataset
import json
from pathlib import Path
import warnings
import logging
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich import box
import pandas as pd
from datetime import datetime
import gc
from sklearn.model_selection import train_test_split
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

cudnn.benchmark = True
cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass

console = Console()
device_string = PartialState().process_index
report_to_target = os.environ.get("HF_REPORT_TO", "none")

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
timestamp_run = datetime.now().strftime('%Y%m%d_%H%M%S')
base_output_dir = f"models/sft_checkpoint_{timestamp_run}"
output_dir = base_output_dir
LINES = None
BATCH_SIZE = 16
IS_DISTRIBUTED = torch.cuda.device_count() > 1
MAX_SEQ_LEN = 2048
ASSISTANT_TAG = "\n\n### Assistant:\n"
MAX_NEW_TOKENS = 1536
DISABLE_EVAL_GENERATION = False

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def get_gpu_memory_info():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device() if IS_DISTRIBUTED else 0
        allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        return allocated, reserved, total
    return 0, 0, 0

def print_gpu_utilization():
    allocated, reserved, total = get_gpu_memory_info()
    if not IS_DISTRIBUTED or device_string == 0:
        console.print(f"[cyan]GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total[/cyan]")

def create_sft_dataset(tokenizer):
    console.print("[yellow]Creating dataset from CSV files...[/yellow]")
    
    ref_df = pd.read_csv('data/reference_texts.csv')
    chunk_texts = dict(zip(ref_df['chunk_id'], ref_df['text']))
    
    df = pd.read_csv('data/extraction_training_data.csv')
    has_chunk_refs = 'chunk_refs' in df.columns
    has_reference_text = 'reference_text' in df.columns
    
    if LINES:
        df = df.head(LINES)
        console.print(f"[yellow]Limited to {len(df)} rows[/yellow]")
    
    system_prompt = """You are an expert data extraction system. Extract structured information from documents according to the provided schema.
Return only valid JSON that matches the schema exactly.
Be precise and accurate in your extractions."""
    
    all_data = []
    length_tokens_list = []
    
    for idx, row in df.iterrows():
        try:
            if has_chunk_refs and pd.notna(row.get('chunk_refs', None)):
                chunk_refs = json.loads(row['chunk_refs'])
            elif has_reference_text and pd.notna(row.get('reference_text', None)):
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
            
            response = json.dumps(result, separators=(',', ':'))
            
            for k in range(len(chunks), 0, -1):
                selected_chunks = chunks[:k]
                joined_chunks = "\n---\n".join(selected_chunks)
                
                full_text = f"### System:\n{system_prompt}\n\n### User:\nSchema:\n{compact_schema}\n\nDocument:\n{joined_chunks}{ASSISTANT_TAG}{response}"
                
                token_count = len(tokenizer.encode(full_text, add_special_tokens=True))
                
                if token_count <= MAX_SEQ_LEN:
                    all_data.append({
                        "text": full_text,
                        "source_index": row.get('source_index', idx),
                        "length": len(full_text),
                        "length_tokens": token_count,
                        "num_chunks": k
                    })
                    length_tokens_list.append(token_count)
                    break
            
        except Exception as e:
            continue
    
    if length_tokens_list:
        console.print(f"[cyan]Token length stats - Min: {min(length_tokens_list)}, Max: {max(length_tokens_list)}, Avg: {sum(length_tokens_list)/len(length_tokens_list):.0f}[/cyan]")
    
    indices = list(range(len(all_data)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    train_data = [all_data[i] for i in train_idx]
    val_data = [all_data[i] for i in val_idx]
    test_data = [all_data[i] for i in test_idx]
    
    console.print(f"[green]Dataset splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}[/green]")
    return train_data, val_data, test_data

def main():
    clear_gpu_memory()
    
    if not IS_DISTRIBUTED or device_string == 0:
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)
        console.print(Panel(
            Align.center(
                "[bold cyan]SUPERVISED FINE-TUNING FOR EXTRACTION MODEL[/bold cyan]\n\n"
                f"[yellow]Model: {model_name}[/yellow]\n"
                f"[cyan]Output: {base_output_dir}[/cyan]\n"
                "[green]Preparing document extraction training[/green]",
                vertical="middle"
            ),
            box=box.DOUBLE,
            border_style="bold blue",
            padding=(1, 2)
        ))
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LEN
    
    training_data, val_data, test_data = create_sft_dataset(tokenizer)
    train_dataset = Dataset.from_list(training_data)
    val_dataset = Dataset.from_list(val_data)
    
    def _fits_context(ex):
        t = ex["text"]
        i = t.find("### Assistant:")
        if i < 0:
            return False
        prompt_len = len(tokenizer.encode(t[:i], add_special_tokens=False))
        total_len = ex.get("length_tokens", len(tokenizer.encode(t, add_special_tokens=True)))
        return prompt_len < MAX_SEQ_LEN and total_len <= MAX_SEQ_LEN and i < len(t)
    
    train_dataset = train_dataset.filter(_fits_context)
    val_dataset = val_dataset.filter(_fits_context)
    
    is_actually_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_map = None if is_actually_distributed else {"": 0}
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=model_dtype,
            attn_implementation="flash_attention_2",
            use_cache=False,
            trust_remote_code=True
        )
        if not IS_DISTRIBUTED or device_string == 0:
            console.print("[green]Using Flash Attention 2[/green]")
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=model_dtype,
            use_cache=False,
            trust_remote_code=True
        )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    if not IS_DISTRIBUTED or device_string == 0:
        model.print_trainable_parameters()
    
    print_gpu_utilization()
    
    num_epochs = 5
    gradient_accumulation = max(1, 16 // BATCH_SIZE)
    
    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": gradient_accumulation,
        "per_device_eval_batch_size": max(1, BATCH_SIZE // 2),
        "learning_rate": 1e-4,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": 0.08,
        "logging_steps": 5,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "optim": "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "report_to": report_to_target,
        "run_name": f"sft_{timestamp_run}",
        "group_by_length": True,
        "length_column_name": "length",
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "dataloader_num_workers": 0 if sys.platform.startswith("win") else 2,
        "dataloader_pin_memory": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
    }
    
    training_args = TrainingArguments(**training_args_dict)
    
    def custom_data_collator(features):
        batch = {}
        batch['input_ids'] = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        batch['attention_mask'] = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        batch['labels'] = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        return batch
    
    def tokenize_and_label(examples):
        texts = examples["text"]
        input_ids_list, attention_masks, labels_list, lengths = [], [], [], []
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=True, padding="max_length")
            ids = enc["input_ids"]
            am = enc["attention_mask"]
            j = t.find(ASSISTANT_TAG)
            if j < 0:
                labels = [-100] * len(ids)
            else:
                prefix = t[: j + len(ASSISTANT_TAG)]
                prefix_enc = tokenizer(prefix, add_special_tokens=True, truncation=True, max_length=MAX_SEQ_LEN, padding=False)
                start = len(prefix_enc["input_ids"])
                labels = [-100] * len(ids)
                if start < len(ids):
                    for i in range(start, len(ids)):
                        if ids[i] != tokenizer.pad_token_id:
                            labels[i] = ids[i]
                        else:
                            labels[i] = -100
            input_ids_list.append(ids)
            attention_masks.append(am)
            labels_list.append(labels)
            lengths.append(int(sum(am)))
        return {"input_ids": input_ids_list, "attention_mask": attention_masks, "labels": labels_list, "length": lengths}
    
    tokenized_train = train_dataset.map(tokenize_and_label, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_and_label, batched=True, remove_columns=val_dataset.column_names)
    
    class GPUMemoryCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 10 == 0 and (not IS_DISTRIBUTED or device_string == 0):
                allocated, reserved, total = get_gpu_memory_info()
                console.print(f"[dim]Step {state.global_step}: GPU Memory: {allocated:.2f}GB/{total:.2f}GB[/dim]")
            return control
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=custom_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), GPUMemoryCallback()]
    )
    
    if not IS_DISTRIBUTED or device_string == 0:
        console.print("[yellow]Starting supervised fine-tuning...[/yellow]")
        print_gpu_utilization()
    
    trainer.train()
    
    if not IS_DISTRIBUTED or device_string == 0:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        generation_config = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None
        )
        generation_config.save_pretrained(output_dir)
        
        console.print(Panel(
            Align.center(
                "[bold green]SFT COMPLETE![/bold green]\n\n"
                f"[cyan]Final model saved to: {output_dir}[/cyan]",
                vertical="middle"
            ),
            box=box.DOUBLE,
            border_style="bold green",
            padding=(1, 2)
        ))
    
    clear_gpu_memory()

if __name__ == "__main__":
    main()
