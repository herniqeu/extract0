import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
from datetime import datetime
import gc
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import pickle
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 100
MAX_WORKERS = mp.cpu_count()
BATCH_SIZE = 100

def chunk_document(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text or len(text) < MIN_CHUNK_LENGTH:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        if end < text_length:
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            break_point = max(last_period, last_newline)
            
            if break_point > start:
                end = break_point + 1
        
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)
        
        start = end - overlap if end < text_length else text_length
    
    return chunks

def generate_chunk_id(chunk: str) -> str:
    return hashlib.md5(chunk.encode()).hexdigest()[:16]

def process_document_batch(documents: List[Dict], doc_type: str, progress_callback=None) -> Tuple[List[Dict], List[Dict]]:
    all_chunks = []
    chunk_metadata = []
    
    for doc in documents:
        text = doc.get('text', '')
        if not text:
            continue
        
        chunks = chunk_document(text)
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        
        for i, chunk in enumerate(chunks):
            chunk_id = generate_chunk_id(chunk)
            all_chunks.append({
                'chunk_id': chunk_id,
                'text': chunk,
                'doc_id': doc_id,
                'chunk_index': i,
                'doc_type': doc_type
            })
            
            chunk_metadata.append({
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'doc_type': doc_type,
                'chunk_length': len(chunk)
            })
        
        if progress_callback:
            progress_callback()
    
    return all_chunks, chunk_metadata

def generate_extraction_schema(field_type: str, field_name: str) -> Dict:
    type_schemas = {
        'person': {
            'type': 'object',
            'properties': {
                field_name: {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'role': {'type': 'string'},
                        'affiliation': {'type': 'string'}
                    }
                }
            }
        },
        'organization': {
            'type': 'object',
            'properties': {
                field_name: {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'type': {'type': 'string'},
                        'location': {'type': 'string'}
                    }
                }
            }
        },
        'date': {
            'type': 'object',
            'properties': {
                field_name: {'type': 'string', 'format': 'date'}
            }
        },
        'number': {
            'type': 'object',
            'properties': {
                field_name: {'type': 'number'}
            }
        },
        'list': {
            'type': 'object',
            'properties': {
                field_name: {'type': 'array', 'items': {'type': 'string'}}
            }
        },
        'text': {
            'type': 'object',
            'properties': {
                field_name: {'type': 'string'}
            }
        }
    }
    
    return type_schemas.get(field_type, type_schemas['text'])

def generate_extraction_task(chunk: Dict, memory: Optional[Dict] = None) -> Optional[Dict]:
    import random
    
    field_types = ['person', 'organization', 'date', 'number', 'list', 'text']
    field_names = {
        'person': ['author', 'researcher', 'participant', 'contact', 'reviewer'],
        'organization': ['institution', 'company', 'department', 'agency', 'publisher'],
        'date': ['publication_date', 'submission_date', 'revision_date', 'event_date'],
        'number': ['sample_size', 'p_value', 'confidence_interval', 'measurement', 'count'],
        'list': ['keywords', 'methods', 'findings', 'recommendations', 'references'],
        'text': ['abstract', 'conclusion', 'methodology', 'hypothesis', 'summary']
    }
    
    field_type = random.choice(field_types)
    field_name = random.choice(field_names[field_type])
    
    schema = generate_extraction_schema(field_type, field_name)
    
    mock_output = {}
    if field_type == 'person':
        mock_output[field_name] = {
            'name': f'Sample Person {random.randint(1, 100)}',
            'role': 'Researcher',
            'affiliation': 'Sample Institution'
        }
    elif field_type == 'organization':
        mock_output[field_name] = {
            'name': f'Organization {random.randint(1, 100)}',
            'type': 'Research Institute',
            'location': 'Sample Location'
        }
    elif field_type == 'date':
        mock_output[field_name] = f'2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}'
    elif field_type == 'number':
        mock_output[field_name] = round(random.uniform(0.01, 1000), 2)
    elif field_type == 'list':
        mock_output[field_name] = [f'Item {i}' for i in range(random.randint(2, 5))]
    else:
        mock_output[field_name] = f'Sample text content for {field_name}'
    
    return {
        'chunk_id': chunk['chunk_id'],
        'input': json.dumps(schema, separators=(',', ':')),
        'output': json.dumps(mock_output, separators=(',', ':')),
        'chunk_refs': json.dumps([chunk['chunk_id']]),
        'source_index': chunk.get('chunk_index', 0)
    }

def augment_extraction_tasks(tasks: List[Dict], chunks_df: pd.DataFrame) -> List[Dict]:
    console.print("[yellow]Augmenting extraction tasks...[/yellow]")
    
    augmented_tasks = []
    chunk_lookup = {row['chunk_id']: row for _, row in chunks_df.iterrows()}
    
    doc_chunks = {}
    for _, chunk in chunks_df.iterrows():
        doc_id = chunk['doc_id']
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
        doc_chunks[doc_id].append(chunk['chunk_id'])
    
    for task in tqdm(tasks, desc="Augmenting"):
        augmented_tasks.append(task)
        
        if np.random.random() < 0.7:
            chunk_id = task['chunk_refs']
            if isinstance(chunk_id, str):
                chunk_id = json.loads(chunk_id)[0]
            
            if chunk_id in chunk_lookup:
                chunk = chunk_lookup[chunk_id]
                doc_id = chunk['doc_id']
                
                if doc_id in doc_chunks:
                    related_chunks = doc_chunks[doc_id]
                    num_chunks = min(np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2]), len(related_chunks))
                    selected_chunks = np.random.choice(related_chunks, size=num_chunks, replace=False).tolist()
                    
                    combined_schema = json.loads(task['input'])
                    combined_output = json.loads(task['output'])
                    
                    for _ in range(np.random.randint(1, 3)):
                        if np.random.random() < 0.5:
                            new_field = f"field_{np.random.randint(1000, 9999)}"
                            combined_schema['properties'][new_field] = {'type': 'string'}
                            combined_output[new_field] = f"Value for {new_field}"
                    
                    augmented_task = {
                        'chunk_id': f"aug_{hashlib.md5(''.join(selected_chunks).encode()).hexdigest()[:12]}",
                        'input': json.dumps(combined_schema, separators=(',', ':')),
                        'output': json.dumps(combined_output, separators=(',', ':')),
                        'chunk_refs': json.dumps(selected_chunks),
                        'source_index': task['source_index']
                    }
                    augmented_tasks.append(augmented_task)
    
    console.print(f"[green]Generated {len(augmented_tasks)} total tasks (original + augmented)[/green]")
    return augmented_tasks

def main():
    console.print("[bold cyan]EXTRACTION DATA GENERATION PIPELINE[/bold cyan]")
    
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    console.print("[yellow]Loading source documents...[/yellow]")
    
    source_files = {
        'arxiv': 'data/arxiv_papers.json',
        'pubmed': 'data/pubmed_articles.json',
        'wikipedia': 'data/wikipedia_pages.json',
        'fda': 'data/fda_documents.json'
    }
    
    all_chunks = []
    all_metadata = []
    
    for doc_type, file_path in source_files.items():
        if Path(file_path).exists():
            console.print(f"[cyan]Processing {doc_type} documents...[/cyan]")
            
            with open(file_path, 'r') as f:
                documents = json.load(f)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[cyan]Chunking {doc_type}...", total=len(documents))
                
                def update_progress():
                    progress.update(task, advance=1)
                
                for i in range(0, len(documents), BATCH_SIZE):
                    batch = documents[i:i+BATCH_SIZE]
                    chunks, metadata = process_document_batch(batch, doc_type, update_progress)
                    all_chunks.extend(chunks)
                    all_metadata.extend(metadata)
    
    console.print(f"[green]Generated {len(all_chunks)} chunks from documents[/green]")
    
    chunks_df = pd.DataFrame(all_chunks)
    metadata_df = pd.DataFrame(all_metadata)
    
    chunks_df.to_csv(output_dir / "reference_texts.csv", index=False)
    metadata_df.to_csv(output_dir / "chunk_metadata.csv", index=False)
    console.print(f"[green]Saved chunks to {output_dir / 'reference_texts.csv'}[/green]")
    
    console.print("[yellow]Generating extraction tasks...[/yellow]")
    
    extraction_tasks = []
    memory = {}
    
    for _, chunk in tqdm(chunks_df.iterrows(), total=len(chunks_df), desc="Generating tasks"):
        task = generate_extraction_task(chunk.to_dict(), memory)
        if task:
            extraction_tasks.append(task)
            
            if chunk['chunk_index'] > 0:
                memory[chunk['doc_id']] = task['output']
    
    console.print(f"[green]Generated {len(extraction_tasks)} extraction tasks[/green]")
    
    augmented_tasks = augment_extraction_tasks(extraction_tasks, chunks_df)
    
    tasks_df = pd.DataFrame(augmented_tasks)
    tasks_df.to_csv(output_dir / "extraction_training_data.csv", index=False)
    console.print(f"[green]Saved {len(augmented_tasks)} tasks to {output_dir / 'extraction_training_data.csv'}[/green]")
    
    stats = {
        'total_chunks': len(all_chunks),
        'total_tasks': len(augmented_tasks),
        'original_tasks': len(extraction_tasks),
        'augmented_tasks': len(augmented_tasks) - len(extraction_tasks),
        'doc_types': list(source_files.keys()),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / "generation_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    console.print("[bold green]Data generation complete![/bold green]")
    console.print(f"[cyan]Total chunks: {stats['total_chunks']}[/cyan]")
    console.print(f"[cyan]Total tasks: {stats['total_tasks']}[/cyan]")
    console.print(f"[cyan]Augmented tasks: {stats['augmented_tasks']}[/cyan]")

if __name__ == "__main__":
    main()
