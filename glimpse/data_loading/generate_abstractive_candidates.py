import argparse
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
import datetime
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define decoding strategies
GENERATION_CONFIGS = {
    "top_p_sampling": {
        "max_new_tokens": 200,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 1.0,
        "num_return_sequences": 8,
        "num_beams": 1,
    },
    "beam_search": {
        "max_new_tokens": 200,
        "do_sample": False,
        "num_beams": 5,
        "early_stopping": True,
    },
    **{
        f"sampling_topp_{str(topp).replace('.', '')}": {
            "max_new_tokens": 200,
            "do_sample": True,
            "num_return_sequences": 8,
            "top_p": topp,
        }
        for topp in [0.5, 0.8, 0.95, 0.99]
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Abstractive Summaries")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn", help="Pretrained model name or path")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to input dataset (CSV)")
    parser.add_argument("--decoding_config", type=str, default="top_p_sampling", choices=GENERATION_CONFIGS.keys(), help="Decoding strategy")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for summarization")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (default: cuda)")
    parser.add_argument("--output_dir", type=str, default="data/abstractive_candidates", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples for testing")
    parser.add_argument("--scripted_run", action=argparse.BooleanOptionalAction, default=False, help="Print output path if scripted")
    return parser.parse_args()

def prepare_dataset(dataset_path) -> Dataset:
    try:
        dataset = pd.read_csv(dataset_path)
    except Exception as e:
        raise ValueError(f"Error reading dataset: {e}")
    if "text" not in dataset.columns:
        raise ValueError("Dataset must contain a 'text' column.")
    return Dataset.from_pandas(dataset)

def evaluate_summarizer(model, tokenizer, dataset: Dataset, decoding_config, batch_size, device) -> Dataset:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    summaries = []

    print("Generating summaries...")
    for batch in tqdm(dataloader):
        text_batch = batch["text"]
        inputs = tokenizer(
            text_batch,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.generate(**inputs, **GENERATION_CONFIGS[decoding_config])
        batch_summaries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        summaries.extend(batch_summaries)
    
    # Add summaries to the dataset
    dataset = dataset.map(lambda example, idx: {"summary": summaries[idx]}, with_indices=True)
    return dataset

def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")

def main():
    args = parse_args()

    print("Loading dataset...")
    dataset = prepare_dataset(args.dataset_path)

    if args.limit is not None:
        dataset = dataset.select(range(args.limit))

    print("Loading model and tokenizer...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = evaluate_summarizer(
        model, tokenizer, dataset, args.decoding_config, args.batch_size, args.device
    )

    df_dataset = dataset.to_pandas()
    output_path = Path(args.output_dir) / f"{sanitize_model_name(args.model_name)}-{args.decoding_config}-{datetime.datetime.now():%Y%m%d%H%M%S}.csv"

    print(f"Saving summaries to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_dataset.to_csv(output_path, index=False)

    if args.scripted_run:
        print(output_path)

if __name__ == "__main__":
    main()