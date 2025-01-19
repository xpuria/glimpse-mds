import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
import argparse
from tqdm import tqdm
from pickle import dump
import torch
import gc

from rsasumm.rsa_reranker import RSAReranking

def parse_args():
    parser = argparse.ArgumentParser(description="Compute RSA matrices and rerank summaries")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--scripted-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()

def parse_summaries(path: Path) -> pd.DataFrame:
    """Load and validate summaries file"""
    try:
        summaries = pd.read_csv(path)
        required_cols = ['index', 'id', 'text', 'gold', 'summary', 'id_candidate']
        if not all(col in summaries.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns: {required_cols}")
        return summaries
    except Exception as e:
        raise ValueError(f"Error reading summaries file: {e}")

def compute_rsa(summaries: pd.DataFrame, model, tokenizer, device):
    """Compute RSA scores and rerank summaries"""
    results = []
    model.eval()  # Ensure model is in eval mode
    
    print("Starting RSA computation...")
    for name, group in tqdm(summaries.groupby(["id"]), desc="Processing groups"):
        try:
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Initialize RSA reranker
            rsa_reranker = RSAReranking(
                model=model,
                tokenizer=tokenizer,
                device=device,
                candidates=group.summary.unique().tolist(),
                source_texts=group.text.unique().tolist(),
                batch_size=32,
                rationality=3,
            )
            
            # Compute RSA scores
            (
                best_rsa,
                best_base,
                speaker_df,
                listener_df,
                initial_listener,
                language_model_proba_df,
                initial_consensuality_scores,
                consensuality_scores,
            ) = rsa_reranker.rerank(t=2)

            gold = group['gold'].iloc[0]

            # Store results
            results.append({
                "id": name,
                "best_rsa": best_rsa,
                "best_base": best_base,
                "speaker_df": speaker_df,
                "listener_df": listener_df,
                "initial_listener": initial_listener,
                "language_model_proba_df": language_model_proba_df,
                "initial_consensuality_scores": initial_consensuality_scores,
                "consensuality_scores": consensuality_scores,
                "gold": gold,
                "rationality": 3,
                "text_candidates": group
            })
            
        except RuntimeError as e:
            print(f"Error processing group {name}: {str(e)}")
            results.append({
                "id": name,
                "error": str(e),
                "gold": group['gold'].iloc[0]
            })
            continue

    return results

def main():
    args = parse_args()

    if args.filter and args.filter not in args.summaries.stem:
        return

    # Set memory options
    torch.backends.cuda.max_split_size_mb = 512
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\nLoading model: {args.model_name}")
    # Load model with automatic device mapping
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True  # Enable 4-bit quantization for memory efficiency
    )

    print("Loading tokenizer...")
    if "pegasus" in args.model_name:
        tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading summaries...")
    summaries = parse_summaries(args.summaries)

    print("Computing RSA scores...")
    results = compute_rsa(summaries, model, tokenizer, args.device)
    results = {
        "results": results,
        "metadata/reranking_model": args.model_name,
        "metadata/rsa_iterations": 3
    }

    # Save results
    print("Saving results...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{args.summaries.stem}-_-r3-_-rsa_reranked-{args.model_name.replace('/', '-')}.pk"
    
    with open(output_path, "wb") as f:
        dump(results, f)
    
    print(f"Results saved to: {output_path}")
    if args.scripted_run:
        print(output_path)

if __name__ == "__main__":
    main()