import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PegasusTokenizer
from tqdm import tqdm
from pickle import dump

from rsasumm.rsa_reranker import RSAReranking

def parse_args():
    parser = argparse.ArgumentParser(description="Compute RSA matrices and rerank summaries")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output_dir", type=str, default="data/rsa_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rationality", type=float, default=3.0)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--scripted-run", action="store_true")
    return parser.parse_args()

def parse_summaries(path: Path) -> pd.DataFrame:
    """Parse and validate input summaries file"""
    try:
        summaries = pd.read_csv(path)
        required_cols = ['index', 'id', 'text', 'gold', 'summary', 'id_candidate']
        if not all(col in summaries.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns: {required_cols}")
        return summaries
    except Exception as e:
        raise ValueError(f"Error reading summaries file: {e}")

def compute_rsa(summaries: pd.DataFrame, model, tokenizer, device: str, 
                batch_size: int = 32, rationality: float = 3.0):
    """Compute RSA scores and rerank summaries"""
    results = []
    
    print("Computing RSA scores...")
    for name, group in tqdm(summaries.groupby(["id"])):
        # Initialize RSA reranker
        rsa_reranker = RSAReranking(
            model=model,
            tokenizer=tokenizer,
            device=device,
            candidates=group.summary.unique().tolist(),
            source_texts=group.text.unique().tolist(),
            batch_size=batch_size,
            rationality=rationality
        )
        
        # Compute RSA scores
        (
            best_rsa,          # Best summaries by RSA
            best_base,         # Best summaries by base model
            speaker_df,        # Speaker probabilities
            listener_df,       # Listener probabilities
            initial_listener,  # Initial listener model
            language_model_proba_df,  # Language model probabilities
            initial_consensuality_scores,  # Initial consensuality
            consensuality_scores,  # Final consensuality
        ) = rsa_reranker.rerank(t=2)  # Use 2 iterations

        # Get gold summary for this group
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
            "rationality": rationality,
            "text_candidates": group
        })

    return results

def main():
    args = parse_args()
    
    # Filter check
    if args.filter and args.filter not in args.summaries.stem:
        return
        
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(args.device)
    if "pegasus" in args.model_name:
        tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and process summaries
    print("Loading summaries...")
    summaries = parse_summaries(args.summaries)
    
    # Compute RSA scores
    results = compute_rsa(
        summaries=summaries,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        rationality=args.rationality
    )
    
    # Package results
    output_data = {
        "results": results,
        "metadata/reranking_model": args.model_name,
        "metadata/rsa_iterations": 2
    }
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{args.summaries.stem}-_-r{args.rationality}-_-rsa_reranked-{args.model_name.replace('/', '-')}.pk"
    
    print(f"Saving results to: {output_path}")
    with open(output_path, "wb") as f:
        dump(output_data, f)
        
    if args.scripted_run:
        print(output_path)

if __name__ == "__main__":
    main()