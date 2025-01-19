import sys
import os.path
# Add path to find rsasumm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
import argparse
from tqdm import tqdm
from pickle import dump
import torch

from rsasumm.rsa_reranker import RSAReranking

DESC = """
Compute the RSA matrices for all the set of multi-document samples and dump these along with additional information in a pickle file.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/pegasus-arxiv")
    parser.add_argument("--summaries", type=Path, default="")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--scripted-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)  # Reduced batch size
    parser.add_argument("--max_length", type=int, default=512)  # Add max length parameter
    return parser.parse_args()

def parse_summaries(path: Path) -> pd.DataFrame:
    try:
        summaries = pd.read_csv(path)
    except:
        raise ValueError(f"Unknown dataset {path}")

    # Check if the dataframe has the right columns
    if not all(
        col in summaries.columns 
        for col in ["index", "id", "text", "gold", "summary", "id_candidate"]
    ):
        raise ValueError(
            "The dataframe must have columns ['index', 'id', 'text', 'gold', 'summary', 'id_candidate']"
        )

    return summaries

def compute_rsa(summaries: pd.DataFrame, model, tokenizer, device, batch_size: int, max_length: int):
    results = []
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    for name, group in tqdm(summaries.groupby(["id"])):
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Truncate texts if they're too long
            truncated_texts = [text[:max_length*4] for text in group.text.unique().tolist()]
            
            rsa_reranker = RSAReranking(
                model,
                tokenizer,
                device=device,
                candidates=group.summary.unique().tolist(),
                source_texts=truncated_texts,
                batch_size=batch_size,
                rationality=3,
            )
            
            with torch.inference_mode():  # Use inference mode instead of no_grad
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

            gold = group['gold'].tolist()[0]

            results.append(
                {
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
                }
            )
        except RuntimeError as e:
            print(f"Error processing group {name}: {str(e)}")
            # Add empty result for failed group
            results.append({
                "id": name,
                "error": str(e),
                "gold": group['gold'].tolist()[0]
            })
            continue

    return results

def main():
    args = parse_args()

    if args.filter is not None:
        if args.filter not in args.summaries.stem:
            return

    # Set memory efficient attention
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Load the model and the tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, 
                                                 device_map=args.device,
                                                 torch_dtype=torch.float16)  # Use fp16
    if "pegasus" in args.model_name:
        tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load the summaries
    print("Loading summaries...")
    summaries = parse_summaries(args.summaries)

    # Rerank the summaries
    print("Computing RSA scores...")
    results = compute_rsa(summaries, model, tokenizer, args.device, args.batch_size, args.max_length)
    results = {"results": results}

    results["metadata/reranking_model"] = args.model_name
    results["metadata/rsa_iterations"] = 3

    # Save the summaries
    print("Saving results...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f"{args.summaries.stem}-_-r3-_-rsa_reranked-{args.model_name.replace('/', '-')}.pk"

    with open(output_path, "wb") as f:
        dump(results, f)
        
    if args.scripted_run: 
        print(output_path)

if __name__ == "__main__":
    main()