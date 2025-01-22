import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
import argparse
from tqdm import tqdm
from pickle import dump
import torch
import gc
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--scripted-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--rationality", type=float, default=3.0)
    return parser.parse_args()

class RSAReranker:
    def __init__(self, model, tokenizer, device, candidates, source_texts, batch_size=8, rationality=3.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.candidates = candidates
        self.source_texts = source_texts
        self.batch_size = batch_size
        self.rationality = rationality
        
        # Ensure max sizes for tensors
        self.max_source_len = 512
        self.max_target_len = 128

    def compute_likelihood(self, x: list, y: list, mean=True):
        """Safe computation of likelihood scores"""
        try:
            # Tokenize with explicit max lengths
            x_tokens = self.tokenizer(
                x, 
                max_length=self.max_source_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            y_tokens = self.tokenizer(
                y,
                max_length=self.max_target_len,
                padding="max_length", 
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model(
                    **x_tokens,
                    labels=y_tokens["input_ids"]
                )
                
                # Get loss per token
                loss = outputs.loss
                if mean:
                    mask = (y_tokens["input_ids"] != self.tokenizer.pad_token_id)
                    loss = loss / mask.float().sum()
                
                return -loss.detach()
                
        except Exception as e:
            logger.error(f"Error in likelihood computation: {str(e)}")
            return None

    def compute_matrix(self):
        """Compute likelihood matrix with batching and error handling"""
        matrix = torch.zeros((len(self.source_texts), len(self.candidates))).to(self.device)
        
        # Process in batches
        for i in range(0, len(self.source_texts), self.batch_size):
            sources = self.source_texts[i:i + self.batch_size]
            
            for j in range(0, len(self.candidates), self.batch_size):
                candidates = self.candidates[j:j + self.batch_size]
                
                # Create pairs
                pairs_x = []
                pairs_y = []
                indices = []
                
                for si, source in enumerate(sources):
                    for ci, candidate in enumerate(candidates):
                        pairs_x.append(source)
                        pairs_y.append(candidate)
                        indices.append((i + si, j + ci))
                
                # Compute likelihoods for batch
                if pairs_x:
                    scores = self.compute_likelihood(pairs_x, pairs_y)
                    if scores is not None:
                        for idx, (si, ci) in enumerate(indices):
                            if si < matrix.shape[0] and ci < matrix.shape[1]:
                                matrix[si, ci] = scores[idx] if isinstance(scores, torch.Tensor) else scores
                
                # Clear cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        return matrix

    def rerank(self, t=2):
        """Rerank summaries using RSA with step-by-step logging"""
        try:
            logger.info("Starting RSA computation...")
            
            # Get initial scores
            logger.info("Computing initial scores...")
            initial_scores = self.compute_matrix()
            logger.info(f"Initial scores shape: {initial_scores.shape}")
            
            # Print example initial scores for first source text
            logger.info("\nInitial scores for first source text:")
            for i, score in enumerate(initial_scores[0]):
                logger.info(f"Candidate {i}: {score.item():.4f}")
            
            # Compute speaker scores
            speaker_scores = torch.log_softmax(initial_scores, dim=-1)
            logger.info("\nInitial speaker probabilities:")
            for i, score in enumerate(speaker_scores[0]):
                logger.info(f"Candidate {i}: {score.item():.4f}")
            
            # Iterate RSA
            for step in range(t):
                logger.info(f"\nRSA Iteration {step + 1}")
                
                # Listener update
                listener_scores = torch.log_softmax(speaker_scores, dim=0)
                logger.info(f"Listener scores for first source text:")
                for i, score in enumerate(listener_scores[0]):
                    logger.info(f"Candidate {i}: {score.item():.4f}")
                
                # Speaker update with rationality
                speaker_scores = torch.log_softmax(
                    initial_scores + self.rationality * listener_scores,
                    dim=-1
                )
                logger.info(f"Updated speaker scores for first source text:")
                for i, score in enumerate(speaker_scores[0]):
                    logger.info(f"Candidate {i}: {score.item():.4f}")
            
            logger.info("\nFinal Selection:")
            # Get best summaries
            best_rsa = []
            best_base = []
            
            with torch.no_grad():
                rsa_indices = speaker_scores.argmax(dim=1).cpu()
                base_indices = initial_scores.argmax(dim=1).cpu()
                
                for i in range(len(self.source_texts)):
                    rsa_idx = rsa_indices[i].item()
                    base_idx = base_indices[i].item()
                    
                    if rsa_idx < len(self.candidates):
                        best_rsa.append(self.candidates[rsa_idx])
                        if i == 0:  # Print for first source text
                            logger.info(f"Selected RSA candidate {rsa_idx} with score {speaker_scores[0][rsa_idx].item():.4f}")
                    else:
                        best_rsa.append(self.candidates[0])
                        
                    if base_idx < len(self.candidates):
                        best_base.append(self.candidates[base_idx])
                        if i == 0:  # Print for first source text
                            logger.info(f"Selected base candidate {base_idx} with score {initial_scores[0][base_idx].item():.4f}")
                    else:
                        best_base.append(self.candidates[0])
            
            return {
                "success": True,
                "best_rsa": best_rsa,
                "best_base": best_base,
                "speaker_scores": speaker_scores.cpu().numpy(),
                "initial_scores": initial_scores.cpu().numpy()
            }
            
        except Exception as e:
            logger.error(f"Error in RSA computation: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

def process_summaries(model, tokenizer, summaries_df, device, batch_size, rationality):
    """Process all summaries with RSA reranking"""
    results = []
    
    for name, group in tqdm(summaries_df.groupby(["id"])):
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        try:
            reranker = RSAReranker(
                model=model,
                tokenizer=tokenizer,
                device=device,
                candidates=group.summary.unique().tolist(),
                source_texts=group.text.unique().tolist(),
                batch_size=batch_size,
                rationality=rationality
            )
            
            result = reranker.rerank()
            
            if result["success"]:
                results.append({
                    "id": name,
                    "best_rsa": result["best_rsa"],
                    "best_base": result["best_base"],
                    "speaker_scores": result["speaker_scores"],
                    "initial_scores": result["initial_scores"],
                    "gold": group['gold'].iloc[0]
                })
            else:
                results.append({
                    "id": name,
                    "error": result["error"],
                    "gold": group['gold'].iloc[0]
                })
                
        except Exception as e:
            logger.error(f"Error processing group {name}: {str(e)}")
            results.append({
                "id": name,
                "error": str(e),
                "gold": group['gold'].iloc[0]
            })
            
    return results

def main():
    args = parse_args()

    if args.filter and args.filter not in args.summaries.stem:
        return

    # Set memory management
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.backends.cuda.max_split_size_mb = 128  # Reduce memory fragmentation
    
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map='auto',  # Use automatic device mapping
        low_cpu_mem_usage=True
    )
    model = model.eval()

    logger.info("Loading tokenizer...")
    if "pegasus" in args.model_name:
        tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading summaries...")
    summaries = pd.read_csv(args.summaries)

    logger.info("Processing with RSA...")
    results = process_summaries(
        model=model,
        tokenizer=tokenizer,
        summaries_df=summaries,
        device=args.device,
        batch_size=args.batch_size,
        rationality=args.rationality
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{args.summaries.stem}-_-rsa_reranked-{args.model_name.replace('/', '-')}.pk"
    
    with open(output_path, "wb") as f:
        dump({
            "results": results,
            "metadata/model": args.model_name,
            "metadata/rationality": args.rationality
        }, f)
    
    logger.info(f"Results saved to: {output_path}")
    if args.scripted_run:
        print(output_path)

if __name__ == "__main__":
    main()