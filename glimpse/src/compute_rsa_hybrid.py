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
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--rationality", type=float, default=3.0)
    parser.add_argument("--rsa_iterations", type=int, default=2)
    return parser.parse_args()

class RSAReranker:
    def __init__(self, model, tokenizer, device, candidates, source_texts, 
                 batch_size=8, rationality=3.0, max_source_len=512, max_target_len=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device != "cuda" else model.device
        self.candidates = candidates
        self.source_texts = source_texts
        self.batch_size = batch_size
        self.rationality = rationality
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def compute_likelihood(self, x: list, y: list, mean=True):
        try:
            # Tokenize with explicit handling of padding
            x_tokens = self.tokenizer(
                x, 
                max_length=self.max_source_len,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            y_tokens = self.tokenizer(
                y,
                max_length=self.max_target_len,
                padding=True, 
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            x_tokens = {k: v.to(self.device) for k, v in x_tokens.items()}
            y_tokens = {k: v.to(self.device) for k, v in y_tokens.items()}
            
            # Forward pass
            with torch.inference_mode():
                outputs = self.model(**x_tokens, labels=y_tokens["input_ids"])
                loss = outputs.loss
                
                if mean:
                    # Average loss over non-padding tokens
                    mask = (y_tokens["input_ids"] != self.tokenizer.pad_token_id)
                    loss = loss / mask.float().sum()
                
                return -loss.detach()
                
        except Exception as e:
            logger.error(f"Error in likelihood computation: {str(e)}")
            return None

    def compute_matrix(self):
        """Compute likelihood matrix with batching"""
        logger.info("Computing likelihood matrix...")
        matrix = torch.zeros((len(self.source_texts), len(self.candidates))).to(self.device)
        
        # Process in batches
        for i in tqdm(range(0, len(self.source_texts), self.batch_size)):
            batch_sources = self.source_texts[i:i + self.batch_size]
            
            for j in range(0, len(self.candidates), self.batch_size):
                batch_candidates = self.candidates[j:j + self.batch_size]
                
                # Create pairs for current mini-batch
                pairs_x = []
                pairs_y = []
                indices = []
                
                for si, source in enumerate(batch_sources):
                    for ci, candidate in enumerate(batch_candidates):
                        pairs_x.append(source)
                        pairs_y.append(candidate)
                        indices.append((i + si, j + ci))
                
                if pairs_x:
                    logger.debug(f"Processing batch with {len(pairs_x)} pairs")
                    scores = self.compute_likelihood(pairs_x, pairs_y)
                    
                    if scores is not None:
                        for idx, (si, ci) in enumerate(indices):
                            if si < matrix.shape[0] and ci < matrix.shape[1]:
                                matrix[si, ci] = scores[idx] if isinstance(scores, torch.Tensor) else scores
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        logger.info("Matrix computation complete")
        return matrix

    def rerank(self, t=2):
        """Rerank summaries with iterative RSA computation"""
        try:
            logger.info("\n" + "="*50)
            logger.info("Starting RSA computation")
            logger.info("="*50)
            
            # Initial likelihood computation
            initial_scores = self.compute_matrix()
            logger.info(f"\nInitial matrix shape: {initial_scores.shape}")
            
            # Log initial scores for first text
            logger.info("\nInitial scores (first text):")
            for i, score in enumerate(initial_scores[0]):
                logger.info(f"Candidate {i}: {score.item():.4f}")
            
            # Initialize speaker probabilities
            speaker_scores = torch.log_softmax(initial_scores, dim=-1)
            logger.info("\nInitial speaker probabilities (first text):")
            for i, score in enumerate(speaker_scores[0]):
                prob = score.exp().item()
                logger.info(f"Candidate {i}: {prob:.4f}")
            
            # RSA iterations
            for step in range(t):
                logger.info(f"\n{'-'*20} RSA Iteration {step + 1} {'-'*20}")
                
                # Listener update
                listener_scores = torch.log_softmax(speaker_scores, dim=0)
                logger.info("\nListener probabilities (first text):")
                for i, score in enumerate(listener_scores[0]):
                    prob = score.exp().item()
                    logger.info(f"Candidate {i}: {prob:.4f}")
                
                # Speaker update
                speaker_scores = torch.log_softmax(
                    initial_scores + self.rationality * listener_scores,
                    dim=-1
                )
                
                logger.info("\nUpdated speaker probabilities (first text):")
                for i, score in enumerate(speaker_scores[0]):
                    prob = score.exp().item()
                    logger.info(f"Candidate {i}: {prob:.4f}")
            
            # Get best summaries
            best_rsa = []
            best_base = []
            
            with torch.no_grad():
                rsa_indices = speaker_scores.argmax(dim=1).cpu()
                base_indices = initial_scores.argmax(dim=1).cpu()
                
                # Process results
                for i in range(len(self.source_texts)):
                    rsa_idx = rsa_indices[i].item()
                    base_idx = base_indices[i].item()
                    
                    # RSA selection
                    if rsa_idx < len(self.candidates):
                        best_rsa.append(self.candidates[rsa_idx])
                        if i == 0:
                            logger.info(f"\nRSA Selection for first text:")
                            logger.info(f"Selected candidate {rsa_idx}")
                            logger.info(f"RSA probability: {speaker_scores[0][rsa_idx].exp().item():.4f}")
                    else:
                        best_rsa.append(self.candidates[0])
                    
                    # Base model selection
                    if base_idx < len(self.candidates):
                        best_base.append(self.candidates[base_idx])
                        if i == 0:
                            logger.info(f"\nBase Selection for first text:")
                            logger.info(f"Selected candidate {base_idx}")
                            logger.info(f"Base score: {initial_scores[0][base_idx].item():.4f}")
                    else:
                        best_base.append(self.candidates[0])
            
            return {
                "success": True,
                "best_rsa": best_rsa,
                "best_base": best_base,
                "speaker_scores": speaker_scores.cpu().numpy(),
                "initial_scores": initial_scores.cpu().numpy(),
                "probabilities": {
                    "final_speaker": speaker_scores.exp().cpu().numpy(),
                    "final_listener": listener_scores.exp().cpu().numpy(),
                    "initial": torch.softmax(initial_scores, dim=-1).cpu().numpy()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in RSA computation: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

def process_groups(model, tokenizer, summaries_df, args):
    results = []
    
    for name, group in tqdm(summaries_df.groupby(["id"]), desc="Processing groups"):
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        try:
            logger.info(f"\nProcessing group {name}")
            
            reranker = RSAReranker(
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                candidates=group.summary.unique().tolist(),
                source_texts=group.text.unique().tolist(),
                batch_size=args.batch_size,
                rationality=args.rationality,
                max_source_len=args.max_source_length,
                max_target_len=args.max_target_length
            )
            
            result = reranker.rerank(t=args.rsa_iterations)
            
            if result["success"]:
                results.append({
                    "id": name,
                    "best_rsa": result["best_rsa"],
                    "best_base": result["best_base"],
                    "speaker_scores": result["speaker_scores"],
                    "initial_scores": result["initial_scores"],
                    "probabilities": result["probabilities"],
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
        
        # Small delay between groups
        time.sleep(0.1)
            
    return results

def main():
    args = parse_args()

    if args.filter and args.filter not in args.summaries.stem:
        return
        
    # Configure CUDA
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.backends.cuda.max_split_size_mb = 128
    
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map='auto',
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
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading summaries...")
    summaries = pd.read_csv(args.summaries)

    logger.info("Starting RSA processing...")
    results = process_groups(model, tokenizer, summaries, args)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{args.summaries.stem}-_-rsa_reranked-{args.model_name.replace('/', '-')}.pk"
    
    with open(output_path, "wb") as f:
        dump({
            "results": results,
            "metadata": {
                "model": args.model_name,
                "rationality": args.rationality,
                "rsa_iterations": args.rsa_iterations,
                "batch_size": args.batch_size,
                "max_source_length": args.max_source_length,
                "max_target_length": args.max_target_length
            }
        }, f)
    
    logger.info(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()