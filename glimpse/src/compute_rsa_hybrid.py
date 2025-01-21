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
from functools import cache

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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--rationality", type=float, default=3.0)
    return parser.parse_args()

def compute_conditionned_likelihood(model, tokenizer, x: list, y: list, device, mean=True):
    """Compute likelihood of y given x using cross entropy loss"""
    assert len(x) == len(y), "x and y must have the same length"
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    batch_size = len(x)
    
    # Tokenize inputs
    x_tokens = tokenizer(x, return_tensors="pt", padding=True, truncation=True)
    y_tokens = tokenizer(y, return_tensors="pt", padding=True, truncation=True)
    
    # Move to device
    x_ids = x_tokens.input_ids.to(device)
    y_ids = y_tokens.input_ids.to(device)
    
    with torch.inference_mode():
        logits = model(
            input_ids=x_ids,
            decoder_input_ids=y_ids,
            attention_mask=x_tokens.attention_mask.to(device),
            decoder_attention_mask=y_tokens.attention_mask.to(device)
        ).logits
        
        # Get likelihood using cross entropy
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_ids = y_ids[..., 1:].contiguous()
        
        likelihood = -loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_ids.view(-1)
        )
        
        likelihood = likelihood.view(batch_size, -1).sum(-1)
        if mean:
            likelihood /= (y_ids != tokenizer.pad_token_id).float().sum(-1)
            
        return likelihood

class RSAReranker:
    def __init__(self, model, tokenizer, device, candidates, source_texts, batch_size=32, rationality=3.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.candidates = candidates
        self.source_texts = source_texts
        self.batch_size = batch_size
        self.rationality = rationality

    def likelihood_matrix(self) -> torch.Tensor:
        """Compute likelihood matrix between all source texts and candidates"""
        likelihood_matrix = torch.zeros(
            (len(self.source_texts), len(self.candidates))
        ).to(self.device)
        
        # Create all pairs
        pairs = []
        for i, source in enumerate(self.source_texts):
            for j, candidate in enumerate(self.candidates):
                pairs.append((i, j, source, candidate))
                
        # Process in batches
        batches = [
            pairs[i:i + self.batch_size]
            for i in range(0, len(pairs), self.batch_size)
        ]
        
        for batch in batches:
            sources = [p[2] for p in batch]
            candidates = [p[3] for p in batch]
            
            with torch.no_grad():
                likelihoods = compute_conditionned_likelihood(
                    self.model,
                    self.tokenizer,
                    sources,
                    candidates,
                    self.device,
                    mean=True
                )
                
            for k, (i, j, _, _) in enumerate(batch):
                likelihood_matrix[i, j] = likelihoods[k]
                
        return likelihood_matrix

    @cache
    def S(self, t):
        """Speaker function: P(u|w) ~ exp(α * log P(w|u))"""
        if t == 0:
            return self.initial_speaker_probas
        else:
            listener = self.L(t - 1)
            prod = listener * self.rationality
            return torch.log_softmax(prod, dim=-1)

    @cache
    def L(self, t):
        """Listener function: P(w|u) ~ P(u|w)P(w)"""
        speaker = self.S(t)
        return torch.log_softmax(speaker, dim=-2)

    def rerank(self, t=2):
        """Rerank candidates using RSA with t iterations"""
        self.initial_speaker_probas = self.likelihood_matrix()
        
        # Initial probabilities
        initial_listener = self.L(0)
        initial_speaker = self.S(0)
        
        # Final RSA probabilities
        speaker_probs = self.S(t)
        listener_probs = self.L(t)
        
        # Get best summaries
        best_rsa = []
        best_base = []
        
        for i in range(len(self.source_texts)):
            rsa_scores = speaker_probs[i].cpu().numpy()
            base_scores = initial_listener[i].cpu().numpy()
            
            best_rsa.append(self.candidates[rsa_scores.argmax()])
            best_base.append(self.candidates[base_scores.argmax()])
            
        return best_rsa, best_base, speaker_probs, listener_probs, initial_listener, initial_speaker

def process_group(model, tokenizer, group, device, batch_size, rationality):
    """Process a single group with RSA reranking"""
    try:
        # Initialize reranker
        reranker = RSAReranker(
            model=model,
            tokenizer=tokenizer,
            device=device,
            candidates=group.summary.unique().tolist(),
            source_texts=group.text.unique().tolist(),
            batch_size=batch_size,
            rationality=rationality
        )
        
        # Compute RSA scores
        best_rsa, best_base, speaker_probs, listener_probs, initial_listener, initial_speaker = reranker.rerank()
        
        return {
            "success": True,
            "best_rsa": best_rsa,
            "best_base": best_base,
            "speaker_probs": speaker_probs.cpu().numpy(),
            "listener_probs": listener_probs.cpu().numpy(),
            "initial_listener": initial_listener.cpu().numpy(),
            "initial_speaker": initial_speaker.cpu().numpy()
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    args = parse_args()

    if args.filter and args.filter not in args.summaries.stem:
        return

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32
    ).to(args.device)
    model.eval()

    logger.info("Loading tokenizer...")
    if "pegasus" in args.model_name:
        tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading summaries...")
    summaries = pd.read_csv(args.summaries)

    logger.info("Processing groups...")
    results = []
    
    for name, group in tqdm(summaries.groupby(["id"])):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        try:
            result = process_group(
                model, 
                tokenizer, 
                group, 
                args.device,
                args.batch_size,
                args.rationality
            )
            
            if result["success"]:
                results.append({
                    "id": name,
                    "best_rsa": result["best_rsa"],
                    "best_base": result["best_base"],
                    "speaker_probs": result["speaker_probs"],
                    "listener_probs": result["listener_probs"],
                    "initial_listener": result["initial_listener"],
                    "initial_speaker": result["initial_speaker"],
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