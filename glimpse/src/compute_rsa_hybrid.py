import sys
import os
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer, AutoConfig
import argparse
from tqdm import tqdm
from pickle import dump
import torch
import gc
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--rationality", type=float, default=3.0)
    parser.add_argument("--rsa_iterations", type=int, default=2)
    return parser.parse_args()

def load_model_safely(model_name, device):
    """Load model with safeguards against meta tensor errors."""
    try:
        # Load model configuration
        config = AutoConfig.from_pretrained(model_name)
        
        # Load model using the configuration
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
        
        # Move model to the specified device
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")

class RSAReranker:
    def __init__(self, model, tokenizer, device, candidates, source_texts, batch_size=8, rationality=3.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.candidates = candidates
        self.source_texts = source_texts
        self.batch_size = batch_size
        self.rationality = rationality

    def compute_likelihood(self, texts, summaries):
        """Compute likelihood scores."""
        with torch.inference_mode():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            labels = self.tokenizer(
                summaries,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs, labels=labels["input_ids"])
            return -outputs.loss

    def compute_matrix(self):
        """Compute likelihood matrix between texts and candidates."""
        logger.info("Computing likelihood matrix...")
        matrix = torch.zeros((len(self.source_texts), len(self.candidates)), device=self.device)
        
        for i in tqdm(range(0, len(self.source_texts), self.batch_size)):
            for j in range(0, len(self.candidates), self.batch_size):
                texts = self.source_texts[i:i + self.batch_size]
                cands = self.candidates[j:j + self.batch_size]
                
                for k, text in enumerate(texts):
                    for l, cand in enumerate(cands):
                        try:
                            score = self.compute_likelihood([text], [cand])
                            matrix[i + k, j + l] = score.item()
                        except Exception as e:
                            logger.warning(f"Error computing likelihood: {e}")
                
                # Clear memory
                torch.cuda.empty_cache()
                gc.collect()
                
        return matrix

    def rerank(self, t=2):
        """Rerank summaries using RSA."""
        try:
            initial_scores = self.compute_matrix()
            logger.info("\nInitial scores computed")
            
            speaker_scores = torch.log_softmax(initial_scores, dim=-1)
            
            for step in range(t):
                logger.info(f"\nRSA Iteration {step + 1}/{t}")
                listener_scores = torch.log_softmax(speaker_scores, dim=0)
                speaker_scores = torch.log_softmax(
                    initial_scores + self.rationality * listener_scores,
                    dim=-1
                )
            
            rsa_indices = speaker_scores.argmax(dim=1).cpu()
            base_indices = initial_scores.argmax(dim=1).cpu()
            
            best_rsa = [self.candidates[i.item()] for i in rsa_indices]
            best_base = [self.candidates[i.item()] for i in base_indices]
            
            return {
                "success": True,
                "best_rsa": best_rsa,
                "best_base": best_base,
                "speaker_scores": speaker_scores.cpu().numpy(),
                "initial_scores": initial_scores.cpu().numpy()
            }
        except Exception as e:
            logger.error(f"Error in RSA computation: {str(e)}")
            return {"success": False, "error": str(e)}

def main():
    args = parse_args()

    if args.filter and args.filter not in args.summaries.stem:
        return

    logger.info(f"Loading model: {args.model_name}")
    device = args.device if torch.cuda.is_available() else "cpu"
    
    try:
        model = load_model_safely(args.model_name, device)
    except RuntimeError as e:
        logger.error(f"Error loading model: {e}")
        raise

    logger.info(f"Model loaded successfully on device: {device}")

    logger.info("Loading tokenizer...")
    tokenizer = (PegasusTokenizer if "pegasus" in args.model_name else AutoTokenizer).from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading summaries...")
    summaries = pd.read_csv(args.summaries)
    results = []
    
    logger.info("Processing groups...")
    for name, group in tqdm(summaries.groupby(["id"])):
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            reranker = RSAReranker(
                model=model,
                tokenizer=tokenizer, 
                device=device,
                candidates=group.summary.unique().tolist(),
                source_texts=group.text.unique().tolist(),
                batch_size=args.batch_size,
                rationality=args.rationality
            )
            
            result = reranker.rerank(t=args.rsa_iterations)
            
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.summaries.stem}-rsa_reranked-{args.model_name.replace('/', '-')}.pk"
    
    with open(output_path, "wb") as f:
        dump({
            "results": results,
            "metadata": {
                "model": args.model_name,
                "rationality": args.rationality,
                "rsa_iterations": args.rsa_iterations
            }
        }, f)
    
    logger.info(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()