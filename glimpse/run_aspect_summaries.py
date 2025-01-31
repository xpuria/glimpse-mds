import argparse
from pathlib import Path
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import datetime
from glimpse.src.aspect_rsa_decoder import AspectRSADecoding

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--dataset_path", type=Path, default="data/processed/all_reviews_2017.csv")
    parser.add_argument("--output_dir", type=str, default="output/aspect_summaries")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load data
    df = pd.read_csv(args.dataset_path)
    if args.limit:
        df = df.head(args.limit)

    # Initialize model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Initialize aspect RSA decoder
    summarizer = AspectRSADecoding(model, tokenizer, args.device)
    
    # Generate summaries for each aspect
    aspects = ["methodology", "strengths", "weaknesses", "impact"]
    results = []
    
    for aspect in aspects:
        print(f"\nGenerating summaries for aspect: {aspect}")
        
        for i, row in tqdm(df.iterrows(), total=len(df)):
            summary = summarizer.generate(
                text=row['review'],
                aspect=aspect,
                max_length=150,
                temperature=0.7,
                rationality=8.0
            )
            
            results.append({
                'id': row.get('id', i),
                'original': row['review'],
                'summary': summary,
                'aspect': aspect,
                'gold': row.get('gold', '')
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    
    # Create output path
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M-%S")
    output_path = (
        Path(args.output_dir) /
        f"aspect_summaries-{args.model_name.replace('/', '_')}-{date}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()