import torch
from typing import Tuple, Dict
from rsasumm.beam_search import RSAContextualDecoding

class AspectRSADecoding(RSAContextualDecoding):
    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)
        self.aspect_keywords = {
            "methodology": [
                # Core methodology
                "method", "approach", "algorithm", "framework", "technique", "procedure",
                "implementation", "process", "system", "model", "design",
                # Technical details
                "code", "pipeline", "structure", "protocol", "architecture",
                "mechanism", "solution", "strategy", "workflow",
                # Experimental
                "experiment", "validation", "evaluation", "test", "benchmark",
                "dataset", "measurement", "configuration", "parameter",
                # Mathematical
                "equation", "formula", "theorem", "proof", "analysis",
                "computation", "optimization", "function", "variable"
            ],
            "strengths": [
                # Innovation
                "novel", "innovative", "original", "unique", "breakthrough",
                "creative", "pioneering", "advanced", "state-of-the-art",
                # Performance
                "strong", "robust", "efficient", "effective", "powerful",
                "excellent", "outstanding", "impressive", "superior",
                # Quality
                "accurate", "precise", "reliable", "stable", "consistent",
                "thorough", "comprehensive", "detailed", "complete",
                # Advantages
                "better", "faster", "improved", "enhanced", "optimized",
                "outperform", "advantage", "benefit", "gain"
            ],
            "weaknesses": [
                # Core issues
                "limitation", "weakness", "drawback", "shortcoming", "flaw",
                "problem", "concern", "deficiency", "gap", "lack",
                # Missing
                "missing", "absent", "incomplete", "insufficient", "inadequate",
                "limited", "sparse", "scarce", "shallow",
                # Quality issues
                "poor", "weak", "inconsistent", "unstable", "unreliable",
                "inaccurate", "imprecise", "error", "failure",
                # Improvement needs
                "need", "require", "should", "improve", "revise",
                "modify", "update", "fix", "address"
            ],
            "impact": [
                # Direct impact
                "impact", "contribution", "influence", "effect", "significance",
                "importance", "relevance", "value", "benefit",
                # Application
                "application", "use", "deployment", "adoption", "integration",
                "practice", "industry", "field", "domain",
                # Potential
                "potential", "promise", "prospect", "future", "opportunity",
                "direction", "roadmap", "vision",
                # Progress
                "advance", "progress", "improvement", "development",
                "evolution", "breakthrough", "milestone", "achievement"
            ]
        }

    def compute_aspect_score(self, text: str, aspect: str) -> float:
        """Compute relevance score for given aspect"""
        if not text or not text.strip():
            return 0.0
            
        words = text.lower().split()
        if not words:
            return 0.0
            
        aspect_words = self.aspect_keywords.get(aspect, [])
        aspect_count = sum(1 for word in words if any(kw in word for kw in aspect_words))
        return min(1.0, aspect_count / max(1, len(words)) * 5)

    def compute_rsa_probas(
        self,
        logits: torch.Tensor,
        prior: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        aspect: str = None,
        rationality: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RSA probabilities with aspect awareness"""
        # Get base RSA probabilities
        S1, L1 = super().compute_rsa_probas(logits, prior, rationality)

        if aspect:
            # Get current generated text
            current_text = self.tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
            aspect_score = self.compute_aspect_score(current_text, aspect)
            
            # Apply aspect weighting
            aspect_weight = torch.tensor([aspect_score], device=self.device)
            S1 = S1 * aspect_weight.unsqueeze(-1)
            L1 = L1 * aspect_weight.unsqueeze(-1)

        return S1, L1

    def generate(
        self,
        text: str,
        aspect: str,
        max_length: int = 150,
        temperature: float = 0.7,
        rationality: float = 8.0
    ) -> str:
        """Generate aspect-focused summary"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        # Initialize generation
        generated_ids = torch.full(
            (1, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        # Generate summary
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    decoder_input_ids=generated_ids
                )
                
                logits = outputs.logits[:, -1, :]
                prior = torch.ones(1, device=self.device)
                
                S1, _ = self.compute_rsa_probas(
                    logits,
                    prior,
                    generated_ids,
                    aspect=aspect,
                    rationality=rationality
                )
                
                # Sample next token
                probs = torch.softmax(S1 / temperature, dim=-1)
                next_token = torch.multinomial(probs[0], num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)