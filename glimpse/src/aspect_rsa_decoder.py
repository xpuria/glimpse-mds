import torch
from typing import Tuple, Dict
from rsasumm.beam_search import RSAContextualDecoding, compute_rsa_probas, sample_from_probs

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
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        do_sample: bool = True,
        top_p: float = None,
        top_k: int = None,
        temperature: float = 1.0,
        rationality: float = 8.0,
        process_logits_before_rsa: bool = True,
        beam_scores: torch.Tensor = None,
        aspect: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RSA probabilities with aspect awareness"""
        # Get base RSA computations
        S1, L1 = super().compute_rsa_probas(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            do_sample,
            top_p,
            top_k,
            temperature,
            rationality,
            process_logits_before_rsa,
            beam_scores
        )

        if aspect:
            # Get current generated text
            current_text = self.tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
            aspect_score = self.compute_aspect_score(current_text, aspect)
            
            # Apply aspect weighting
            aspect_weight = torch.tensor([aspect_score], device=self.device)
            aspect_weight = aspect_weight.view(*[1 for _ in range(len(S1.shape) - 1)], 1)
            S1 = S1 * aspect_weight
            L1 = L1 * aspect_weight

        return S1, L1

    def generate(
        self,
        target_id: int,
        source_texts_ids: torch.Tensor,
        source_text_attention_mask: torch.Tensor,
        aspect: str,
        max_length: int = 100,
        num_beams: int = 8,
        do_sample=True,
        top_p: float = 0.95,
        temperature: float = 1.0,
        rationality: float = 8.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate with aspect-aware RSA
        """
        self.num_beam = num_beams
        self.world_size = source_texts_ids.shape[0]

        self.prior = torch.ones((self.world_size, self.num_beam)).to(self.device) / self.world_size
        beam_scores = torch.zeros(self.num_beam).to(self.device)

        # Initialize decoder inputs
        decoder_input_ids = torch.full(
            (self.num_beam, 2),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        decoder_attention_mask = torch.ones_like(decoder_input_ids).to(self.device)

        new_beams = []
        finished_beams = []

        # Generate with beam search
        for _ in range(max_length):
            # Compute RSA probabilities with aspect awareness
            S1, L1 = self.compute_rsa_probas(
                source_texts_ids,
                source_text_attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                rationality=rationality,
                aspect=aspect
            )

            # Sample from probabilities
            idx_beam, idx_token, tokens_scores = sample_from_probs(
                S1[target_id].squeeze(), num_beams, do_sample
            )

            # Create new beams
            new_beams = []
            for idx_t, idx_b, token_score in zip(idx_token, idx_beam, tokens_scores):
                new_beams.append(
                    (
                        decoder_input_ids[idx_b].tolist() + [idx_t.item()],
                        beam_scores[idx_b] + token_score.item(),
                        L1[:, idx_b, idx_t.item()],
                    )
                )

            # Process beams as in original implementation
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
            new_beams = new_beams[: self.num_beam]

            # Check for finished beams
            _new_beams = []
            for beam in new_beams:
                if beam[0][-1] == self.tokenizer.eos_token_id:
                    finished_beams.append(beam)
                else:
                    _new_beams.append(beam)

            new_beams = _new_beams
            if len(new_beams) == 0:
                break

            # Update beams
            max_beam_len = max(len(x[0]) for x in new_beams)
            new_beams = [
                (
                    x[0] + [self.tokenizer.pad_token_id] * (max_beam_len - len(x[0])),
                    x[1],
                    x[2],
                )
                for x in new_beams
            ]

            # Update beam scores and decoder inputs
            beam_scores = torch.tensor([x[1] for x in new_beams]).to(self.device)
            decoder_input_ids = torch.tensor(
                [x[0] for x in new_beams], device=self.device
            )
            decoder_attention_mask = (
                decoder_input_ids != self.tokenizer.pad_token_id
            ).long()

            self.prior = torch.stack([x[2] for x in new_beams], dim=1).to(self.device)

        # Process results
        results = []
        max_beam_len = max(len(x[0]) for x in finished_beams + new_beams)
        for x in finished_beams + new_beams:
            results.append(
                (
                    x[0] + [self.tokenizer.pad_token_id] * (max_beam_len - len(x[0])),
                    x[1],
                    x[2],
                )
            )

        decoder_input_ids = torch.tensor([x[0] for x in results], device=self.device)
        beam_scores = torch.tensor([x[1] for x in results]).to(self.device)

        return decoder_input_ids, beam_scores