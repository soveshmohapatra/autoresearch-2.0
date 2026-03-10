"""
Hyperparameter Agent - Specialized in training hyperparameters.
Suggests modifications to: batch sizes, sequence length, data loading.
"""

from typing import List, Dict, Any


class HyperparameterAgent:
    """Agent specialized in hyperparameter experimentation."""
    
    def __init__(self, memory=None):
        self.memory = memory
        self.name = "HyperparameterAgent"
        self.specialties = [
            "batch_size",
            "sequence_length",
            "gradient_accumulation",
            "data_loading",
        ]
    
    def get_suggestions(self, current_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hyperparameter modification suggestions."""
        suggestions = []
        
        context = self.memory.get_context_for_agent(self.name) if self.memory else ""
        
        # Batch size modifications
        current_batch = current_config.get("total_batch_size", 2**19)
        current_device_batch = current_config.get("device_batch_size", 128)
        
        suggestions.extend([
            {
                "type": "batch",
                "change": f"Increase total batch size from {current_batch} to {current_batch * 2}",
                "rationale": "Larger batches provide more stable gradients",
                "risk": "medium",
                "expected_impact": "More stable training, possible generalization improvement",
            },
            {
                "type": "batch",
                "change": f"Decrease total batch size from {current_batch} to {current_batch // 2}",
                "rationale": "Smaller batches add noise, may help escape local minima",
                "risk": "low",
                "expected_impact": "Noisier but potentially better generalization",
            },
            {
                "type": "batch",
                "change": f"Increase device batch size (currently {current_device_batch})",
                "rationale": "Better GPU utilization, faster training",
                "risk": "low",
                "expected_impact": "Faster training if memory allows",
            },
        ])
        
        # Sequence length modifications
        current_seq = current_config.get("max_seq_len", 2048)
        suggestions.extend([
            {
                "type": "sequence",
                "change": f"Increase sequence length from {current_seq} to {current_seq * 2}",
                "rationale": "Longer context allows learning longer dependencies",
                "risk": "medium",
                "expected_impact": "Better for tasks requiring long-range context",
            },
            {
                "type": "sequence",
                "change": f"Decrease sequence length from {current_seq} to {current_seq // 2}",
                "rationale": "Shorter sequences train faster, more updates per time",
                "risk": "low",
                "expected_impact": "Faster training, more gradient updates",
            },
        ])
        
        # Gradient accumulation
        suggestions.extend([
            {
                "type": "grad_accum",
                "change": "Reduce gradient accumulation steps",
                "rationale": "More frequent updates may improve convergence",
                "risk": "low",
                "expected_impact": "Slightly noisier but potentially better",
            },
        ])
        
        # Data loading
        suggestions.extend([
            {
                "type": "data",
                "change": "Increase dataloader buffer size",
                "rationale": "Better data shuffling, more randomness",
                "risk": "low",
                "expected_impact": "Slight generalization improvement",
            },
        ])
        
        return self._rank_suggestions(suggestions, context)
    
    def _rank_suggestions(self, suggestions: List[Dict], context: str) -> List[Dict]:
        """Rank suggestions."""
        risk_scores = {"low": 1, "medium": 2, "high": 3}
        for sug in suggestions:
            sug["priority"] = risk_scores.get(sug["risk"], 2)
        return sorted(suggestions, key=lambda x: x["priority"])
    
    def get_code_modifications(self, suggestion_type: str) -> str:
        """Get code modification hints."""
        hints = {
            "batch": "Modify TOTAL_BATCH_SIZE and DEVICE_BATCH_SIZE in train.py.",
            "sequence": "Modify MAX_SEQ_LEN in prepare.py (requires re-prep) or override in train.py.",
            "grad_accum": "Gradient accumulation is automatic based on batch sizes.",
            "data": "Modify buffer_size in make_dataloader() call.",
        }
        return hints.get(suggestion_type, "See train.py hyperparameters section.")
