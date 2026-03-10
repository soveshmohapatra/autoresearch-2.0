"""
Optimizer Agent - Specialized in optimization strategies.
Suggests modifications to: learning rates, schedulers, optimizer hyperparameters.
"""

from typing import List, Dict, Any


class OptimizerAgent:
    """Agent specialized in optimization experimentation."""
    
    def __init__(self, memory=None):
        self.memory = memory
        self.name = "OptimizerAgent"
        self.specialties = [
            "learning_rates",
            "lr_schedules",
            "optimizer_hyperparameters",
            "weight_decay",
            "momentum",
        ]
    
    def get_suggestions(self, current_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimizer modification suggestions."""
        suggestions = []
        
        context = self.memory.get_context_for_agent(self.name) if self.memory else ""
        
        # Learning rate modifications
        current_lr = current_config.get("matrix_lr", 0.04)
        suggestions.extend([
            {
                "type": "lr",
                "change": f"Increase matrix LR from {current_lr} to {current_lr * 1.5}",
                "rationale": "Higher LR may speed up convergence if not at optimum",
                "risk": "medium",
                "expected_impact": "Faster convergence or divergence (watch for loss spikes)",
            },
            {
                "type": "lr",
                "change": f"Decrease matrix LR from {current_lr} to {current_lr * 0.5}",
                "rationale": "Lower LR provides more stable training",
                "risk": "low",
                "expected_impact": "More stable but slower convergence",
            },
            {
                "type": "lr",
                "change": f"Increase embedding LR (currently {current_config.get('embedding_lr', 0.6)})",
                "rationale": "Embeddings often benefit from higher LR",
                "risk": "medium",
                "expected_impact": "Better token representations",
            },
        ])
        
        # LR schedule modifications
        suggestions.extend([
            {
                "type": "schedule",
                "change": "Add warmup (5-10% of training)",
                "rationale": "Warmup stabilizes early training, prevents divergence",
                "risk": "low",
                "expected_impact": "More stable training, possible improvement",
            },
            {
                "type": "schedule",
                "change": "Use cosine decay instead of linear warmdown",
                "rationale": "Cosine decay often provides better convergence",
                "risk": "low",
                "expected_impact": "0.005-0.02 bpb improvement",
            },
            {
                "type": "schedule",
                "change": "Implement One-cycle policy",
                "rationale": "One-cycle can find better minima faster",
                "risk": "medium",
                "expected_impact": "Faster convergence to good solution",
            },
        ])
        
        # Weight decay modifications
        current_wd = current_config.get("weight_decay", 0.2)
        suggestions.extend([
            {
                "type": "weight_decay",
                "change": f"Increase weight decay from {current_wd} to {current_wd * 1.5}",
                "rationale": "Higher regularization may improve generalization",
                "risk": "medium",
                "expected_impact": "Possible improvement if overfitting",
            },
            {
                "type": "weight_decay",
                "change": f"Decrease weight decay from {current_wd} to {current_wd * 0.5}",
                "rationale": "Lower regularization allows more model capacity",
                "risk": "low",
                "expected_impact": "Better fit if underfitting",
            },
            {
                "type": "weight_decay",
                "change": "Use decoupled weight decay (AdamW style)",
                "rationale": "Decoupled WD provides cleaner regularization",
                "risk": "low",
                "expected_impact": "More predictable regularization effect",
            },
        ])
        
        # Momentum/beta modifications
        suggestions.extend([
            {
                "type": "momentum",
                "change": "Increase Muon momentum (currently 0.95)",
                "rationale": "Higher momentum smooths gradients",
                "risk": "low",
                "expected_impact": "Smoother training trajectory",
            },
            {
                "type": "momentum",
                "change": "Adjust Adam betas from (0.8, 0.95) to (0.9, 0.95)",
                "rationale": "Higher beta1 provides more momentum",
                "risk": "low",
                "expected_impact": "More stable but slower adaptation",
            },
        ])
        
        # Alternative optimizers
        suggestions.extend([
            {
                "type": "alternative",
                "change": "Try Lion optimizer instead of Muon+AdamW",
                "rationale": "Lion is memory-efficient, shows good results",
                "risk": "high",
                "expected_impact": "Similar performance with less memory",
            },
            {
                "type": "alternative",
                "change": "Try Adafactor for adaptive LR without momentum",
                "rationale": "Adafactor adapts LR per-parameter",
                "risk": "high",
                "expected_impact": "May find different minima",
            },
        ])
        
        return self._rank_suggestions(suggestions, context)
    
    def _rank_suggestions(self, suggestions: List[Dict], context: str) -> List[Dict]:
        """Rank suggestions based on risk and expected impact."""
        risk_scores = {"low": 1, "medium": 2, "high": 3}
        
        for sug in suggestions:
            sug["priority"] = risk_scores.get(sug["risk"], 2)
        
        return sorted(suggestions, key=lambda x: x["priority"])
    
    def get_code_modifications(self, suggestion_type: str) -> str:
        """Get code modification hints."""
        hints = {
            "lr": "Modify MATRIX_LR, EMBEDDING_LR, UNEMBEDDING_LR constants in train.py.",
            "schedule": "Modify get_lr_multiplier() function. WARMUP_RATIO and WARMDOWN_RATIO control schedule.",
            "weight_decay": "Modify WEIGHT_DECAY constant. Applied in Muon optimizer.",
            "momentum": "Modify ADAM_BETAS tuple and muon_momentum in get_muon_momentum().",
            "alternative": "Implement new optimizer class in train.py, replace MuonAdamW.",
        }
        return hints.get(suggestion_type, "See train.py optimizer setup section.")
