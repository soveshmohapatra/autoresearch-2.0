"""
Architecture Agent - Specialized in model architecture changes.
Suggests modifications to: depth, width, attention, activations, normalization.
"""

from typing import List, Dict, Any, Optional


class ArchitectureAgent:
    """Agent specialized in neural architecture experimentation."""
    
    def __init__(self, memory=None):
        self.memory = memory
        self.name = "ArchitectureAgent"
        self.specialties = [
            "model_depth",
            "model_width", 
            "attention_mechanisms",
            "activation_functions",
            "normalization_layers",
            "residual_connections",
        ]
    
    def get_suggestions(self, current_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate architecture modification suggestions."""
        suggestions = []
        
        # Get context from memory
        context = self.memory.get_context_for_agent(self.name) if self.memory else ""
        
        # Depth modifications
        current_depth = current_config.get("depth", 8)
        suggestions.extend([
            {
                "type": "depth",
                "change": f"Increase depth from {current_depth} to {current_depth + 2}",
                "rationale": "Deeper models can learn more complex representations",
                "risk": "medium",
                "expected_impact": "0.01-0.05 bpb improvement if model is underparameterized",
            },
            {
                "type": "depth", 
                "change": f"Decrease depth from {current_depth} to {max(2, current_depth - 2)}",
                "rationale": "Shallower models train faster, may generalize better with limited data",
                "risk": "low",
                "expected_impact": "Faster training, possible slight bpb increase",
            },
        ])
        
        # Width modifications
        current_aspect = current_config.get("aspect_ratio", 64)
        suggestions.extend([
            {
                "type": "width",
                "change": f"Increase aspect ratio from {current_aspect} to {int(current_aspect * 1.5)}",
                "rationale": "Wider models have more capacity per layer",
                "risk": "medium",
                "expected_impact": "0.01-0.03 bpb improvement",
            },
        ])
        
        # Attention modifications
        current_window = current_config.get("window_pattern", "SSSL")
        suggestions.extend([
            {
                "type": "attention",
                "change": f"Change window pattern from '{current_window}' to 'LLLL' (full attention)",
                "rationale": "Full attention allows all tokens to attend to each other",
                "risk": "low",
                "expected_impact": "Possible improvement for tasks requiring long-range dependencies",
            },
            {
                "type": "attention",
                "change": f"Change window pattern from '{current_window}' to 'SSSS' (half attention)",
                "rationale": "Reduced attention may improve generalization and speed",
                "risk": "medium",
                "expected_impact": "Faster training, possible slight bpb increase",
            },
            {
                "type": "attention",
                "change": "Implement Grouped Query Attention (GQA)",
                "rationale": "GQA reduces KV cache, improves efficiency",
                "risk": "high",
                "expected_impact": "Similar bpb with better efficiency",
            },
        ])
        
        # Activation modifications
        suggestions.extend([
            {
                "type": "activation",
                "change": "Replace ReLU² with SwiGLU activation",
                "rationale": "SwiGLU is used in LLaMA, shows better performance",
                "risk": "medium",
                "expected_impact": "0.01-0.02 bpb improvement",
            },
            {
                "type": "activation",
                "change": "Replace ReLU² with GeLU activation",
                "rationale": "GeLU is smoother, used in many successful models",
                "risk": "low",
                "expected_impact": "Similar or slight improvement",
            },
        ])
        
        # Normalization modifications
        suggestions.extend([
            {
                "type": "normalization",
                "change": "Replace RMSNorm with LayerNorm",
                "rationale": "LayerNorm is more stable, widely used",
                "risk": "low",
                "expected_impact": "Similar bpb, possibly more stable training",
            },
            {
                "type": "normalization",
                "change": "Try Pre-Norm architecture (norm before attention/MLP)",
                "rationale": "Pre-norm improves training stability for deep models",
                "risk": "medium",
                "expected_impact": "Better convergence for deep models",
            },
        ])
        
        # Residual connection modifications
        suggestions.extend([
            {
                "type": "residual",
                "change": "Add residual scaling factors (trainable lambda)",
                "rationale": "Learnable residual weights can optimize information flow",
                "risk": "medium",
                "expected_impact": "0.005-0.02 bpb improvement",
            },
        ])
        
        return self._rank_suggestions(suggestions, context)
    
    def _rank_suggestions(self, suggestions: List[Dict], context: str) -> List[Dict]:
        """Rank suggestions based on risk and expected impact."""
        # Simple ranking: prefer low risk, high impact
        risk_scores = {"low": 1, "medium": 2, "high": 3}
        
        for sug in suggestions:
            sug["priority"] = risk_scores.get(sug["risk"], 2)
        
        # Sort by priority (lower is better)
        return sorted(suggestions, key=lambda x: x["priority"])
    
    def get_code_modifications(self, suggestion_type: str) -> str:
        """Get code modification hints for a suggestion type."""
        hints = {
            "depth": "Modify DEPTH constant in train.py. Remember to adjust aspect_ratio if needed.",
            "width": "Modify ASPECT_RATIO constant. Model dim = depth * aspect_ratio.",
            "attention": "Modify WINDOW_PATTERN. L=full attention, S=half context.",
            "activation": "Replace F.relu(x).square() in MLP.forward() with F.silu() for SwiGLU.",
            "normalization": "Replace F.rms_norm with F.layer_norm in norm() function.",
        }
        return hints.get(suggestion_type, "See train.py for modification points.")
