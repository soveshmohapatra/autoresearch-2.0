"""
Analyst Agent - Specialized in experiment analysis and decision making.
Analyzes results, decides keep/discard, generates insights.
"""

from typing import List, Dict, Any, Optional


class AnalystAgent:
    """Agent specialized in experiment analysis."""
    
    def __init__(self, memory=None):
        self.memory = memory
        self.name = "AnalystAgent"
        self.specialties = [
            "result_analysis",
            "keep_discard_decision",
            "trend_detection",
            "insight_generation",
        ]
    
    def analyze_result(self, result: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single experiment result."""
        analysis = {
            "val_bpb": result.get("val_bpb", float('inf')),
            "baseline_bpb": baseline.get("val_bpb", result.get("val_bpb", 0)),
            "improvement": baseline.get("val_bpb", 0) - result.get("val_bpb", 0),
            "memory_change": result.get("memory_mb", 0) - baseline.get("memory_mb", 0),
            "decision": "pending",
            "confidence": 0.0,
            "reasoning": "",
        }
        
        # Calculate improvement
        bpb_diff = analysis["baseline_bpb"] - analysis["val_bpb"]
        
        # Decision logic
        if result.get("status") == "crash":
            analysis["decision"] = "discard"
            analysis["confidence"] = 1.0
            analysis["reasoning"] = "Experiment crashed - cannot evaluate"
        elif bpb_diff > 0.001:  # Improvement
            analysis["decision"] = "keep"
            analysis["confidence"] = min(0.9, 0.5 + bpb_diff * 10)
            analysis["reasoning"] = f"Improved by {bpb_diff:.6f} bpb"
            
            # Check memory tradeoff
            if analysis["memory_change"] > 1000:  # > 1GB increase
                analysis["reasoning"] += f" but uses {analysis['memory_change']/1024:.1f}GB more VRAM"
                analysis["confidence"] *= 0.9
        elif bpb_diff < -0.001:  # Degradation
            analysis["decision"] = "discard"
            analysis["confidence"] = min(0.9, 0.5 + abs(bpb_diff) * 10)
            analysis["reasoning"] = f"Degraded by {abs(bpb_diff):.6f} bpb"
        else:  # Neutral
            # Check if simpler (less memory)
            if analysis["memory_change"] < -100:
                analysis["decision"] = "keep"
                analysis["confidence"] = 0.7
                analysis["reasoning"] = f"Similar performance with {abs(analysis['memory_change']/1024):.1f}GB less VRAM"
            else:
                analysis["decision"] = "discard"
                analysis["confidence"] = 0.5
                analysis["reasoning"] = "No significant improvement"
        
        return analysis
    
    def get_experiment_summary(self, experiments: List[Dict]) -> str:
        """Generate a summary of multiple experiments."""
        if not experiments:
            return "No experiments to summarize."
        
        kept = [e for e in experiments if e.get("status") == "keep"]
        discarded = [e for e in experiments if e.get("status") == "discard"]
        crashed = [e for e in experiments if e.get("status") == "crash"]
        
        summary = f"""
=== Experiment Summary ===
Total: {len(experiments)}
Kept: {len(kept)} ({len(kept)/len(experiments)*100:.1f}%)
Discarded: {len(discarded)}
Crashed: {len(crashed)}

"""
        
        if kept:
            best = min(kept, key=lambda x: x.get("val_bpb", float('inf')))
            summary += f"Best val_bpb: {best['val_bpb']:.6f}\n"
            summary += f"Best experiment: {best.get('description', 'N/A')}\n"
        
        if len(kept) >= 2:
            first_bpb = kept[0].get("val_bpb", 0)
            last_bpb = kept[-1].get("val_bpb", 0)
            total_improvement = first_bpb - last_bpb
            summary += f"\nTotal improvement: {total_improvement:.6f} bpb\n"
        
        return summary
    
    def generate_insights(self, experiments: List[Dict]) -> List[str]:
        """Generate insights from experiment history."""
        insights = []
        
        if len(experiments) < 3:
            return ["Need more experiments for meaningful insights."]
        
        kept = [e for e in experiments if e.get("status") == "keep"]
        discarded = [e for e in experiments if e.get("status") == "discard"]
        
        # Analyze what works
        if len(kept) >= 3:
            # Check for trends in descriptions
            descriptions = [e.get("description", "").lower() for e in kept]
            
            lr_mentions = sum(1 for d in descriptions if "lr" in d or "learning rate" in d)
            if lr_mentions >= 2:
                insights.append("Learning rate modifications have been successful - continue tuning LR")
            
            depth_mentions = sum(1 for d in descriptions if "depth" in d or "layer" in d)
            if depth_mentions >= 2:
                insights.append("Model depth changes are working - explore depth further")
            
            batch_mentions = sum(1 for d in descriptions if "batch" in d)
            if batch_mentions >= 2:
                insights.append("Batch size tuning is productive - continue exploring")
        
        # Analyze failures
        if len(discarded) >= 5:
            discard_rate = len(discarded) / len(experiments)
            if discard_rate > 0.7:
                insights.append("High discard rate - consider more conservative changes or reset to best config")
        
        # Memory insights
        memory_changes = [e.get("memory_mb", 0) for e in kept]
        if memory_changes and max(memory_changes) - min(memory_changes) > 2000:
            insights.append("Significant memory variation in successful experiments - consider memory-efficiency tradeoffs")
        
        return insights
    
    def recommend_next_direction(self, experiments: List[Dict]) -> str:
        """Recommend the next experimental direction."""
        if not experiments:
            return "Start with baseline experiment."
        
        kept = [e for e in experiments if e.get("status") == "keep"]
        recent = experiments[-5:] if len(experiments) >= 5 else experiments
        
        # Check recent trend
        recent_kept = [e for e in recent if e.get("status") == "keep"]
        
        if len(recent_kept) >= 3:
            # Check if improving
            bpb_values = [e.get("val_bpb", 0) for e in recent_kept]
            if bpb_values[0] > bpb_values[-1] + 0.01:
                return "Recent trend is positive - continue current strategy with refinements"
            elif bpb_values[0] < bpb_values[-1] - 0.01:
                return "Recent trend is negative - consider resetting to best configuration and trying orthogonal direction"
        
        # Check if stuck
        if len(kept) >= 10:
            last_5_bpb = [e.get("val_bpb", 0) for e in kept[-5:]]
            if max(last_5_bpb) - min(last_5_bpb) < 0.005:
                return "Progress has plateaued - try more radical architectural changes"
        
        # Default recommendation
        if len(kept) < 5:
            return "Still exploring baseline - continue with incremental improvements"
        else:
            return "Good exploration so far - balance exploitation (refining successful changes) with exploration (new directions)"
