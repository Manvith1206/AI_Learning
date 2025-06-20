import pandas as pd
from typing import Dict, List, Any
from UI.ui_components import UIComponents


class MetricsDisplay:
    """Component for displaying evaluation and performance metrics"""
    
    @staticmethod
    def display_evaluation_metrics(metrics: Dict[str, float]):
        """Display evaluation metrics in a formatted way"""
        UIComponents.write("**Evaluation Metrics:**")
        
        metrics_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Score": list(metrics.values())
        })
        
        UIComponents.display_dataframe(metrics_df)
        
        # Calculate overall score
        overall_score = sum(metrics.values()) / len(metrics) if metrics else 0
        UIComponents.write(f"Overall Score: {overall_score:.4f}")
        
        # Display bar chart
        UIComponents.display_bar_chart(metrics_df.set_index("Metric"))
        
        return metrics_df
    
    @staticmethod
    def display_pipeline_metrics(metrics: Dict[str, Any]):
        """Display pipeline performance metrics"""
        UIComponents.write("**Pipeline Performance Metrics:**")
        
        for component_name, component_metrics in metrics.items():
            with UIComponents.create_expander(f"📊 {component_name} - Performance & Cost", expanded=False):
                cost, time_taken = component_metrics
                
                col1, col2 = UIComponents.create_columns(2)
                col1.metric("🕒 Time Taken", time_taken)
                col2.metric("💲 Estimated Cost", cost)
    
    @staticmethod
    def display_evaluation_section(on_evaluate_callback, ground_truth_default=""):
        """Display the evaluation section with input and button"""
        UIComponents.create_subheader_UI("Evaluation")
        
        ground_truth = UIComponents.create_text_area(
            "Ground Truth Answer", 
            value=ground_truth_default,
            help="Enter the correct answer to evaluate the RAG pipeline's response"
        )
        
        if UIComponents.create_button("Evaluate Last Query"):
            if ground_truth:
                with UIComponents.display_spinner("Evaluating..."):
                    try:
                        metrics = on_evaluate_callback(ground_truth)
                        MetricsDisplay.display_evaluation_metrics(metrics)
                    except Exception as e:
                        UIComponents.display_error(f"Error during evaluation: {str(e)}")
            else:
                UIComponents.display_warning("Please enter a ground truth answer for evaluation.")
