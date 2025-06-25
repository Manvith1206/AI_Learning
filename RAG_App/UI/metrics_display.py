import pandas as pd
from typing import Dict, Any, Callable
from UI.ui_components import UIComponents

class MetricsDisplay:
    """Component for displaying evaluation and performance metrics"""

    @staticmethod
    def display_evaluation_metrics(metrics: Dict[str, float]):
        """Display evaluation metrics in a formatted way"""
        import pandas as pd
        UIComponents.write("**Evaluation Metrics:**")

        if not isinstance(metrics, dict) or not metrics:
            UIComponents.display_info("No evaluation metrics to display.")
            return None

        metrics_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Score": list(metrics.values())
        })

        UIComponents.display_dataframe(metrics_df)

        # Calculate and display overall score safely
        overall_score = sum(metrics.values()) / len(metrics)
        UIComponents.write(f"Overall Score: {overall_score:.4f}")

        # Display bar chart
        UIComponents.display_bar_chart(metrics_df.set_index("Metric"))
        return metrics_df

    @staticmethod
    def display_pipeline_metrics(metrics: Dict[str, Any]):
        """Display pipeline performance metrics with robust checking."""
        UIComponents.write("**Pipeline Performance Metrics:**")

        for component_name, component_metrics in metrics.items():
            with UIComponents.create_expander(f"📊 {component_name} - Performance & Cost", expanded=False):
                if isinstance(component_metrics, (list, tuple)) and len(component_metrics) == 2:
                    cost, time_taken = component_metrics
                    col1, col2 = UIComponents.create_columns(2)
                    col1.metric("🕒 Time Taken", f"{time_taken:.4f}s")
                    col2.metric("💲 Estimated Cost", f"${cost:.6f}")
                else:
                    UIComponents.display_warning(f"Metrics for {component_name} are in an unexpected format.")

    @staticmethod
    def display_evaluation_section(on_evaluate_callback: Callable[[str], Dict[str, float]], ground_truth_default: str = ""):
        """Display the evaluation section with input and button."""
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
                        if metrics:
                            MetricsDisplay.display_evaluation_metrics(metrics)
                        else:
                            UIComponents.display_warning("Evaluation did not return any metrics.")
                    except Exception as e:
                        UIComponents.display_error(f"Error during evaluation: {str(e)}")
            else:
                UIComponents.display_warning("Please enter a ground truth answer for evaluation.")
