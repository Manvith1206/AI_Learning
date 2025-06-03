import streamlit as st
import pandas as pd
from typing import Dict

class MetricsDisplay:
    @staticmethod
    def display_evaluation_metrics(metrics: Dict[str, float]):
        """Display evaluation metrics in a formatted way"""
        st.write("**Evaluation Metrics:**")
        if metrics:
            metrics_df = pd.DataFrame({
                "Metric": list(metrics.keys()),
                "Score": list(metrics.values())
            })
            st.dataframe(metrics_df)
            overall_score = sum(metrics.values()) / len(metrics)
            st.write(f"Overall Score: {overall_score:.4f}")
            st.bar_chart(metrics_df.set_index("Metric"))
            return metrics_df
        else:
            st.info("No evaluation metrics available to display.")
            return pd.DataFrame()

    @staticmethod
    def display_pipeline_step_metrics(key: str, metrics: Dict[str, float], pipeline_instance):
        """
        Display metrics for a single pipeline step.
        """
        from rag_modular.Common import RAG_Constants as constants 

        cost = metrics.get('cost', 0.0)
        time_taken = metrics.get('time_taken', 0.0)
        with st.expander(f"📊 {key} - Performance & Cost", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("🕒 Time Taken", f"{time_taken:.4f}s" if isinstance(time_taken, float) else time_taken)
            col2.metric("💲 Estimated Cost", f"${cost:.6f}" if isinstance(cost, float) else cost)
            
            if pipeline_instance and pipeline_instance.config_manager:
                cfg = pipeline_instance.config_manager.get_config(key)
                if cfg and isinstance(cfg, dict): 
                    name = cfg.get(constants.CONFIG_TYPE_PARAM, "N/A")
                    col3.markdown(f"**Type:**\n`{name}`")
                else:
                    col3.markdown(f"**Type:**\n`Config not found`")
            else:
                col3.markdown(f"**Type:**\n`Pipeline not initialized`")
