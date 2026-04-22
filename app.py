import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import os
import io
import time

from src.data_processor import DataProcessor
from src.estatistic_analysis import run_eda
from src.model_pipeline import ModelPipeline
from src.visualizer import plot_tsne, plot_combined_roc_curves, plot_metrics_evolution

st.set_page_config(
    page_title="GeneSight.AI", 
    page_icon=":material/biotech:", 
    layout="wide"
)

def main():
    st.title("GeneSight.AI")
    st.markdown("---")

    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"

    st.sidebar.header("Data Ingestion")
    uploaded_file = st.sidebar.file_uploader("Upload the pickle dataset (.p)", type=["p"])
    
    if uploaded_file is not None:
        temp_path = Path("data/temp_upload.p")
        temp_path.parent.mkdir(exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.sidebar.button("Run Full Pipeline"):
            with st.status("Executing ML Pipeline...", expanded=True) as status:
                
                st.write("Running Statistical Analysis...")
                run_eda(str(temp_path), figures_dir)
                
                st.write("Normalizing Embeddings...")
                processor = DataProcessor(file_path=str(temp_path))
                raw_data = processor.load_data()
                df = processor.flatten_data(raw_data)
                df = processor.treat_embeddings(df)

                st.write("Generating t-SNE Clusters...")
                plot_tsne(df, save_dir=figures_dir)
                
                st.write("Training KNN Models (10-fold CV)...")
                pipeline = ModelPipeline(df)
                results_df = pipeline.run_cross_validation(max_k=15, n_splits=10)
                
                st.write("Exporting Metrics...")
                best_euclidean = pipeline.find_optimal_k(results_df[results_df['distance_metric'] == 'euclidean'])
                best_cosine = pipeline.find_optimal_k(results_df[results_df['distance_metric'] == 'cosine'])
                
                plot_combined_roc_curves(best_euclidean, best_cosine, pipeline.num_classes, figures_dir)
                
                csv_path = tables_dir / 'knn_evaluation_metrics.csv'
                results_df.drop(columns=['y_true_all_folds', 'y_prob_all_folds']).to_csv(csv_path, index=False)
                plot_metrics_evolution(csv_path, figures_dir)
                
                status.update(label="Pipeline Completed!", state="complete", expanded=False)
                if temp_path.exists():
                    os.remove(temp_path)
                
            
            st.success("Analysis ready! Explore the tabs below.")

    else:
        st.info("Please upload the genetic embeddings file (.p) in the sidebar to begin.")

    st.markdown("### Pipeline Results")
    tab1, tab2, tab3 = st.tabs(["Exploratory Data", "Spatial Clusters", "Performance Results"])
    
    with tab1:
        col1, col2 = st.columns(2)
        if (figures_dir / "eda_syndrome_distribution.png").exists():
            col1.image(str(figures_dir / "eda_syndrome_distribution.png"), caption="Class Distribution")
        if (figures_dir / "eda_outliers_detection.png").exists():
            col2.image(str(figures_dir / "eda_outliers_detection.png"), caption="Outlier Detection")
        elif uploaded_file is None:
            st.write("Upload data and run the pipeline to see the Exploratory Data Analysis.")

    with tab2:
        if (figures_dir / "tsne_clusters.jpg").exists():
            st.image(str(figures_dir / "tsne_clusters.jpg"), use_container_width=True, caption="t-SNE Embedding Map")
        elif uploaded_file is None:
            st.write("Upload data and run the pipeline to see the Clusters.")

    with tab3:
        csv_path = tables_dir / "knn_evaluation_metrics.csv"
        if csv_path.exists():
            df_metrics = pd.read_csv(csv_path)
            st.dataframe(df_metrics, use_container_width=True)
            
            col3, col4 = st.columns(2)
            if (figures_dir / "metrics_evolution.png").exists():
                col3.image(str(figures_dir / "metrics_evolution.png"), caption="K-Selection Curves")
            if (figures_dir / "roc_curves_comparison.png").exists():
                col4.image(str(figures_dir / "roc_curves_comparison.png"), caption="ROC Curves Comparison")
        elif uploaded_file is None:
            st.write("Upload data and run the pipeline to see the Performance Metrics.")

if __name__ == "__main__":
    main()