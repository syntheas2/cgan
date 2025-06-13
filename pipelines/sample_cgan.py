import __init__ # noqa: F401
from zenml import pipeline
import mlflow  # Import the mlflow library
from datetime import datetime # For the run description
import torch
from steps.load_data import load_data4sample_step
from steps.sample import sample_step
from steps.prepare_data import prepare_data_step
from pipelines.sample_cgan_args import CGANArgs


@pipeline
def sample_cigan_pipeline():
    args = CGANArgs()
    # --- Set the MLflow run name using a tag ---
    now = datetime.now() # Use current time for uniqueness
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    mlflow.set_experiment(args.mlflow_experiment_name)
    mlflow.set_tag("mlflow.runName", f"{args.mlflow_run_name}_{timestamp_str}")
    # --- Add run description from config as a tag ---
    if hasattr(args, "mlflow_run_desc") and args.mlflow_run_desc:
        mlflow.set_tag("mlflow.runDesc", str(args.mlflow_run_desc))
    
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'



    generator, transformer, discrete_column_category_prob_flatten, discrete_column_matrix_st, n_categories, orig_columns = load_data4sample_step(args)
    df_syn = sample_step(
        args,
        generator,
        transformer, discrete_column_category_prob_flatten, discrete_column_matrix_st, n_categories, orig_columns
    )

    return df_syn


if __name__ == "__main__":
    sample_cigan_pipeline.with_options(
        # enable_cache=False  
    )()