import __init__ # noqa: F401
from zenml import pipeline
import mlflow  # Import the mlflow library
from datetime import datetime # For the run description
import torch
from steps.load_data import load_data_step
from steps.train_cgan_step import train_evaluate_cgan_step
from steps.prepare_data import prepare_data_step
from pipelines.train_cgan_args import CGANArgs


@pipeline
def train_cigan_pipeline():
    args = CGANArgs()
    # --- Set the MLflow run name using a tag ---
    now = datetime.now() # Use current time for uniqueness
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    mlflow.set_experiment(args.mlflow_experiment_name)
    mlflow.set_tag("mlflow.runName", f"{args.mlflow_experiment_name}_{timestamp_str}")
    if not args.bestmodels_runid:
        args.bestmodels_runid = mlflow.active_run().info.run_id
    
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    df_train, df_val, discrete_columns = load_data_step()
    X_val, y_val, data_transformer = prepare_data_step(
        discrete_columns=discrete_columns,
        df_train=df_train,
        df_val=df_val,
    )

    # Step 2: Train and evaluate the VAE model
    return train_evaluate_cgan_step(
        config=args,
        discrete_columns=discrete_columns,
        data_transformer=data_transformer,
        df_val=df_val,
        df_train=df_train,
        X_val=X_val,
        y_val=y_val
    )


if __name__ == "__main__":
    train_cigan_pipeline.with_options(
        # enable_cache=False  
    )()