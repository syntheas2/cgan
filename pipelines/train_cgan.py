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
    mlflow.set_tag("mlflow.runName", f"{args.mlflow_run_name}_{timestamp_str}")
    # --- Add run description from config as a tag ---
    if hasattr(args, "mlflow_run_desc") and args.mlflow_run_desc:
        mlflow.set_tag("mlflow.runDesc", str(args.mlflow_run_desc))

    if not args.bestmodels_runid:
        args.bestmodels_runid = mlflow.active_run().info.run_id
    
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    discrete_condcolumns=set(['impact'])
    df_train, df_val, discrete_columns, one_hot_columns = load_data_step()
    X_val, y_val, data_transformer, data_sampler = prepare_data_step(
        args,
        discrete_columns=discrete_columns,
        one_hot_columns=one_hot_columns,
        discrete_condcolumns=discrete_condcolumns,  # Assuming 'impact' is the condition column
        df_train=df_train,
        df_val=df_val,
    )

    # Step 2: Train and evaluate the VAE model
    return train_evaluate_cgan_step(
        config=args,
        data_transformer=data_transformer,
        data_sampler=data_sampler,
        df_val=df_val,
        X_val=X_val,
        y_val=y_val
    )


if __name__ == "__main__":
    train_cigan_pipeline.with_options(
        # enable_cache=False  
    )()