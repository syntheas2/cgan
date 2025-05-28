import __init__
from typing import Annotated, Dict, Any
from zenml import step, get_step_context # Keep ZenML decorator and context
from zenml.logger import get_logger # ZenML's logger
import mlflow
import mlflow.pytorch
from pipelines.train_cgan_args import CGANArgs

# Assuming these are correctly importable from your project structure
from cgan.main import CGAN

logger = get_logger(__name__) # Use ZenML's logger for the step

ReturnType = Annotated[Dict[str, Any], "model_dict"] # Define return type for clarity

@step(enable_cache=False) # ZenML step decorator
def train_evaluate_cgan_step(
    config: CGANArgs, # Your configuration class
    data_transformer=None, data_sampler=None, df_val=None,
     X_val=None, y_val=None
) -> ReturnType: # Returns path to the best model's state_dict (ZenML output)

    current_run = mlflow.active_run()
    if not current_run:
        logger.info("No active MLflow run found initially, will rely on autolog or subsequent calls to create/get it.")
    run_id = current_run.info.run_id if current_run else None
    mlflow.pytorch.autolog(
        log_models=True, 
        checkpoint=True, 
        disable_for_unsupported_versions=True,
        registered_model_name=None
    )
    logger.info(f"Starting CIGAN training on device: {config.device}. MLflow autologging enabled.")
    # Log all config parameters manually
    mlflow.log_params(config.model_dump())

    generator_dim = [int(x) for x in config.generator_dim.split(',')]
    discriminator_dim = [int(x) for x in config.discriminator_dim.split(',')]
    model = CGAN(
        embedding_dim=config.embedding_dim,
        generator_dim=generator_dim,
        discriminator_dim=discriminator_dim,
        generator_lr=config.generator_lr,
        generator_decay=config.generator_decay,
        discriminator_lr=config.discriminator_lr,
        discriminator_decay=config.discriminator_decay,
        batch_size=config.batch_size,
        epochs=config.epochs,
        verbose=True,
        pac=config.pac,
        device=config.device,
        config=config,
    )
    best_model_data_to_save = model.fit(run_id, data_transformer, data_sampler, df_val, X_val, y_val, config=config)



    return best_model_data_to_save



    