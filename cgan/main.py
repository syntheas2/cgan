"""CTGAN module."""

import warnings

import numpy as np
import pandas as pd
import torch
from typing import Annotated, Dict, Any
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from tqdm import tqdm
from cgan.data_sampler import DataSampler
from cgan.data_transformer import DataTransformer
from cgan.errors import InvalidDataError
from cgan.synthesizers.base import BaseSynthesizer, random_state
from cgan.data import get_sample_conditions
from cgan.utils import split_num_cat_target, recover_data
from cgan.utils_evaluation import transform_val_data
from pythelpers.ml.mlflow import load_latest_checkpoint, rotate_checkpoints, rotate_bestmodels
from pythelpers.ml.nn import StepLRScheduler
from pythelpers.ml.mlflow_log import log_ur
from pathlib import Path
import mlflow
import tempfile # For temporary file handling
import pickle
from zenml.logger import get_logger # ZenML's logger
from pipelines.train_cgan_args import CGANArgs
logger = get_logger(__name__) # Use ZenML's logger for the step

import torch.serialization
from cgan.utils_evaluation import evaluate_samples, evaluate_linearsvc_performance, get_x_y, get_val_data

class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)



    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


class CGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
        config: CGANArgs | None =None,
        device=None,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None

        self.best_linearsvc_acc = 0

        self.config: CGANArgs = config

    def load_fromckp_4sampling(self, load_checkpoint, discrete_columns, data):
        # load ckpt

        if self._verbose:
            print(f"Loading checkpoint from {load_checkpoint}")
        try:
            self._transformer = DataTransformer()
            self._transformer.fit(data, discrete_columns)

            data = self._transformer.transform(data)

            self._data_sampler = DataSampler(
                data, self._transformer.output_info_list, self._log_frequency
            )

            data_dim = self._transformer.output_dimensions
            self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
            ).to(self._device)

            # trusting the source -> weights_only=True
            checkpoint = torch.load(load_checkpoint, map_location=self._device, weights_only=False)
            # Load model states
            self._generator.load_state_dict(checkpoint['generator_state_dict'])
            if self._verbose:
                print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")

        except Exception as e:
            if self._verbose:
                print(f"Error loading checkpoint: {e}")



    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]



    @random_state
    def fit(self, run_id=None, train_data = None, data_transformer=None, df_val=None,  X_val=None,y_val=None, discrete_columns=(), discrete_condcolumns=(), config=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.X_val, self.y_val = X_val, y_val
        self.df_val = df_val

        epochs = config.epochs

        self._transformer = data_transformer

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, 
            self._transformer.output_info_list, 
            self._log_frequency,
            discrete_columns=discrete_columns,
            cond_column_names=discrete_condcolumns
        )

        self.data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, self.data_dim
        ).to(self._device)

        discriminator = Discriminator(
            self.data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        
        # Log model description
        logger.info(f"GAN Training - Generator LR: {optimizerG.param_groups[0]['lr']}, Discriminator LR: {optimizerD.param_groups[0]['lr']}")

        # load checkpoint if provided
        start_epoch = 0
        generator_loss, discriminator_loss = None, None
        if config.load_from_checkpoint:
            ckp_run_id = config.load_ckp_from_run_id
            if ckp_run_id:
                logger.info(f"Attempting to load checkpoint from MLflow run '{ckp_run_id}', subdir '{config.manual_checkpoint_subdir}'...")
                checkpoint = load_latest_checkpoint(
                    run_id=ckp_run_id,
                    artifact_subdir=config.manual_checkpoint_subdir,
                    device=config.device
                )
                if checkpoint:
                    self._generator.load_state_dict(checkpoint['generator_state_dict'])
                    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
                    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

                    # Set starting epoch
                    start_epoch = checkpoint['epoch']
                    if self._verbose:
                        print(f"Successfully loaded checkpoint from epoch {start_epoch}")
                        print(f"Generator loss: {checkpoint['generator_loss']}, Discriminator loss: {checkpoint['discriminator_loss']}")

                    # Log the loaded checkpoint losses
                    mlflow.log_metric('g_loss', checkpoint['generator_loss'], step=start_epoch)
                    mlflow.log_metric('d_loss', checkpoint['discriminator_loss'], step=start_epoch)

                    generator_loss, discriminator_loss = checkpoint['generator_loss'], checkpoint['discriminator_loss']
                    optimizerG.param_groups[0]['lr'], optimizerD.param_groups[0]['lr'] = checkpoint['generator_lr'], checkpoint['discriminator_lr']
                    self.best_linearsvc_acc = checkpoint['best_linearsvc_acc']

                    # Model, optimizer, states are loaded by the helper
                    logger.info(f"Successfully loaded checkpoint. Resuming training from epoch {start_epoch}.")
                else:
                    logger.info("No suitable checkpoint found. Starting training from scratch.")
            else:
                logger.warning("MLflow run ID not available, cannot attempt to load checkpoint.")


        # Initialize schedulers
        scheduler_G = StepLRScheduler(
            initial_lr=optimizerG.param_groups[0]['lr'], step_size=80, gamma=0.8, min_lr=1e-5
        )
        
        scheduler_D = StepLRScheduler(
            initial_lr=optimizerD.param_groups[0]['lr'], step_size=80, gamma=0.8, min_lr=1e-5
        )

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(start_epoch, epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))
        
        # steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        steps_per_epoch = 4

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1
        gd_performancmeasure = None
        
        g_stepcount = 0
        d_stepcount = 0

        for i in epoch_iterator:
            # Bestimme die Anzahl der Trainingsschritte basierend auf aktueller Performance
            d_steps, g_steps = self.determine_training_steps(gd_performancmeasure)
            # Update learning rates using scheduler at the beginning of each epoch
            optimizerG.param_groups[0]['lr'] = scheduler_G.step(i)
            optimizerD.param_groups[0]['lr'] = scheduler_D.step(i)

            # Log learning rates and steps at the beginning of each epoch
            mlflow.log_metric('d_steps', d_steps, step=i)
            mlflow.log_metric('g_steps', g_steps, step=i)
            mlflow.log_metric('g_learning_rate', optimizerG.param_groups[0]['lr'], step=i)
            mlflow.log_metric('d_learning_rate', optimizerD.param_groups[0]['lr'], step=i)
                
            # These variables will track the losses for the current epoch
            generator_losses = []
            generator_wassersteinlosses = []
            generator_crossentropylosses = []
            discriminator_losses = []

            for e_step in range(steps_per_epoch):
                for n in range(d_steps):
                    #  The first strategy is to sample input random noise from a Gaussian
                    #  Mixture Model (GMM) for the generator. This GMM is a probability
                    #  density estimator for positive samples. In a traditional GAN, the input
                    #  noise of the generator is governed by a normal or uniform distributed
                    #  without considering any prior information
                    fakez = torch.normal(mean=mean, std=std)
                    # fakez_gmm = self.generate_gmm_noise(self._batch_size, self._embedding_dim)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    discriminator_losses.append(loss_d.detach().cpu().item())

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    log_ur(discriminator,  d_stepcount, lr=optimizerD.param_groups[0]['lr'], model_prefix='d')
                    optimizerD.step()

                    d_stepcount += 1

                for n in range(g_steps):
                    fakez = torch.normal(mean=mean, std=std)
                    condvec = self._data_sampler.sample_condvec(self._batch_size)

                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if c1 is not None:
                        y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                    else:
                        y_fake = discriminator(fakeact)

                    if condvec is None:
                        cross_entropy = 0
                    else:
                        cross_entropy = self._cond_loss(fake, c1, m1)

                    loss_g_wasserstein = -torch.mean(y_fake)
                    loss_g = loss_g_wasserstein + cross_entropy
                    generator_losses.append(loss_g.detach().cpu().item())
                    generator_wassersteinlosses.append(loss_g_wasserstein.detach().cpu().item())
                    generator_crossentropylosses.append(cross_entropy.detach().cpu().item())

                    optimizerG.zero_grad(set_to_none=False)
                    loss_g.backward()
                    log_ur(self._generator, g_stepcount, lr=optimizerG.param_groups[0]['lr'], model_prefix='g')
                    optimizerG.step()

                    g_stepcount += 1

            # Compute average losses for the epoch
            generator_loss = np.mean(generator_losses) if generator_losses else 0
            generator_wassersteinloss = np.mean(generator_wassersteinlosses) if generator_wassersteinlosses else 0
            generator_crossentropyloss = np.mean(generator_crossentropylosses) if generator_crossentropylosses else 0
            discriminator_loss = np.mean(discriminator_losses) if discriminator_losses else 0
            
            # Log the epoch average losses using MLflow
            mlflow.log_metric('g_loss', generator_loss, step=i)
            mlflow.log_metric('g_loss_wasserstein', generator_wassersteinloss, step=i)
            mlflow.log_metric('g_loss_crossentropy', generator_crossentropyloss, step=i)
            mlflow.log_metric('d_loss', discriminator_loss, step=i)

            # Berechne das gd unterschied -> distance between Wasserstein loss and discriminator loss
            d_nachteil = int(discriminator_loss > 0) * (1 + discriminator_loss)
            g_nachteil = int(generator_loss > 0) * (1 + generator_loss)
            gd_performancmeasure = g_nachteil if g_nachteil != 0 and d_nachteil == 0 else 0
            gd_performancmeasure = -d_nachteil if d_nachteil != 0 and g_nachteil == 0 else gd_performancmeasure
            mlflow.log_metric('g_d_performancmeasure', gd_performancmeasure, step=i)

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )

            if (i + 1) % 5 == 0:
                # Evaluate the model
                self.evaluate(i)

            save_this_epoch = (i % config.checkpoint_save_interval == 0) or \
                            (i == config.epochs - 1)
            if save_this_epoch and run_id:
                checkpoint_state: Dict[str, Any] =  {
                    'epoch': i + 1,
                    'generator_state_dict': self._generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'generator_loss': generator_loss,
                    'discriminator_loss': discriminator_loss,
                    'generator_lr': optimizerG.param_groups[0]['lr'],
                    'discriminator_lr': optimizerD.param_groups[0]['lr'],
                    'best_linearsvc_acc': self.best_linearsvc_acc,
                }
                ckpt_filename = f"model_checkpoint_epoch_{i:04d}.pt" # Padded epoch for sorting
                with tempfile.TemporaryDirectory() as tmpdir:
                    local_tmp_checkpoint_path = Path(tmpdir) / ckpt_filename
                    torch.save(checkpoint_state, local_tmp_checkpoint_path)
                    
                    mlflow.log_artifact(
                        str(local_tmp_checkpoint_path), 
                        artifact_path=f"{config.manual_checkpoint_subdir}" 
                    )
                logger.info(f"Saved manual checkpoint to MLflow: {config.manual_checkpoint_subdir}")
                # --- Checkpoint Rotation ---
                rotate_checkpoints(
                    run_id=run_id,
                    artifact_subdir=config.manual_checkpoint_subdir,
                    max_checkpoints=config.max_checkpoints_to_keep
                )
                bestmodel_filename = f"model_acc_{self.best_linearsvc_acc:.3f}.pt" # Padded epoch for sorting
                with tempfile.TemporaryDirectory() as tmpdir:
                    local_tmp_checkpoint_path = Path(tmpdir) / bestmodel_filename
                    torch.save(checkpoint_state, local_tmp_checkpoint_path)
                    
                    mlflow.log_artifact(
                        str(local_tmp_checkpoint_path), 
                        artifact_path=f"{config.manual_bestmodel_subdir}",
                        run_id=config.bestmodels_runid
                    )
                rotate_bestmodels(
                    run_id=config.bestmodels_runid,
                    metric='acc',
                    maximize=True,
                    artifact_subdir=config.manual_bestmodel_subdir,
                    max=config.max_bestmodel_to_keep,
                )
                logger.info(f"Saved manual best model to MLflow: {config.manual_bestmodel_subdir}")

    def evaluate(self, epoch):
        """Evaluate the model on the test data."""
        # generate samples
        generated_data = self.sample(10000)

        # Wiederherstellung der ursprünglichen Datenstruktur
        generated_data = generated_data[self.df_val.columns]

        # evaluate samples
        evaluate_samples(
            generated_data, self.df_val, epoch
        )
        

        # evaluate linear SVC performance
        X_train, y_train = get_x_y(generated_data)
        acc, _ = evaluate_linearsvc_performance(X_train, y_train, self.X_val, self.y_val, epoch)
        if acc > self.best_linearsvc_acc:
            self.best_linearsvc_acc = acc


    def generate_gmm_noise(self, batch_size, embedding_dim):
        """Generiert Noise-Samples aus dem GMM-Modell"""
        if not hasattr(self, 'gmm'):
            # Fallback auf normales Rauschen, wenn kein GMM verfügbar
            if self._verbose:
                print("Warnung: Kein GMM-Modell geladen, verwende Normalverteilung.")
            mean = torch.zeros(batch_size, embedding_dim, device=self._device)
            std = mean + 1
            return torch.normal(mean=mean, std=std)
        
        # Samples aus dem GMM ziehen
        noise_samples, _ = self.gmm.sample(batch_size)
        
        # Falls die Dimensionalität des GMM nicht mit embedding_dim übereinstimmt,
        # müssen wir anpassen
        if noise_samples.shape[1] != embedding_dim:
            if noise_samples.shape[1] < embedding_dim:
                # Mit zufälligen Werten auffüllen
                padding = np.random.normal(0, 1, (batch_size, embedding_dim - noise_samples.shape[1]))
                noise_samples = np.hstack([noise_samples, padding])
            else:
                # Dimensionen reduzieren
                noise_samples = noise_samples[:, :embedding_dim]
        
        # Zu Torch Tensor konvertieren und auf das entsprechende Device verschieben
        return torch.tensor(noise_samples, dtype=torch.float32).to(self._device)
    

    def adjust_learning_rates(self, g_lr, d_lr, g_loss, d_loss, min_lr=1e-6, max_lr=1e-3):
        ratio = abs(g_loss) / abs(d_loss)
        
        # Zielbereich für das Verhältnis (z.B. 0.7 bis 1.3)
        target_min, target_max = 0.7, 1.3
        
        if ratio < target_min:  # Generator zu schwach
            g_lr *= 1.1  # Erhöhe Generator-Lernrate
            d_lr *= 0.9  # Reduziere Diskriminator-Lernrate
        elif ratio > target_max:  # Generator zu stark
            g_lr *= 0.9  # Reduziere Generator-Lernrate
            d_lr *= 1.1  # Erhöhe Diskriminator-Lernrate
            
        # Grenzen einhalten
        g_lr = min(max(g_lr, min_lr), max_lr)
        d_lr = min(max(d_lr, min_lr), max_lr)
        
        return g_lr, d_lr    
    
    def determine_training_steps(self, gd_performanceratio=None, base_steps=1):
        if gd_performanceratio is None:
            return base_steps, base_steps
        
        # Wenn Diskriminator Schwierigkeiten hat (Generator zu stark)
        if gd_performanceratio < -0.5:
            d_steps = base_steps + 1  # Mehr Schritte für den Diskriminator
            g_steps = base_steps
        # Wenn Generator Schwierigkeiten hat (Diskriminator zu stark)
        elif gd_performanceratio > 0.5:
            d_steps = base_steps
            g_steps = base_steps + 1  # Ein Extra-Schritt für den Generator
        else:
            d_steps = base_steps
            g_steps = base_steps
            
        return d_steps, g_steps

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            # fakez_gmm = self.generate_gmm_noise(self._batch_size, self._embedding_dim)
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)

    def load_gmm(self, gmm_path):
        """Lädt ein vortrainiertes GMM-Modell"""
        try:
            with open(gmm_path, 'rb') as f:
                model_info = pickle.load(f)
            
            self.gmm = model_info['gmm']
            self.gmm_k = model_info['k']
            
            if self._verbose:
                print(f"GMM-Modell mit {self.gmm_k} Komponenten erfolgreich geladen.")
            
            return True
        except Exception as e:
            if self._verbose:
                print(f"Fehler beim Laden des GMM-Modells: {e}")
            return False

    def generate_gmm_noise(self, batch_size, embedding_dim):
        """Generiert Noise-Samples aus dem GMM-Modell"""
        if not hasattr(self, 'gmm'):
            # Fallback auf normales Rauschen, wenn kein GMM verfügbar
            if self._verbose:
                print("Warnung: Kein GMM-Modell geladen, verwende Normalverteilung.")
            mean = torch.zeros(batch_size, embedding_dim, device=self._device)
            std = mean + 1
            return torch.normal(mean=mean, std=std)
        
        # Samples aus dem GMM ziehen
        noise_samples, _ = self.gmm.sample(batch_size)
        
        # Falls die Dimensionalität des GMM nicht mit embedding_dim übereinstimmt,
        # müssen wir anpassen
        if noise_samples.shape[1] != embedding_dim:
            if noise_samples.shape[1] < embedding_dim:
                # Mit zufälligen Werten auffüllen
                padding = np.random.normal(0, 1, (batch_size, embedding_dim - noise_samples.shape[1]))
                noise_samples = np.hstack([noise_samples, padding])
            else:
                # Dimensionen reduzieren
                noise_samples = noise_samples[:, :embedding_dim]
        
        # Zu Torch Tensor konvertieren und auf das entsprechende Device verschieben
        return torch.tensor(noise_samples, dtype=torch.float32).to(self._device)
