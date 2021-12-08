"""Basic neural network trainer"""

import re
import os
import copy
import abc
import numpy as np

from matplotlib import pyplot as plt

import misc

from pytorchutils.globals import nn, torch, DEVICE


class BasicTrainer(metaclass=abc.ABCMeta):
    """Class for basic trainer"""

    def __init__(self, config, model, preprocessor):
        # When working with NN, usually a random initialion is used.
        # Store the random seed and actually use it in PyTorch
        # to be able to reproduce the initialization
        # (torch.manual_seed() for example).
        torch.manual_seed(config.get('random_seed', 1234))
        self.preprocessor = preprocessor
        self.config = copy.deepcopy(config)

        self.models_dir = self.config.get('models_dir', './')
        self.results_dir = self.config.get('results_dir', './')
        self.max_models_to_keep = self.config.get('max_to_keep', -1)

        # Remember current epoch for saving/resuming training
        self.current_epoch = 0
        # At least one epoch is conducted
        self.max_iter = config.get('max_iter', 1)
        self.learning_rate = config.get('learning_rate', -1)
        if self.learning_rate == -1:
            print(
                "Warning: No learning rate defined. "
                "This might be not beneficial for most model architectures, "
                "especially for those using backpropagation."
            )
        loss_label = config.get('loss', 'MSELoss') # MSE being default
        self.loss = getattr(nn, loss_label)()

        # Every trainer has one or an array of models.
        parameters = []
        if isinstance(model, (list, np.ndarray)):
            self.model = [m.to(DEVICE) for m in model]
            for idx, __ in enumerate(self.model):
                parameters += list(self.model[idx].parameters())
        else:
            self.model = model.to(DEVICE)
            parameters = list(self.model.parameters())
        self.optimizer = getattr(torch.optim, self.config.get('optimizer', 'Adam'))(
            parameters,
            lr=self.learning_rate,
            betas=self.config.get('optim_betas', (0.9, 0.999)),
            weight_decay=config.get('reg_lambda', 0.0)
        )

        self._get_batches_fn = None

        # Initialize specified layers using specified method
        # self.model.apply(self.init_weights)

        # Load previously saved state dicts to resume training
        epoch_idx = 0
        checkpoint_files = [
            filename for filename in os.listdir(self.models_dir) if filename.endswith('.pth.tar')
        ]
        for filename in checkpoint_files:
            index = int(
                re.search(
                    r'\d+', os.path.basename(filename).split('_')[-2]
                ).group()
            )
            if index > epoch_idx:
                epoch_idx = index

        if isinstance(self.model, (list, np.ndarray)):
            for idx, __ in enumerate(self.model):
                self.load_model(self.model[idx], epoch_idx, idx)
        else:
            self.load_model(self.model, epoch_idx)
        self.load_optimizer(epoch_idx)

    def init_weights(self, model):
        """Initialize model weights using specified method. Xavier initialization is the default"""
        if isinstance(model, self.config['init_layers']):
            getattr(nn.init, self.config.get('init', 'xavier_uniform_'))(model.weight)
            torch.nn.init.zeros_(model.bias)

    @misc.lazy_property
    def get_batches_fn(self):
        """Property for get_batches functions"""
        return self._get_batches_fn

    @get_batches_fn.setter
    def get_batches_fn(self, f_n):
        """Setter for get_batches functions property"""
        self._get_batches_fn = f_n

    def load_model(self, model, epoch_idx, model_idx=0):
        """Initialize the model"""

        checkpoint_file = '{}/{}_{}_epoch{}_checkpoint.pth.tar'.format(
            self.models_dir,
            type(model).__name__,
            model_idx,
            epoch_idx
        )
        if os.path.isfile(checkpoint_file):
            print("Loading model checkpoint from {}".format(checkpoint_file))
            state = torch.load(checkpoint_file)
            model.load_state_dict(state['state_dict'])

    def load_optimizer(self, epoch_idx):
        """Initialize the optimizer"""
        checkpoint_file = '{}/{}_epoch{}_checkpoint.pth.tar'.format(
            self.models_dir,
            type(self.optimizer).__name__,
            epoch_idx
        )
        if os.path.isfile(checkpoint_file):
            print("Loading optimizer checkpoint from {}".format(checkpoint_file))
            state = torch.load(checkpoint_file)
            self.optimizer.load_state_dict(state['optimizer'])
            self.current_epoch = state['epoch'] + 1

    @abc.abstractmethod
    def learn_from_epoch(self):
        """Separate the function to train per epoch and the function to train globally"""
        return

    def get_weights(self):
        """Template for retrieval of model weights"""

    def validate(self, epoch_idx):
        """The preprocessor has to provide a validate function"""
        try:
            return self.preprocessor.validate(
                self.evaluate,
                epoch_idx
            )
        except AttributeError as exc:
            print(
                "Error: No validate method provided by preprocessor "
                "although a validation was required. Error: {}".format(repr(exc))
            )

    @abc.abstractmethod
    def evaluate(self):
        """Function for prediction capabilities of the model"""
        return

    def train(self, validate_every=0, save_every=1):
        """This function is usually common to all models"""
        print("Start training...")
        start_epoch = self.current_epoch
        if validate_every > 0:
            errors = np.empty(int((self.max_iter - start_epoch) / validate_every))
            stds = np.empty(int((self.max_iter - start_epoch) / validate_every))
        for epoch_idx in range(start_epoch, self.max_iter):
            print("Epoch {}...".format(epoch_idx))
            if isinstance(self.model, (list, np.ndarray)):
                for idx, __ in enumerate(self.model):
                    self.model[idx].train()
            else:
                self.model.train()
            epoch_loss = self.learn_from_epoch()
            print("Epoch loss: {}".format(epoch_loss))
            # Negative number to not save during training
            if save_every > 0 and epoch_idx % save_every == 0:
                if isinstance(self.model, (list, np.ndarray)):
                    for idx, __ in enumerate(self.model):
                        self.save_model(
                            self.model[idx],
                            epoch_idx,
                            max_to_keep=self.max_models_to_keep,
                            model_idx=idx,
                        )
                else:
                    self.save_model(self.model, epoch_idx, max_to_keep=self.max_models_to_keep)
                self.save_optimizer(epoch_idx, max_to_keep=self.max_models_to_keep)
            if validate_every > 0 and epoch_idx % validate_every == 0:
                error, std = self.validate(epoch_idx)
                errors[int((epoch_idx - start_epoch) / validate_every)] = error
                stds[int((epoch_idx - start_epoch) / validate_every)] = std
                print("Validation error: {} % +- {} %".format(error, std))
            self.current_epoch += 1

        # Save error progression
        if validate_every > 0:
            np.save(
                f'{self.results_dir}/error_progression.npy',
                np.array([
                    np.arange(start_epoch, self.max_iter, validate_every),
                    errors,
                    stds
                ])
            )
            # Plot error progression
            __, axs = plt.subplots(1, 1)
            axs.plot(np.arange(start_epoch, self.max_iter, validate_every), errors)
            axs.fill_between(
                np.arange(start_epoch, self.max_iter, validate_every),
                errors - stds,
                errors + stds,
                alpha=.5
            )
            axs.set_xlabel("Epoch")
            axs.set_ylabel("Validation accuracy")
            plt.savefig(
                f'{self.results_dir}/accuracy_progression.png',
                format='png',
                dpi=600,
                bbox_inches='tight'
            )
            plt.close()

    def infer(self, *args):
        """Utilized trained model to do predictions"""
        try:
            self.preprocessor.infer(self.evaluate, args)
        except AttributeError as exc:
            print(
                "Error: No inference method provided by preprocessor. "
                "Error: {}".format(repr(exc))
            )

    def save_model(self, model, epoch_idx, max_to_keep=-1, model_idx=0):
        """This function is usually common to all models"""
        checkpoint_file = '{}/{}_{}_epoch{}_checkpoint.pth.tar'.format(
            self.models_dir,
            type(model).__name__,
            model_idx,
            epoch_idx
        )
        print("Saving model to {}".format(checkpoint_file))
        state = {
            'state_dict': model.state_dict(),
        }
        torch.save(state, checkpoint_file)

        if max_to_keep > 0:
            model_files = [
                filename for filename in os.listdir(self.models_dir)
                if filename.startswith(type(model).__name__)
            ]
            if len(model_files) > max_to_keep:
                model_files.sort(key=lambda f: int(re.search(r'\d+', f.split('_')[-2]).group()))
                for filename in model_files[:-max_to_keep]:
                    os.remove(f'{self.models_dir}/{filename}')

    def save_optimizer(self, epoch_idx, max_to_keep=-1):
        """This function is usually common to all models"""
        checkpoint_file = '{}/{}_epoch{}_checkpoint.pth.tar'.format(
            self.models_dir,
            type(self.optimizer).__name__,
            epoch_idx
        )
        print("Saving optimizer to {}".format(checkpoint_file))
        state = {
            'epoch': epoch_idx,
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, checkpoint_file)

        if max_to_keep > 0:
            opt_files = [
                filename for filename in os.listdir(self.models_dir)
                if filename.startswith(type(self.optimizer).__name__)
            ]
            if len(opt_files) > max_to_keep:
                opt_files.sort(key=lambda f: int(re.search(r'\d+', f.split('_')[-2]).group()))
                for filename in opt_files[:-max_to_keep]:
                    os.remove(f'{self.models_dir}/{filename}')
