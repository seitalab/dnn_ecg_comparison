from typing import Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from codes.supports import utils
from codes.supports.monitor import Monitor
from codes.functions.train_base import BaseTrainer

class ModelTrainer(BaseTrainer):

    def _train(self, iterator: Iterable) -> Tuple[float, float]:
        """
        Run train mode iteration.

        Args:
            iterator (Iterable):
        Returns:
            loss (float):
            score (float):
        """

        monitor = Monitor()
        self.model.train()

        for X, y in tqdm(iterator):
            self.optimizer.zero_grad()
            X = X.float().to(self.device)
            y = y.float().to(self.device)
            y_pred = self.model(X)

            minibatch_loss = self.loss_func(y_pred, y)
            minibatch_loss.backward()
            self.optimizer.step()

            monitor.store_loss(float(minibatch_loss), len(X))
            monitor.store_result(y, F.sigmoid(y_pred))
            # monitor.store_result(y, y_pred)

        loss = monitor.average_loss()
        score = monitor.macro_auc_roc()
        return loss, score

    def _evaluate(self, iterator: Iterable) -> Tuple[float, float]:
        """
        Run evaluation mode iteration.

        Args:
            iterator (Iterable):
        Returns:
            loss (float):
            score (float):
        """

        monitor = Monitor()
        self.model.eval()

        with torch.no_grad():
            for X, y in tqdm(iterator):
                X = X.float().to(self.device)
                y = y.float().to(self.device)
                y_pred = utils.aggregator(self.model, X)

                monitor.store_loss(0, len(X))
                monitor.store_result(y, F.sigmoid(y_pred))
                # monitor.store_result(y, y_pred)

        loss = monitor.average_loss()
        score = monitor.macro_auc_roc()
        return loss, score

    def run(self, train_loader: Iterable, valid_loader: Iterable) -> None:
        """
        Args:
            train_loader (Iterable): Dataloader for training data.
            valid_loader (Iterable): Dataloader for validation data.
        Returns:
            None
        """

        # best_loss = np.inf # Sufficietly large
        # early_stopper = utils.EarlyStopper(mode="min", self.patience)

        best_score = -1 * np.inf # Sufficiently small
        early_stopper = utils.EarlyStopper(mode="max", patience=self.patience)
        writer = SummaryWriter(self.log_dir)

        for epoch in range(1, self.epochs+1):
            print("-"*80)
            print(f"Epoch {epoch}")
            train_loss, train_score = self._train(train_loader)
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_auc_roc", train_score, epoch)
            print(f'-> Train loss: {train_loss:.4f}, score: {train_score:.4f}')

            if epoch % self.eval_every == 0:
                eval_loss, eval_score = self._evaluate(valid_loader)
                writer.add_scalar("eval_loss", eval_loss, epoch)
                writer.add_scalar("eval_auc_roc", eval_score, epoch)
                print(f'-> Eval loss: {eval_loss:.4f}, score: {eval_score:.4f}')

                # if eval_loss < best_loss:
                    # print(f"Validation loss improved {best_loss:.4f} -> {eval_loss:.4f}")
                    # best_loss = eval_loss
                    # self._save_model()
                if eval_score > best_score:
                    print(f"Validation score improved {best_score:.4f} -> {eval_score:.4f}")
                    best_score = eval_score
                    self._save_model(best_score)

                if early_stopper.stop_training(eval_score):
                    print("Early stopping applied, stop training")
                    break
        print("-"*80)

class ModelTrainerMC(ModelTrainer):

    def set_lossfunc(self, weight:Optional[np.ndarray]=None) -> None:
        """
        Set loss function.
        Args:
            weight (Optional[np.ndarray]):
        Returns:
            None
        """
        if weight is not None:
            weight = torch.Tensor(weight).to(self.device)
        self.loss_func = nn.CrossEntropyLoss(weight=weight, reduction="sum")

    def _train(self, iterator: Iterable) -> Tuple[float, float]:
        """
        Run train mode iteration.

        Args:
            iterator (Iterable):
        Returns:
            loss (float):
            score (float):
        """

        monitor = Monitor()
        self.model.train()

        for X, y in tqdm(iterator):
            self.optimizer.zero_grad()
            X = X.float().to(self.device)
            y = y.long().to(self.device)
            y_pred = self.model(X)

            minibatch_loss = self.loss_func(y_pred, y)
            minibatch_loss.backward()
            self.optimizer.step()

            monitor.store_loss(float(minibatch_loss), len(X))
            monitor.store_result(y, F.softmax(y_pred))
            # monitor.store_result(y, y_pred)

        loss = monitor.average_loss()
        score = monitor.macro_f1()
        return loss, score

    def _evaluate(self, iterator: Iterable) -> Tuple[float, float]:
        """
        Run evaluation mode iteration.

        Args:
            iterator (Iterable):
        Returns:
            loss (float):
            score (float):
        """

        monitor = Monitor()
        self.model.eval()

        with torch.no_grad():
            for X, y in tqdm(iterator):
                X = X.float().to(self.device)
                y = y.long().to(self.device)

                y_pred = utils.aggregator(self.model, X)
                monitor.store_loss(0, len(X))
                monitor.store_result(y, F.softmax(y_pred))
                # monitor.store_result(y, y_pred)

        loss = monitor.average_loss()
        score = monitor.macro_f1()
        return loss, score
