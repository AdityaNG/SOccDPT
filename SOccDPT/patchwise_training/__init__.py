"""
PatchWise training, SOccDPT
"""
import math

import torch


class PatchWise:

    """
    This class is used to train the network in a patch-wise manner.
    The network is trained in a patch-wise manner to reduce
    the memory requirements.
    Example use
    -----------
    ```
    for batch in train_set:

        x, x_raw, mask_disp, y_disp, mask_seg, y_seg = batch

        for net_patch in PatchWise(net, training_percentage):

            with torch.cuda.amp.autocast(enabled=amp):
                y_disp_pred, y_seg_pred, points = net_patch(x)

                loss = loss_logic(
                    y_disp_pred, y_disp, mask_disp, y_seg_pred, y_seg, mask_seg
                )

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        pbar.update(batch_size)
    ```
    """

    def __init__(self, net, train_percentage):
        assert isinstance(
            net, torch.nn.Module
        ), "The network must be a torch.nn.Module, got {}".format(type(net))
        self.net = net
        self.train_percentage = train_percentage
        self.last_start = 0

        # Only consider parameters which have been enabled for training
        self.parameters = list(self.net.parameters())
        self._saved_requires_grad = {}
        for index, param in enumerate(self.parameters):
            self._saved_requires_grad[index] = param.requires_grad

        self.trainable_parameters = [
            p for p in self.parameters if p.requires_grad
        ]
        self.N = len(self.trainable_parameters)
        assert (
            self.N > 0
        ), "The number of parameters is 0, \
            check the network"

        # M: Number of parameters to unfreeze
        self.M = math.ceil(self.N * self.train_percentage)
        self.M = min(self.M, self.N)
        assert (
            self.M > 0
        ), "The number of parameters to unfreeze is 0, choose a higher \
            training percentage N={}, M={}".format(
            self.N, self.M
        )

        self._num_iterations = math.ceil(self.N / self.M)

        # Save all parameter weights
        # This is done so that all the parameters
        # can be updated at once in the end
        self._saved_weights = {}
        for index, param in enumerate(self.trainable_parameters):
            self._saved_weights[index] = param.clone().detach()

        self._updated_weights = {}

    def __len__(self):
        return self._num_iterations

    def __iter__(self):
        return self

    def __next__(self):
        #####################################################################
        # Save the parameters from the previous iteration
        if self.last_start > 0:
            sart_save = self.last_start - self.M
            end_save = self.last_start
            save_indices = range(sart_save, end_save, 1)

            for index, param in enumerate(self.trainable_parameters):
                if index in save_indices:
                    self._updated_weights[index] = param.clone().detach()
        #####################################################################
        # Stop iteration if all parameters have been trained
        if self.last_start >= self.N:
            # Batch update all parameters
            assert (
                len(self._updated_weights) == self.N
            ), "Not all parameters were updated"
            assert (
                len(self._saved_weights) == self.N
            ), "Not all parameters were saved"

            for index, param in enumerate(self.trainable_parameters):
                param.data = self._updated_weights[index].data

            # Restore requires_grad
            for index, param in enumerate(self.parameters):
                param.requires_grad = self._saved_requires_grad[index]

            raise StopIteration
        #####################################################################
        # If not all the parameters have been trained, then:
        #####################################################################
        # Reset all the parameters to the saved weights before starting
        # new iteration
        for index, param in enumerate(self.trainable_parameters):
            param.data = self._saved_weights[index].data
        #####################################################################
        # Unfreeze the next M parameters
        end_index = self.last_start + self.M
        end_index = min(end_index, self.N)

        unfreeze_indices = range(self.last_start, end_index, 1)
        assert len(unfreeze_indices) > 0, "No parameters were unfrozen"

        for index, param in enumerate(self.trainable_parameters):
            if index in unfreeze_indices:
                param.requires_grad = True
            else:
                param.requires_grad = False
        #####################################################################
        # Update the last start index
        self.last_start = end_index
        #####################################################################
        # Return the updated network for training of the selected parameters
        return self.net


class PatchWiseInplace:

    """
    This class is used to train the network in a patch-wise manner.
    The network is trained in a patch-wise manner to reduce the memory
    requirements.
    Example use
    -----------
    ```
    for batch in train_set:

        x, x_raw, mask_disp, y_disp, mask_seg, y_seg = batch

        for net_patch in PatchWiseInplace(net, training_percentage):

            with torch.cuda.amp.autocast(enabled=amp):
                y_disp_pred, y_seg_pred, points = net_patch(x)

                loss = loss_logic(
                    y_disp_pred, y_disp, mask_disp, y_seg_pred, y_seg, mask_seg
                )

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        pbar.update(batch_size)
    ```
    """

    def __init__(self, net, train_percentage):
        assert isinstance(
            net, torch.nn.Module
        ), "The network must be a torch.nn.Module, got {}".format(type(net))
        self.net = net
        self.train_percentage = train_percentage
        self.last_start = 0

        # Only consider parameters which have been enabled for training
        self.parameters = list(self.net.parameters())
        self._saved_requires_grad = {}
        for index, param in enumerate(self.parameters):
            self._saved_requires_grad[index] = param.requires_grad

        self.trainable_parameters = [
            p for p in self.parameters if p.requires_grad
        ]
        self.N = len(self.trainable_parameters)
        assert self.N > 0, "The number of parameters is 0, check the network"

        # M: Number of parameters to unfreeze
        self.M = math.ceil(self.N * self.train_percentage)
        self.M = min(self.M, self.N)
        assert (
            self.M > 0
        ), "The number of parameters to unfreeze is 0, \
            choose a higher training \
            percentage N={}, M={}".format(
            self.N, self.M
        )

        self._num_iterations = math.ceil(self.N / self.M)

    def __len__(self):
        return self._num_iterations

    def __iter__(self):
        return self

    def __next__(self):
        #####################################################################
        # Stop iteration if all parameters have been trained
        if self.last_start >= self.N:
            # Restore requires_grad
            for index, param in enumerate(self.parameters):
                param.requires_grad = self._saved_requires_grad[index]

            raise StopIteration
        #####################################################################
        # If not all the parameters have been trained, then:
        #####################################################################
        # Reset all the parameters to the saved weights before starting
        #   new iteration
        # for index, param in enumerate(self.trainable_parameters):
        #     param.data = self._saved_weights[index].data
        #####################################################################
        # Unfreeze the next M parameters
        end_index = self.last_start + self.M
        end_index = min(end_index, self.N)

        unfreeze_indices = range(self.last_start, end_index, 1)
        assert len(unfreeze_indices) > 0, "No parameters were unfrozen"

        for index, param in enumerate(self.trainable_parameters):
            if index in unfreeze_indices:
                param.requires_grad = True
            else:
                param.requires_grad = False
        #####################################################################
        # Update the last start index
        self.last_start = end_index
        #####################################################################
        # Return the updated network for training of the selected parameters
        return self.net
