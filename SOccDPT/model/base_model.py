import torch


class BaseModel(torch.nn.Module):
    def load_net(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        if path is None:
            # No model to load
            return
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            print("Loading optimizer state dict")
            parameters = parameters["model"]

        incompatible_keys = self.load_state_dict(
            parameters,
            strict=True,
        )
        print("incompatible_keys", incompatible_keys)

        del parameters
        torch.cuda.empty_cache()

    def get_device(
        self,
    ):
        return next(self.parameters()).device
