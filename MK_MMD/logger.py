import os

class CompleteLogger:
    def __init__(self, root, phase='train'):
        self.get_checkpoint_path = None
        pass

    def get_checkpoint_path(self, name=None):
        """
        """
        return os.path.join(self.checkpoint_directory, str(name) + ".pth")