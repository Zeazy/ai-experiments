from typing import Any


class TextRecognizer:
    """
    Image text recognition model
    """
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
    
    def predict(self, image) -> str:
        """
        Predict text from image
        """
        pass
    
    def load_model(self, model_path: str) -> Any:
        """
        Load model from path
        """
        pass