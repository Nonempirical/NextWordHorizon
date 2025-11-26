"""
Model Adapters - Interface for connecting different language models.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np

Tensor = Union[np.ndarray, "torch.Tensor"]


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Tokenizes text to a list of token IDs."""
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Detokenizes a list of token IDs back to text."""
        pass
    
    @abstractmethod
    def get_logits(self, token_ids: List[int]) -> Tensor:
        """Gets logits for a sequence of token IDs."""
        pass
    
    @abstractmethod
    def get_token_embedding(self, token_id: int) -> Tensor:
        """Gets embedding vector for a specific token."""
        pass
    
    @abstractmethod
    def get_sequence_embedding(self, token_ids: List[int]) -> Tensor:
        """Gets embedding for a whole sequence of tokens."""
        pass


class LocalHFAdapter(ModelAdapter):
    """Adapter for local HuggingFace models."""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        **kwargs
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self.model_name = model_name
        self.kwargs = kwargs
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            **kwargs
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            **kwargs
        )
        
        if device != "cuda" or "device_map" not in kwargs:
            self.model = self.model.to(device)
        
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=True)
    
    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_logits(self, token_ids: List[int]) -> Tensor:
        import torch
        
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0]
        
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        
        return logits
    
    def get_token_embedding(self, token_id: int) -> Tensor:
        import torch
        
        embedding_layer = self.model.get_input_embeddings()
        token_tensor = torch.tensor([[token_id]], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            emb = embedding_layer(token_tensor)[0, 0]
        
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        
        return emb
    
    def get_sequence_embedding(self, token_ids: List[int]) -> Tensor:
        import torch
        
        embedding_layer = self.model.get_input_embeddings()
        token_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            embeddings = embedding_layer(token_tensor)[0]
            mean_embedding = embeddings.mean(dim=0)
        
        if isinstance(mean_embedding, torch.Tensor):
            mean_embedding = mean_embedding.cpu().numpy()
        
        return mean_embedding


class RemoteHTTPAdapter(ModelAdapter):
    """Adapter for remote API via HTTP (not implemented)."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
    
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError("RemoteHTTPAdapter.encode not yet implemented")
    
    def decode(self, token_ids: List[int]) -> str:
        raise NotImplementedError("RemoteHTTPAdapter.decode not yet implemented")
    
    def get_logits(self, token_ids: List[int]) -> Tensor:
        raise NotImplementedError("RemoteHTTPAdapter.get_logits not yet implemented")
    
    def get_token_embedding(self, token_id: int) -> Tensor:
        raise NotImplementedError("RemoteHTTPAdapter.get_token_embedding not yet implemented")
    
    def get_sequence_embedding(self, token_ids: List[int]) -> Tensor:
        raise NotImplementedError("RemoteHTTPAdapter.get_sequence_embedding not yet implemented")

