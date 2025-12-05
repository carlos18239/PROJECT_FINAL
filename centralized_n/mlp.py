"""
MLP Model for Tabular NCD Classification.

Red neuronal MLP para clasificación binaria de mortalidad prematura
por Enfermedades No Transmisibles (ENT).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Perceptrón Multicapa para clasificación binaria.
    
    Arquitectura:
        Input (N features) -> FC(120) -> ReLU -> Dropout
                           -> FC(84)  -> ReLU -> Dropout
                           -> FC(1)   -> Output (logit)
    
    Devuelve logits (sin sigmoid) para usar con BCEWithLogitsLoss.
    """
    
    def __init__(self, in_features: int, seed: int = 42, p_dropout: float = 0.3):
        """
        Args:
            in_features: Número de features de entrada
            seed: Semilla para reproducibilidad
            p_dropout: Probabilidad de dropout
        """
        super().__init__()
        torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(in_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.dropout = nn.Dropout(p=p_dropout)
        
        # Guardar dimensión de entrada para referencia
        self.in_features = in_features

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)  # logits


class MLPLarger(nn.Module):
    """
    MLP más grande para datasets con más features.
    
    Arquitectura:
        Input -> FC(256) -> ReLU -> Dropout
              -> FC(128) -> ReLU -> Dropout
              -> FC(64)  -> ReLU -> Dropout
              -> FC(1)   -> Output
    """
    
    def __init__(self, in_features: int, seed: int = 42, p_dropout: float = 0.3):
        super().__init__()
        torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=p_dropout)
        
        self.in_features = in_features

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)
