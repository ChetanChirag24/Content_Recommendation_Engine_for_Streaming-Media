"""
Wide & Deep Neural Network for Content Recommendation
Combines memorization (wide) and generalization (deep) for personalized recommendations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WideAndDeepModel(nn.Module):
    """
    Wide & Deep Learning architecture for recommendation systems
    
    References:
        Cheng et al., 2016: Wide & Deep Learning for Recommender Systems
        https://arxiv.org/abs/1606.07792
    """
    
    def __init__(
        self,
        wide_dim: int,
        deep_dim: int,
        embedding_dims: Dict[str, Tuple[int, int]],
        hidden_units: List[int] = [512, 256, 128, 64],
        dropout_rate: float = 0.3
    ):
        """
        Args:
            wide_dim: Dimension of wide (linear) features
            deep_dim: Dimension of continuous deep features
            embedding_dims: Dict mapping categorical feature names to (vocab_size, embed_dim)
            hidden_units: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super(WideAndDeepModel, self).__init__()
        
        self.wide_dim = wide_dim
        self.deep_dim = deep_dim
        self.embedding_dims = embedding_dims
        
        # Wide component - linear model
        self.wide = nn.Linear(wide_dim, 1)
        
        # Embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0
        for feat_name, (vocab_size, embed_dim) in embedding_dims.items():
            self.embeddings[feat_name] = nn.Embedding(vocab_size, embed_dim)
            total_embed_dim += embed_dim
        
        # Deep component - DNN
        deep_input_dim = deep_dim + total_embed_dim
        
        self.deep_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        prev_dim = deep_input_dim
        for hidden_dim in hidden_units:
            self.deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final output layer (combines wide and deep)
        self.output = nn.Linear(prev_dim + 1, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
    
    def forward(self, wide_features, deep_cont_features, deep_cat_features):
        """
        Forward pass
        
        Args:
            wide_features: Tensor of shape (batch_size, wide_dim)
            deep_cont_features: Tensor of shape (batch_size, deep_dim)
            deep_cat_features: Dict of tensors for categorical features
        
        Returns:
            logits: Tensor of shape (batch_size, 1)
        """
        # Wide component
        wide_out = self.wide(wide_features)
        
        # Deep component - process embeddings
        embed_list = []
        for feat_name, feat_tensor in deep_cat_features.items():
            embed = self.embeddings[feat_name](feat_tensor)
            embed_list.append(embed)
        
        # Concatenate continuous features and embeddings
        if embed_list:
            deep_input = torch.cat([deep_cont_features] + embed_list, dim=1)
        else:
            deep_input = deep_cont_features
        
        # Deep neural network
        deep_out = deep_input
        for linear, bn, dropout in zip(self.deep_layers, self.batch_norms, self.dropouts):
            deep_out = linear(deep_out)
            deep_out = bn(deep_out)
            deep_out = F.relu(deep_out)
            deep_out = dropout(deep_out)
        
        # Combine wide and deep
        combined = torch.cat([wide_out, deep_out], dim=1)
        logits = self.output(combined)
        
        return logits


class RecommendationDataset(Dataset):
    """PyTorch Dataset for recommendation data"""
    
    def __init__(self, df, wide_features, deep_cont_features, deep_cat_features, target_col='clicked'):
        self.df = df
        self.wide_features = wide_features
        self.deep_cont_features = deep_cont_features
        self.deep_cat_features = deep_cat_features
        self.target_col = target_col
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Wide features
        wide = torch.tensor(row[self.wide_features].values, dtype=torch.float32)
        
        # Deep continuous features
        deep_cont = torch.tensor(row[self.deep_cont_features].values, dtype=torch.float32)
        
        # Deep categorical features
        deep_cat = {}
        for feat in self.deep_cat_features:
            deep_cat[feat] = torch.tensor(row[feat], dtype=torch.long)
        
        # Target
        target = torch.tensor(row[self.target_col], dtype=torch.float32)
        
        return wide, deep_cont, deep_cat, target


class ModelTrainer:
    """Training pipeline for Wide & Deep model"""
    
    def __init__(
        self,
        model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.001,
        weight_decay=1e-5
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for wide, deep_cont, deep_cat, targets in train_loader:
            wide = wide.to(self.device)
            deep_cont = deep_cont.to(self.device)
            deep_cat = {k: v.to(self.device) for k, v in deep_cat.items()}
            targets = targets.to(self.device).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(wide, deep_cont, deep_cat)
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for wide, deep_cont, deep_cat, targets in val_loader:
                wide = wide.to(self.device)
                deep_cont = deep_cont.to(self.device)
                deep_cat = {k: v.to(self.device) for k, v in deep_cat.items()}
                targets = targets.to(self.device).unsqueeze(1)
                
                logits = self.model(wide, deep_cont, deep_cat)
                loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_targets, all_preds)
        
        return avg_loss, auc
    
    def fit(self, train_loader, val_loader, epochs=10, early_stopping_patience=3):
        """Train the model"""
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_auc = self.evaluate(val_loader)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val AUC: {val_auc:.4f}"
            )
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/best_model.pt')
                logger.info(f"New best model saved with AUC: {best_val_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        logger.info(f"Training completed. Best Val AUC: {best_val_auc:.4f}")
        return self.history
    
    def predict(self, test_loader):
        """Generate predictions"""
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for wide, deep_cont, deep_cat, _ in test_loader:
                wide = wide.to(self.device)
                deep_cont = deep_cont.to(self.device)
                deep_cat = {k: v.to(self.device) for k, v in deep_cat.items()}
                
                logits = self.model(wide, deep_cont, deep_cat)
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)
        
        return np.array(all_preds)


def create_sample_model():
    """Create a sample Wide & Deep model"""
    
    # Define feature dimensions
    wide_dim = 100  # Cross-product features
    deep_dim = 50   # Continuous features
    
    # Embedding dimensions for categorical features
    embedding_dims = {
        'user_id': (10000, 32),      # 10k users, 32-dim embedding
        'content_id': (50000, 64),   # 50k content items, 64-dim embedding
        'genre': (20, 8),            # 20 genres, 8-dim embedding
        'device': (5, 4),            # 5 device types, 4-dim embedding
        'time_bucket': (24, 4),      # 24 hour buckets, 4-dim embedding
    }
    
    model = WideAndDeepModel(
        wide_dim=wide_dim,
        deep_dim=deep_dim,
        embedding_dims=embedding_dims,
        hidden_units=[512, 256, 128, 64],
        dropout_rate=0.3
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    logger.info("Creating Wide & Deep model...")
    model = create_sample_model()
    
    # Print model architecture
    logger.info(f"\nModel Architecture:")
    logger.info(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\nTotal parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")