import os
import torch
import logging
import argparse
import torch.optim as optim

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, random_split

# Import model and dataset
from model import BatchDialogueSimilarityModel
from dataset2 import DialogueSimilarityDataset, create_dataloader


class DialogueTrainer:
    """
    Trainer for dialogue similarity models with streamlined functionality.
    """
    def __init__(self, train_data_path, val_data_path,gpu_id=0):
        # Basic configuration
        self.config = {
            'embed_dim': 768,
            'num_heads': 8,
            'num_layers': 2,
            'dropout_rate': 0.15,
            'temperature': 0.07,
            'batch_size': 256,
            'eval_batch_size': 64,
            'num_epochs': 30,
            'learning_rate': 1e-4,
            'train_file_path': train_data_path,
            'val_file_path': val_data_path,
            'max_seq_length': 10,
            'n_candidates': 10,
            'sentence_model': 'all-mpnet-base-v2',
            'cache_dir': '/data/home/Jianxin/MyProject/ContextCache/cache/test',
            'gpu_id': gpu_id
        }
        
        # Setup device
        self.device = torch.device(f"cuda:{self.config['gpu_id']}" if torch.cuda.is_available() else "cpu")
        
        # Setup experiment directory and logging
        self.setup_experiment()
        
    def setup_experiment(self):
        """Setup experiment directory and logging"""
        # Create experiment directory
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(f"results/exp_{time_str}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Log basic info
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        self.logger.info(f"Device: {self.device}")
    
    def setup_logging(self):
        """Configure logging to file and console"""
        log_file = self.exp_dir / "training.log"
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Get logger for this class
        self.logger = logging.getLogger(__name__)
        
    def save_checkpoint(self, model, optimizer, epoch, loss, is_best=False):
        """Save model checkpoint"""
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            checkpoint_path = self.exp_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model if indicated
        if is_best:
            best_model_path = self.exp_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"Best model saved: {best_model_path}")
    
    def prepare_data(self):
        """Prepare datasets and dataloaders"""
        
        # Initialize dataset
        self.logger.info("Initializing train dataset...")
        train_dataset = DialogueSimilarityDataset(
            data_file_path=self.config['train_file_path'],
            max_seq_length=self.config['max_seq_length'],
            n_candidates=self.config['n_candidates'],
            sentence_model_name=self.config['sentence_model'],
            batch_size=self.config['batch_size'],
            cache_dir=self.config['cache_dir'],
            log=self.logger,
            device=self.device
        )
        
        self.logger.info("Initializing val dataset...")
        val_dataset = DialogueSimilarityDataset(
            data_file_path=self.config['val_file_path'],
            max_seq_length=self.config['max_seq_length'],
            n_candidates=self.config['n_candidates'],
            sentence_model_name=self.config['sentence_model'],
            batch_size=self.config['batch_size'],
            cache_dir=self.config['cache_dir'],
            log=self.logger,
            device=self.device
        )
        
        self.logger.info(f"Splitting dataset: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        
        # Create dataloaders
        train_loader = create_dataloader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=4
        )
        
        val_loader = create_dataloader(
            val_dataset, 
            batch_size=self.config['eval_batch_size'], 
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def train(self):
        """Train the model"""
        # Prepare data
        train_loader, val_loader = self.prepare_data()
        
        # Initialize model
        self.logger.info("Initializing model...")
        model = BatchDialogueSimilarityModel(
            embed_dim=self.config['embed_dim'],
            num_heads=self.config['num_heads'],
            dropout_rate=self.config['dropout_rate'],
            num_layers=self.config['num_layers'],
            temperature=self.config['temperature']
        ).to(self.device)
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Training loop variables
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            # Training phase
            train_loss = self._run_epoch(model, train_loader, optimizer, is_training=True)
            self.logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}, Train Loss: {train_loss:.6f}")
            
            # Validation phase
            val_loss = self._run_epoch(model, val_loader, is_training=False)
            self.logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}, Val Loss: {val_loss:.6f}")
            
            # Check if this is the best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.logger.info(f"New best validation loss: {best_val_loss:.6f}")
            
            # Save checkpoint
            self.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                is_best=is_best
            )
            
        self.logger.info("Training completed!")
        return model
    
    def _run_epoch(self, model, data_loader, optimizer=None, is_training=True):
        """Run a single epoch of training or evaluation"""
        # Set model mode
        model.train() if is_training else model.eval()
        
        epoch_loss = 0.0
        batch_count = 0
        
        # Create progress bar
        desc = "Training" if is_training else "Validating"
        progress_bar = tqdm(data_loader, desc=desc)
        
        # Context manager for gradients
        context = torch.enable_grad() if is_training else torch.no_grad()
        
        # Additional metrics
        total_samples = 0
        correct_predictions = 0
        
        with context:
            for dialogues, masks in progress_bar:
                # Move to device
                dialogues = dialogues.to(self.device)
                masks = masks.to(self.device) if masks is not None else None
                
                # Forward pass
                similarities, loss = model(dialogues, masks)
                
                # Calculate accuracy: first candidate is the correct one
                batch_size = dialogues.size(0)
                total_samples += batch_size
                
                # Get predictions (highest similarity)
                predictions = torch.argmax(similarities, dim=1)
                correct_predictions += (predictions == 0).sum().item()  # First candidate is correct
                
                # Backward pass and optimization (if training)
                if is_training and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Update metrics
                batch_loss = loss.item()
                epoch_loss += batch_loss
                batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{batch_loss:.6f}',
                    'avg_loss': f'{epoch_loss/batch_count:.6f}'
                })
        
        # Calculate epoch-level metrics
        avg_epoch_loss = epoch_loss / batch_count
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        # Log metrics
        phase = "Training" if is_training else "Validation"
        self.logger.info(f"{phase} epoch completed with avg loss: {avg_epoch_loss:.6f}, accuracy: {accuracy:.4f}")
        
        return avg_epoch_loss
    
    def evaluate(self, model_path=None):
        """Evaluate a trained model"""
        # Prepare data
        _, val_loader = self.prepare_data()
        
        # Initialize model
        model = BatchDialogueSimilarityModel(
            embed_dim=self.config['embed_dim'],
            num_heads=self.config['num_heads'],
            dropout_rate=self.config['dropout_rate'],
            num_layers=self.config['num_layers'],
            temperature=self.config['temperature']
        ).to(self.device)
        
        # Load model checkpoint
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded model from {model_path}")
        else:
            # Try to load best model
            best_model_path = self.exp_dir / "best_model.pth"
            if best_model_path.exists():
                checkpoint = torch.load(best_model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Loaded best model from {best_model_path}")
            else:
                self.logger.warning("No model found for evaluation")
                return None
        
        # Evaluate
        with torch.no_grad():
            val_loss = self._run_epoch(model, val_loader, is_training=False)
        
        self.logger.info(f"Evaluation complete. Loss: {val_loss:.6f}")
        return {'val_loss': val_loss}


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a dialogue similarity model")
    parser.add_argument("--train_data_path", 
                        type=str, 
                        default= "/data/home/Jianxin/MyProject/ContextCache/data/new/train.jsonl", 
                        help="Path to data file")
    parser.add_argument("--val_data_path", 
                        type=str, 
                        default= "/data/home/Jianxin/MyProject/ContextCache/data/new/val.jsonl", 
                        help="Path to data file")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--evaluate", default=False, help="Evaluate model instead of training")
    parser.add_argument("--model_path", type=str, default= None,help="Path to model for evaluation")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = DialogueTrainer(train_data_path=args.train_data_path, 
                              val_data_path=args.val_data_path, 
                              gpu_id=args.gpu_id)
    
    if args.evaluate:
        # Evaluate model
        trainer.evaluate(model_path=args.model_path)
    else:
        # Train model
        trainer.train()