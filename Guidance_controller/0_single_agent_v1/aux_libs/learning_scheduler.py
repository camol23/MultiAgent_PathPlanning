import torch
import torch.optim as optim

import math
import matplotlib.pyplot as plt

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        """
        Cosine learning rate scheduler with warmup.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs for warmup
            total_epochs: Total number of training epochs
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Store base learning rate for each parameter group
        self.base_lrs = []
        for group in optimizer.param_groups:
            self.base_lrs.append(group['lr'])
    
    def get_lr(self):
        """Calculate learning rates for current epoch"""
        lrs = []
        for base_lr in self.base_lrs:
            if self.current_epoch <= self.warmup_epochs:
                # Linear warmup
                lr = base_lr * (self.current_epoch / self.warmup_epochs)
            else:
                # Cosine decay
                progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs
        
    def step(self):
        """Update learning rate in optimizer"""
        # Calculate new learning rates
        lrs = self.get_lr()
        
        # Update learning rate for each parameter group
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
            
        self.current_epoch += 1



# # Example usage showing the learning rate updates:
# def example_usage():
#     # Create a simple model
#     model = torch.nn.Linear(10, 2)
    
#     # Initialize optimizer with multiple parameter groups
#     optimizer = optim.AdamW([
#         {'params': model.weight, 'lr': 0.001},
#         {'params': model.bias, 'lr': 0.002}
#     ])
    
#     # Create scheduler
#     scheduler = CosineWarmupScheduler(
#         optimizer=optimizer,
#         warmup_epochs=5,
#         total_epochs=30
#     )
    
#     # Training loop example
#     print("Initial learning rates:", [group['lr'] for group in optimizer.param_groups])
    
#     for epoch in range(30):
#         # Your training code here
        
#         # Update scheduler
#         scheduler.step()
        
#         # Print current learning rates
#         current_lrs = [group['lr'] for group in optimizer.param_groups]
#         print(f"Epoch {epoch + 1}, Learning rates: {current_lrs}")



def plot_lr_schedule(scheduler, steps):
    """
    Visualize the learning rate schedule.
    
    Args:
        scheduler: Learning rate scheduler
        steps: Number of steps to plot
    """
    lrs = []
    for i in range(steps):
        scheduler.step()
        lrs.append(scheduler.optimizer.param_groups[0]['lr'])
    
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.show()
