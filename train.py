import sys
from typing import NamedTuple
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
from tqdm.notebook import tqdm, trange



def _apply_valid_kwargs(callable_, /, **kwargs):
    import inspect
    from functools import partial

    sig = inspect.signature(callable_)
    valid_keys = {
        p.name for p in sig.parameters.values()
        if p.kind in {p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD}
    }

    return partial(callable_, **{k: val for k, val in kwargs.items() if k in valid_keys})


class TrainingResult(NamedTuple):
    epoch: int
    train_acc: float
    train_loss: float
    val_acc: float
    val_acc_per_class: MulticlassAccuracy
    val_f1: float
    val_conf_matrix: MulticlassConfusionMatrix


def train_model(
        model: nn.Module, 
        /, 
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]], 
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        *,
        criterion: type[nn.modules.loss._Loss] = nn.CrossEntropyLoss, 
        optimizer: type[optim.Optimizer] = optim.Adam, 
        num_epochs: int = 50, 
        patience: int = sys.maxsize, 
        min_acc_improvment: float = 2e-3, # 0.2%
        num_classes: int = 10, 
        device: torch.device = torch.device('cpu'), 
        out_file: str = 'best_model.pth', 
        **kwargs 
    ) -> TrainingResult: 
    was_training = model.training
    criterion_ = _apply_valid_kwargs(criterion, **kwargs)().to(device)
    optimizer_ = _apply_valid_kwargs(optimizer, **kwargs)(model.parameters())
    
    best_results: TrainingResult | None = None
    epochs_no_improve = 0
    
    for epoch in (epoch_range := trange(0, max(1, num_epochs), desc="Epochs", leave=True, unit='epoch')):
        model.train()
        train_acc = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
        train_cost = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Training", leave=False, colour='goldenrod', unit='batch'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer_.zero_grad()
            preds = model(images)
            loss = criterion_(preds, labels)
            loss.backward()
            optimizer_.step()

            train_cost += loss.item()
            preds = torch.argmax(preds, dim=1)
            train_acc.update(preds, labels)
        
        train_loss = train_cost / (num_batches := len(train_loader))
        computed_train_acc = train_acc.compute().item()

        model.eval()
        val_acc = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
        val_acc_per_class = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)
        val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
        val_conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validating", leave=False, colour='blueviolet', unit='batch'):
                images, labels = images.to(device), labels.to(device)
                
                preds = model(images)
                preds = torch.argmax(preds, dim=1)
                val_acc.update(preds, labels)
                val_acc_per_class.update(preds, labels)
                val_f1.update(preds, labels)
                val_conf_matrix.update(preds, labels)

        computed_val_acc = val_acc.compute().item()
        computed_val_f1 = val_f1.compute().item()

        epoch_range.set_postfix(
            loss = "%.3f" % train_loss, 
            acc = "%.3f" % computed_train_acc,
            val_acc = "%.3f" % computed_val_acc,
            val_f1 = "%.3f" % computed_val_f1,
        )
        
        if best_results is None or computed_val_acc > best_results.val_acc + min_acc_improvment:
            epochs_no_improve = 0
            best_results = TrainingResult(
                epoch, computed_train_acc, train_loss, computed_val_acc, val_acc_per_class, computed_val_f1, val_conf_matrix
            )
            torch.save(model.state_dict(), out_file)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    assert best_results is not None, "Must train for at least one epoch"
    best_model_state = torch.load(out_file)
    model.load_state_dict(best_model_state)
    model.train(was_training)
    return best_results


def visualize_training_results(results: TrainingResult, rich_fmt: bool = True, labels: list['str'] | None = None): 
    from IPython.display import display, Markdown
    import matplotlib.pyplot as plt
    
    summary = f"""
### Training Results   

Best model at epoch {results.epoch}.

- **Train loss:** {results.train_loss:.3f}  
- **Train accuracy:** {results.train_acc:.3f}  
- **Validation accuracy:** {results.val_acc:.3f}  
- **Validation F1 score:** {results.val_f1:.3f}  
"""

    display(Markdown(summary) if rich_fmt else summary)

    fig1, ax1 = results.val_acc_per_class.plot()
    fig2, ax2 = results.val_conf_matrix.plot(cmap='Purples')
    
    ax1.set_title("Per-Class Accuracy")
    ax2.set_title("Confusion Matrix")

    if labels:
        handles, _ = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels, ncol=3)
        ax2.set_xticklabels(labels, rotation=90, ha='center')
        ax2.set_yticklabels(labels, rotation=0, va='center')

    #plt.tight_layout()
    plt.show()
    