"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import torch.nn as nn
loss_function = nn.KLDivLoss(reduction="batchmean")


from typing import Tuple
from tqdm import tqdm
import torch

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler,
               epoch_num: int,
               device: torch.device) -> Tuple[float, float]:

    model.train()
    train_loss, correct, total = 0.0, 0, 0

    prog_bar = tqdm(dataloader, total=len(dataloader),desc=f"Train Epoch {epoch_num + 1}", unit="batch", leave=True)

    for batch_idx, (X, y) in enumerate(prog_bar):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        y_pred_class = y_pred.argmax(dim=1)
        correct += (y_pred_class == y).sum().item()
        total += y.size(0)

        avg_loss = train_loss / (batch_idx + 1)
        avg_acc = correct / total
        prog_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    final_loss = train_loss / len(dataloader)
    final_acc = correct / total

    return final_loss, final_acc



def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,epoch_num,
              device: torch.device) -> Tuple[float, float]:

    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0


    prog_bar = tqdm(dataloader, total=len(dataloader), desc=f"Train Epoch {epoch_num + 1}", unit="batch")

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(prog_bar):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,scheduler,
          device: torch.device) -> Dict[str, List]:

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,scheduler=scheduler,epoch_num=epoch,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,epoch_num=epoch,
          device=device)

        # Print out what's happening
        print(f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results


def train_kd_step(student: torch.nn.Module, teacher: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,scheduler,epoch_num,
               device: torch.device) -> Tuple[float, float]:

    # Put model in train mode
    student.train()
    teacher.eval()
    temperature = 5
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0


    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            teacher_logits = teacher(X)

        # 1. Forward pass student
        student_logits  = student(X)

        # 2. Calculate  and accumulate loss

        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)

        distillation_loss = loss_function(soft_student, soft_teacher) * (temperature ** 2)

        hard_loss = loss_fn(student_logits, y)

        loss = 0.5 * distillation_loss + 0.5 * hard_loss
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        scheduler.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(student_logits, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(student_logits)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def train_kd(model_student: torch.nn.Module,model_teacher: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int, scheduler,
          device: torch.device) -> Dict[str, List]:
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # Make sure model on target device
    model_student.to(device)
    model_teacher.to(device)


    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_kd_step(model_student,model_teacher,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer, scheduler=scheduler, epoch_num=epoch,
                                           device=device)
        test_loss, test_acc = test_step(model=model_student,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn, epoch_num=epoch,
                                        device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results
