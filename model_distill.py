import os
from optimizer import *
from torchvision import transforms
import torch
import torchvision
from torch import nn
from torchvision import transforms
from helper_functions import download_data, set_seeds, plot_loss_curves
from going_modular import data_setup, engine
from PIL import Image
import time
from torchinfo import summary
from going_modular import utils
from going_modular.predictions import pred_and_plot_image
from torchvision.models import mobilenet_v3_small,mobilenet_v2
device = "cuda" if torch.cuda.is_available() else "cpu"
import timm
from torchvision import datasets
from torch.utils.data import DataLoader
from huggingface_hub import login
from torchvision import datasets, transforms, models



import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--img_size',default=224,help='image size',type=int)
parser.add_argument('--weight_decay',default=0.0005,help='L2 regularization',type=float)

parser.add_argument('--gradient_clipping',default=1.,help='gradient_clipping',type=float)


parser.add_argument('--lr',default=1e-3,help='learning rate',type=float)
parser.add_argument('--momentum',default=0.9,help='previous accumulation',type=float)
parser.add_argument('--optimizer',default="AdamW",help='optimizer',type=str)


parser.add_argument('--epochs',default=150,help='number of epochs to train for',type=int)
parser.add_argument('--batch',default=2,help='batch size for data loader',type=int)
parser.add_argument('--scheduler',action='store_true',)
args = parser.parse_args()




if __name__ == '__main__':

    transform = transforms.Compose(
                [transforms.Resize((args.img_size,args.img_size)),transforms.ToTensor(),transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)





    student = models.mobilenet_v2(weights=None)
    num_classes = 10
    student.classifier[1] = nn.Linear(student.classifier[1].in_features, num_classes)

    """model.load_state_dict(torch.load("/models/mobilenet.pth"))
    model.eval()
    student=model.to(device)"""



    teacher = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)
    teacher.to(device)
    teacher.load_state_dict(torch.load("models/pretrained_vit.pth"))
    teacher.eval()
    teacher=teacher.to(device)


    optimizer = smart_optimizer(student, args.optimizer, args.lr, args.momentum, args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = args.epochs
    warmup_steps = len(train_loader)*int(epochs*0.15)
    stable_steps = len(train_loader)*int(epochs*0.05)
    decay_steps = len(train_loader)*int(epochs*0.8)
    scheduler = WarmupStableDecayLR( optimizer, warmup_steps, stable_steps, decay_steps, warmup_start_lr=1e-6, base_lr=1e-3, final_lr=1e-6)



    mobilenet_distill = engine.train_kd(model_student=student,model_teacher=teacher,
                                        train_dataloader=train_loader,test_dataloader=test_loader,
                                        optimizer=optimizer,loss_fn=loss_fn,
                                        epochs=epochs,
                                        scheduler=scheduler,
                                        device=device)

    utils.save_model(model=student, target_dir="models", model_name="mobilenet_distill.pth")