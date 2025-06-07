# kd-transformers-mobilenet

This repository explores response-based knowledge distillation (KD), where a lightweight student model ( MobileNet ) is trained to mimic the output logits (responses) of a larger teacher model (ViT).

The goal is to improve the performance of compact models without increasing inference cost ideal for deployment on edge devices or low resource environments.

## Key Concepts

Knowledge Distillation (KD): A training strategy where the student learns from both ground truth labels and the soft output (logits) of a larger teacher model.

Response-Based KD: The student is trained to match the final-layer outputs (logits) of the teacher — without requiring intermediate feature matching or structural alignment.

Models Used:

Teacher: Large Transformer ( ViT) 

Student: MobileNetV2
