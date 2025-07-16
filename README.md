# MASC-AE: Multi-Attention Stress Crack AutoEncoder

A deep learning framework for Von Mises stress prediction from crack patterns using advanced attention mechanisms and feature pyramid networks.

## Overview

MASC-AE is a sophisticated autoencoder architecture designed to predict Von Mises stress distributions from crack pattern images. The model incorporates multiple attention mechanisms including CBAM (Convolutional Block Attention Module), self-attention, and Feature Pyramid Networks (FPN) to achieve high-quality stress field reconstruction.

## Key Features

- **Multi-Attention Architecture**: Combines channel attention, spatial attention, and self-attention mechanisms
- **CBAM Integration**: Convolutional Block Attention Module for enhanced feature representation
- **Feature Pyramid Network**: Multi-scale feature fusion for better spatial understanding
- **Advanced Loss Functions**: Combines MSE, SSIM, and gradient losses for comprehensive training
- **Comprehensive Evaluation**: Multiple metrics including MSE, RMSE, MAE, MAPE, R², SSIM, and PSNR
- **Data Augmentation**: Spatial transformations with consistent crack-stress pair augmentation
- **TensorBoard Integration**: Real-time training monitoring and visualization

## Architecture Components

### Attention Mechanisms
- **Self-Attention**: Captures long-range spatial dependencies
- **Channel Attention**: Emphasizes important feature channels
- **Spatial Attention**: Focuses on critical spatial locations
- **CBAM**: Combined channel and spatial attention module

### Network Structure
- **Encoder**: Progressive downsampling with attention-enhanced feature extraction
- **Decoder**: Upsampling with skip connections and attention refinement
- **Feature Pyramid Network**: Multi-scale feature integration

## Requirements
numpy==1.24.3
torch==2.0.1
torchvision==0.15.2
matplotlib==3.7.1
scikit-learn==1.2.2
tqdm==4.65.0 

