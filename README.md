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

## Dataset Structure
The dataset should be organized as follows:
data/
├── crack_images/
│   ├── crack_1_1.npy
│   ├── crack_1_2.npy
│   └── ...
└── stress_fields/
    ├── 1_var_Stress_1_Z.npy
    ├── 1_var_Stress_2_Z.npy
    └── ...
- Crack Images : Binary or grayscale crack pattern images (256×256)
- Stress Fields : Corresponding Von Mises stress distributions (256×256)

## Usage
### Training
Train the simplified autoencoder model:
python train.py --crack_dir /path/to/crack_images \
                --stress_dir /path/to/stress_fields \
                --batch_size 8 \
                --num_epochs 100 \
                --latent_dim 256 \
                --learning_rate 0.001 \
                --save_dir checkpoints \
                --model_type simplified
### Testing
Evaluate the trained model:
python test.py --model_path checkpoints/best_model.pth \
               --crack_dir /path/to/test_crack_images \
               --stress_dir /path/to/test_stress_fields \
               --output_dir results \
               --latent_dim 256 \
               --model_type simplified

### Key Parameters
- --latent_dim : Latent space dimension (default: 256)
- --batch_size : Training batch size (default: 8)
- --learning_rate : Learning rate (default: 0.001)
- --dropout_p : Dropout probability (default: 0.2)
- --stress_weight : Weight for stress loss (default: 1.0)
- --mse_weight : Weight for MSE loss component (default: 1.0)
- --ssim_weight : Weight for SSIM loss component (default: 1.0)
- --gradient_weight : Weight for gradient loss component (default: 1.0)

## Model Architecture Details
### SimplifiedVonMisesAutoEncoder
The main model consists of:

1. Encoder Path :
   
   - 5 encoder blocks with progressive channel increase (1→16→32→64→128→256)
   - Global self-attention after the final encoder block
   - Adaptive average pooling to latent space
2. Decoder Path :
   
   - 5 decoder blocks with progressive channel decrease (256→128→64→32→16→1)
   - Skip connections from corresponding encoder layers
   - CBAM attention modules for feature refinement
3. Attention Modules :
   
   - Self-Attention : Query-key-value mechanism for spatial relationships
   - Channel Attention : Squeeze-and-excitation style channel weighting
   - Spatial Attention : Spatial feature descriptor generation
   - CBAM : Sequential application of channel and spatial attention

## Loss Functions
The framework employs a multi-component loss function:
Total Loss = w_mse × MSE + w_ssim × (1 - SSIM) + w_grad × Gradient Loss
- MSE Loss : Pixel-wise mean squared error
- SSIM Loss : Structural similarity index for perceptual quality
- Gradient Loss : Sobel operator-based edge preservation

## Evaluation Metrics
The model is evaluated using comprehensive metrics:

- MSE : Mean Squared Error
- RMSE : Root Mean Squared Error
- MAE : Mean Absolute Error
- MAPE : Mean Absolute Percentage Error
- R² : Coefficient of Determination
- SSIM : Structural Similarity Index
- PSNR : Peak Signal-to-Noise Ratio

## Results Visualization
The testing script generates:

- Comparative visualizations of input cracks, ground truth stress, and predictions
- Detailed metrics for each test sample
- Statistical summaries and performance plots
- TensorBoard logs for training monitoring

## File Structure
MASC-AE/
├── model.py              # Model architecture definitions
├── train.py              # Training script
├── test.py               # Testing and evaluation script
├── dataset.py            # Dataset loading and preprocessing
├── requirements.txt      # Dependencies
├── utils/
│   └── loss_utils.py     # Loss function implementations
└── README.md            # This file

## Key Features Implementation
### Data Augmentation
- Synchronized spatial transformations for crack-stress pairs
- Random horizontal/vertical flips and rotations
- Gaussian noise injection for robustness
### Training Strategy
- Single-stage training with simultaneous stress prediction
- Early stopping with validation loss monitoring
- Learning rate scheduling with ReduceLROnPlateau
- Comprehensive logging with TensorBoard
### Model Flexibility
- Configurable latent dimensions
- Adjustable dropout rates
- Modular attention mechanisms
- Extensible loss function combinations
  
## Technical Details
### SSIM Implementation
The project includes a custom SSIM (Structural Similarity Index) implementation for perceptual loss calculation:

- Gaussian window-based similarity measurement
- Multi-scale structural comparison
- Differentiable implementation for gradient-based optimization
### Attention Mechanisms
- Self-Attention : Implements query-key-value attention for capturing long-range dependencies
- Channel Attention : Uses global average and max pooling for channel-wise feature recalibration
- Spatial Attention : Generates spatial attention maps using channel-wise statistics
- CBAM : Combines channel and spatial attention in sequence
### Feature Pyramid Network
- Top-down pathway with lateral connections
- Multi-scale feature fusion
- Nearest neighbor upsampling for feature alignment

## Performance Optimization
### Memory Efficiency
- Gradient checkpointing for large models
- Efficient attention computation
- Batch-wise processing for large datasets
### Training Stability
- Batch normalization in encoder/decoder blocks
- Dropout regularization
- Learning rate scheduling
- Early stopping mechanism

## Troubleshooting
### Common Issues
1. CUDA Out of Memory : Reduce batch size or use gradient accumulation
2. NaN Loss : Check learning rate and data normalization
3. Poor Convergence : Adjust loss weights and learning rate schedule
### Data Requirements
- Input images should be normalized to [0, 1] range
- Crack and stress data must be properly paired
- Consistent image dimensions (256×256) required

## Citation
If you use this code in your research, please cite:
@article{masc_ae_2025,
  title={ROCKS (Risk, Operations, Crack, Knowledge, Stress), a georeferenced framework for real-time tunnel face risk assessment and construction advice via AI-driven stress analysisn},
  author={Yang Gang},
  journal={Journal},
  year={2025}
}
