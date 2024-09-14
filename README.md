# KBY-Net: Deep Image Restoration for Improving Object Detection

KBY-Net is a novel end-to-end Y-Net architecture built upon YOLOv8 that addresses the challenge of object detection in rainy weather conditions. It leverages multi-task learning for simultaneous image restoration and object detection.

## Table of Contents
- [Abstract](#abstract)
- [Problem Statement & Objectives](#problem-statement--objectives)
- [Features](#features)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Conclusion](#conclusion)


## Abstract

KBY-Net incorporates a KBY-decoder for image deraining, a transposed attention (MDTA) module for capturing long-range dependencies, and a multi-axis feature fusion (MFF) block for refined feature extraction. Evaluations on RainCityscapes and RAIN-KITTI datasets demonstrate that KBY-Net significantly outperforms state-of-the-art object detection approaches in challenging rainy scenes.

## Problem Statement & Objectives

### Problem Statement
- Rain reduces object detection effectiveness.
- Image quality deteriorates due to rainfall.
- Existing solutions (single image deraining, domain adaptation) have limitations.

### Objectives
- Develop a shared-backbone model for detection and deraining.
- Enhance object detection in rainy images with a multi-task approach.
- Optimize image restoration as a beneficial byproduct.

## Features

- End-to-end Y-Net architecture based on YOLOv8
- Simultaneous image deraining and object detection
- KBY-decoder for image deraining
- Transposed attention (MDTA) module
- Multi-axis feature fusion (MFF) block
- Improved performance in rainy weather conditions

## Architecture

KBY-Net consists of two main branches:

1. Image Deraining Branch:
   - Skip Connections
   - KB-CSP blocks
   - PixelShuffle

2. Object Detection Branch:
   - Based on YOLOv8-small architecture
   - MDTA and MFF blocks

## Datasets

- RainCityscapes: 2500 training images and 450 testing images, each with 10 rain variants.
- RAIN-KITTI: 2500 training images and 450 testing images, each with 5 rain variants.

## Implementation Details

- Batch Size: 16
- Optimizer: SGD
- Learning Rate:
  - Deraining Module: 2e-2
  - Detection Module: 2e-4

## Results

KBY-Net outperforms state-of-the-art methods in object detection while maintaining a good balance with image restoration quality. For detailed results, please refer to the paper.

## Conclusion

KBY-Net introduces a novel Y-Net architecture for concurrent object detection and image restoration in rainy conditions. It outperforms state-of-the-art methods in object detection while maintaining a good balance with image restoration quality.


For more details, please refer to the full paper or contact the contributors.
