#!/usr/bin/env python3
"""
U-Net Model Implementations for Medical Image Segmentation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, Optional


class UNet:
    """Standard U-Net implementation for medical image segmentation"""

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 dropout_rate: float = 0.1):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def conv_block(self, inputs, filters, kernel_size=3, padding='same'):
        """Convolutional block with BatchNorm and ReLU"""
        x = layers.Conv2D(filters, kernel_size, padding=padding)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        return x

    def encoder_block(self, inputs, filters):
        """Encoder block with convolution and max pooling"""
        conv = self.conv_block(inputs, filters)
        pool = layers.MaxPooling2D((2, 2))(conv)
        pool = layers.Dropout(self.dropout_rate)(pool)

        return conv, pool

    def decoder_block(self, inputs, skip_features, filters):
        """Decoder block with upsampling and skip connections"""
        upsample = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)

        # Concatenate skip connection
        concat = layers.Concatenate()([upsample, skip_features])
        conv = self.conv_block(concat, filters)

        return conv

    def build_model(self) -> keras.Model:
        """Build U-Net model"""
        inputs = layers.Input(shape=self.input_shape)

        # Encoder
        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)

        # Bottleneck
        b1 = self.conv_block(p4, 1024)

        # Decoder
        d1 = self.decoder_block(b1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)

        # Output layer
        outputs = layers.Conv2D(self.num_classes, 1, padding='same', activation='softmax')(d4)

        model = keras.Model(inputs, outputs, name='UNet')
        return model


class AttentionUNet(UNet):
    """U-Net with attention gates"""

    def attention_gate(self, g, x, filters):
        """Attention gate mechanism"""
        # Gating signal
        g_conv = layers.Conv2D(filters, 1, padding='same')(g)
        g_bn = layers.BatchNormalization()(g_conv)

        # Input signal
        x_conv = layers.Conv2D(filters, 1, padding='same')(x)
        x_bn = layers.BatchNormalization()(x_conv)

        # Attention coefficients
        add = layers.Add()([g_bn, x_bn])
        relu = layers.ReLU()(add)
        attention = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(relu)

        # Apply attention
        multiply = layers.Multiply()([x, attention])

        return multiply

    def decoder_block_with_attention(self, inputs, skip_features, filters):
        """Decoder block with attention gate"""
        upsample = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)

        # Apply attention gate
        attended_skip = self.attention_gate(upsample, skip_features, filters // 2)

        # Concatenate
        concat = layers.Concatenate()([upsample, attended_skip])
        conv = self.conv_block(concat, filters)

        return conv

    def build_model(self) -> keras.Model:
        """Build Attention U-Net model"""
        inputs = layers.Input(shape=self.input_shape)

        # Encoder
        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)

        # Bottleneck
        b1 = self.conv_block(p4, 1024)

        # Decoder with attention
        d1 = self.decoder_block_with_attention(b1, s4, 512)
        d2 = self.decoder_block_with_attention(d1, s3, 256)
        d3 = self.decoder_block_with_attention(d2, s2, 128)
        d4 = self.decoder_block_with_attention(d3, s1, 64)

        # Output layer
        outputs = layers.Conv2D(self.num_classes, 1, padding='same', activation='softmax')(d4)

        model = keras.Model(inputs, outputs, name='AttentionUNet')
        return model


class UNetPlusPlus:
    """U-Net++ implementation with nested skip connections"""

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 dropout_rate: float = 0.1,
                 deep_supervision: bool = False):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.deep_supervision = deep_supervision

    def conv_block(self, inputs, filters, kernel_size=3, padding='same'):
        """Convolutional block"""
        x = layers.Conv2D(filters, kernel_size, padding=padding)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        return x

    def build_model(self) -> keras.Model:
        """Build U-Net++ model"""
        inputs = layers.Input(shape=self.input_shape)

        # Initialize feature maps dictionary
        features = {}

        # First column (encoder path)
        features[(0, 0)] = self.conv_block(inputs, 64)
        features[(1, 0)] = self.conv_block(layers.MaxPooling2D((2, 2))(features[(0, 0)]), 128)
        features[(2, 0)] = self.conv_block(layers.MaxPooling2D((2, 2))(features[(1, 0)]), 256)
        features[(3, 0)] = self.conv_block(layers.MaxPooling2D((2, 2))(features[(2, 0)]), 512)
        features[(4, 0)] = self.conv_block(layers.MaxPooling2D((2, 2))(features[(3, 0)]), 1024)

        # Nested connections
        for i in range(4):
            for j in range(1, 4-i):
                # Upsample from lower level
                upsample = layers.Conv2DTranspose(
                    64 * (2**(3-j)), (2, 2), strides=2, padding='same'
                )(features[(i+1, j-1)])

                # Collect skip connections
                skip_connections = [features[(i, k)] for k in range(j)]
                skip_connections.append(upsample)

                # Concatenate all connections
                concat = layers.Concatenate()(skip_connections)

                # Convolutional block
                features[(i, j)] = self.conv_block(concat, 64 * (2**(3-j)))

        # Output layer
        if self.deep_supervision:
            # Multiple outputs for deep supervision
            outputs = []
            for j in range(1, 4):
                output = layers.Conv2D(
                    self.num_classes, 1, padding='same', activation='softmax'
                )(features[(0, j)])
                outputs.append(output)

            model = keras.Model(inputs, outputs, name='UNetPlusPlus')
        else:
            # Single output from the final layer
            outputs = layers.Conv2D(
                self.num_classes, 1, padding='same', activation='softmax'
            )(features[(0, 3)])

            model = keras.Model(inputs, outputs, name='UNetPlusPlus')

        return model


class ResUNet(UNet):
    """U-Net with residual connections"""

    def residual_block(self, inputs, filters):
        """Residual block"""
        shortcut = inputs

        # First conv
        x = layers.Conv2D(filters, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Second conv
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        # Adjust shortcut if needed
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # Add residual connection
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)

        return x

    def build_model(self) -> keras.Model:
        """Build ResU-Net model"""
        inputs = layers.Input(shape=self.input_shape)

        # Encoder with residual blocks
        s1 = self.residual_block(inputs, 64)
        p1 = layers.MaxPooling2D((2, 2))(s1)

        s2 = self.residual_block(p1, 128)
        p2 = layers.MaxPooling2D((2, 2))(s2)

        s3 = self.residual_block(p2, 256)
        p3 = layers.MaxPooling2D((2, 2))(s3)

        s4 = self.residual_block(p3, 512)
        p4 = layers.MaxPooling2D((2, 2))(s4)

        # Bottleneck
        b1 = self.residual_block(p4, 1024)

        # Decoder
        d1 = self.decoder_block(b1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)

        # Output layer
        outputs = layers.Conv2D(self.num_classes, 1, padding='same', activation='softmax')(d4)

        model = keras.Model(inputs, outputs, name='ResUNet')
        return model


class SegmentationModelBuilder:
    """Factory class for building segmentation models"""

    @staticmethod
    def build_model(model_type: str,
                   input_shape: Tuple[int, int, int] = (224, 224, 3),
                   num_classes: int = 2,
                   **kwargs) -> keras.Model:
        """Build segmentation model based on type"""

        if model_type.lower() == 'unet':
            builder = UNet(input_shape, num_classes, **kwargs)
            return builder.build_model()

        elif model_type.lower() == 'attention_unet':
            builder = AttentionUNet(input_shape, num_classes, **kwargs)
            return builder.build_model()

        elif model_type.lower() == 'unetplusplus':
            builder = UNetPlusPlus(input_shape, num_classes, **kwargs)
            return builder.build_model()

        elif model_type.lower() == 'resunet':
            builder = ResUNet(input_shape, num_classes, **kwargs)
            return builder.build_model()

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def compile_model(model: keras.Model,
                     optimizer: str = 'adam',
                     learning_rate: float = 0.001,
                     loss: str = 'sparse_categorical_crossentropy',
                     metrics: list = None) -> keras.Model:
        """Compile segmentation model"""

        if metrics is None:
            metrics = ['accuracy']

        # Create optimizer
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer

        # Compile model
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )

        return model


def main():
    """Test segmentation model building"""
    print("Testing segmentation model architectures...")

    # Test U-Net
    print("Building U-Net...")
    unet_model = SegmentationModelBuilder.build_model('unet', num_classes=3)
    unet_model = SegmentationModelBuilder.compile_model(unet_model)
    print(f"U-Net parameters: {unet_model.count_params():,}")

    # Test Attention U-Net
    print("Building Attention U-Net...")
    attention_unet = SegmentationModelBuilder.build_model('attention_unet', num_classes=3)
    attention_unet = SegmentationModelBuilder.compile_model(attention_unet)
    print(f"Attention U-Net parameters: {attention_unet.count_params():,}")

    print("All segmentation models built successfully!")


if __name__ == "__main__":
    main()
