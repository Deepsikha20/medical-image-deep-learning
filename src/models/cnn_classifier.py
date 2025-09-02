#!/usr/bin/env python3
"""
CNN Classifier Models for Medical Image Classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16
import numpy as np
from typing import Tuple, Optional


class BasicCNN:
    """Basic CNN architecture for medical image classification"""

    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 dropout_rate: float = 0.5):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def build_model(self) -> keras.Model:
        """Build basic CNN model"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model


class TransferLearningCNN:
    """Transfer learning CNN using pre-trained models"""

    def __init__(self,
                 base_model_name: str = 'resnet50',
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 dropout_rate: float = 0.5,
                 fine_tune: bool = False,
                 fine_tune_layers: int = 50):

        self.base_model_name = base_model_name.lower()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.fine_tune = fine_tune
        self.fine_tune_layers = fine_tune_layers

    def _get_base_model(self) -> keras.Model:
        """Get pre-trained base model"""
        if self.base_model_name == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'efficientnetb0':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'vgg16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")

        return base_model

    def build_model(self) -> keras.Model:
        """Build transfer learning model"""
        # Get base model
        base_model = self._get_base_model()

        # Freeze base model layers initially
        base_model.trainable = False

        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model

    def unfreeze_layers(self, model: keras.Model) -> keras.Model:
        """Unfreeze layers for fine-tuning"""
        if self.fine_tune:
            base_model = model.layers[0]
            base_model.trainable = True

            # Fine-tune from this layer onwards
            fine_tune_at = len(base_model.layers) - self.fine_tune_layers

            # Freeze all the layers before fine_tune_at
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

        return model


class AttentionCNN:
    """CNN with attention mechanism for medical images"""

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 dropout_rate: float = 0.5):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def attention_block(self, x, filters):
        """Create attention block"""
        # Global average pooling
        gap = layers.GlobalAveragePooling2D()(x)

        # Dense layers for attention weights
        attention = layers.Dense(filters // 8, activation='relu')(gap)
        attention = layers.Dense(filters, activation='sigmoid')(attention)

        # Reshape for broadcasting
        attention = layers.Reshape((1, 1, filters))(attention)

        # Apply attention
        attended = layers.Multiply()([x, attention])

        return attended

    def build_model(self) -> keras.Model:
        """Build CNN with attention mechanisms"""
        inputs = layers.Input(shape=self.input_shape)

        # First block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Second block with attention
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = self.attention_block(x, 128)
        x = layers.MaxPooling2D((2, 2))(x)

        # Third block with attention
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = self.attention_block(x, 256)
        x = layers.MaxPooling2D((2, 2))(x)

        # Fourth block with attention
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = self.attention_block(x, 512)
        x = layers.MaxPooling2D((2, 2))(x)

        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)

        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        return model


class ModelBuilder:
    """Factory class for building different model architectures"""

    @staticmethod
    def build_model(model_type: str,
                   input_shape: Tuple[int, int, int] = (224, 224, 3),
                   num_classes: int = 2,
                   **kwargs) -> keras.Model:
        """Build model based on type"""

        if model_type.lower() == 'basic_cnn':
            builder = BasicCNN(input_shape, num_classes, **kwargs)
            return builder.build_model()

        elif model_type.lower() in ['transfer_learning', 'resnet50', 'efficientnetb0', 'vgg16']:
            base_model = model_type.lower() if model_type.lower() in ['resnet50', 'efficientnetb0', 'vgg16'] else 'resnet50'
            builder = TransferLearningCNN(base_model, input_shape, num_classes, **kwargs)
            return builder.build_model()

        elif model_type.lower() == 'attention_cnn':
            builder = AttentionCNN(input_shape, num_classes, **kwargs)
            return builder.build_model()

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def compile_model(model: keras.Model,
                     optimizer: str = 'adam',
                     learning_rate: float = 0.001,
                     loss: str = 'categorical_crossentropy',
                     metrics: list = None) -> keras.Model:
        """Compile model with specified parameters"""

        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall']

        # Create optimizer
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
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
    """Test model building"""
    print("Testing model architectures...")

    # Test basic CNN
    print("Building Basic CNN...")
    basic_model = ModelBuilder.build_model('basic_cnn', num_classes=2)
    basic_model = ModelBuilder.compile_model(basic_model)
    print(f"Basic CNN parameters: {basic_model.count_params():,}")

    # Test transfer learning
    print("Building ResNet50 Transfer Learning...")
    resnet_model = ModelBuilder.build_model('resnet50', num_classes=4)
    resnet_model = ModelBuilder.compile_model(resnet_model)
    print(f"ResNet50 parameters: {resnet_model.count_params():,}")

    # Test attention CNN
    print("Building Attention CNN...")
    attention_model = ModelBuilder.build_model('attention_cnn', num_classes=2)
    attention_model = ModelBuilder.compile_model(attention_model)
    print(f"Attention CNN parameters: {attention_model.count_params():,}")

    print("All models built successfully!")


if __name__ == "__main__":
    main()
