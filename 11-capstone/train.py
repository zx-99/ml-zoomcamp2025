import os
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=0):
    set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

def img_dataset_path(path):
    dataset_dir = path
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "valid")
    test_dir = os.path.join(dataset_dir, "test")
    return train_dir, val_dir, test_dir
  
def img_load_process(train, val, test):
    batch_size = 32
    # Create preprocess parameter (No augumetation)
    train_datagen_augm = ImageDataGenerator(rescale=1./255,
                                        # Augmentation parameters
                                        rotation_range=50,
                                        zoom_range = 0.1,
                                        brightness_range=[0.8, 1.2],
                                        horizontal_flip=True)
    
    val_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator_augm = train_datagen_augm.flow_from_directory(train,
                                                        shuffle = True,
                                                        target_size = (128,128),
                                                        batch_size = batch_size,
                                                        class_mode = 'binary')

    val_generator = val_datagen.flow_from_directory(val,
                                                        shuffle = False,
                                                        target_size = (128,128),
                                                        batch_size = batch_size,
                                                        class_mode = 'binary')
    
    test_generator = test_datagen.flow_from_directory(test,
                                                    shuffle = False,
                                                    target_size = (128,128),
                                                    batch_size = batch_size,
                                                    class_mode = 'binary')
    
    return train_generator_augm, val_generator, test_generator

def create_fit_model(train, val):
    # Create the Xception model
    
    print('<<<<<<Downloading the pretrained model>>>>>>')
    base_model = Xception(weights='imagenet',
                    include_top=False,
                    input_shape=(128,128,3))
    print('<<<<<<Finish download>>>>>>')
    
    base_model.trainable = False

    input = layers.Input(shape=(128,128,3))

    x = base_model(input, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu')(x)

    x = layers.Dropout(0.2)(x)

    output = layers.Dense(1, activation='sigmoid')(x)

    model_transfer = keras.Model(input, output)

    # 优化
    opt = keras.optimizers.Adam(learning_rate = 0.001)

    loss = keras.losses.BinaryCrossentropy()

    metrics = ['Recall',
               keras.metrics.AUC(curve="PR", name = 'pr_auc')]

    model_transfer.compile(optimizer=opt,
                loss=loss,
                metrics=metrics)

    early_stopping = EarlyStopping(monitor= 'val_pr_auc', mode = 'max', min_delta=0.001, patience= 5)

    # Create checkpoint
    checkpoint = keras.callbacks.ModelCheckpoint(
        '/root/autodl-tmp/capstone/model/xception_v3_best.keras',
        save_best_only=True,
        monitor = 'val_pr_auc',
        mode='max'
    )
    
    print('Fitting...❗️')
    start = time.time()
    history = model_transfer.fit(
        # augmentation
        train,
        epochs = 20,
        validation_data = val,
        verbose = 1,
        callbacks=[early_stopping, checkpoint]
    )
    print('Finished✅')
    end = time.time() 
    print('Model is saved✅')
    print('⏰Time used (minutes): ', (end-start) / 60)
    best_model = keras.models.load_model('xception_v3_best.keras')
    
    return best_model

def test_evaluation(model ,test):
    test.reset()
    test_results = model.evaluate(test, verbose = 1)
    print("Test Result:")
    print(f"Loss: {test_results[0]:.3f}")
    print(f"Recall: {test_results[1]:.3f}")
    print(f"PR AUC: {test_results[2]:.3f}")

def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    model_tflite = converter.convert()

    with open('/root/autodl-tmp/capstone/model/best_model.tflite', 'wb') as f_out:
        f_out.write(model_tflite)
    
def main():
    path = "/root/autodl-tmp/fire_data"
    set_global_determinism(seed=42)
    train_dir, val_dir, test_dir = img_dataset_path(path)
    train_generator_augm, val_generator, test_generator = img_load_process(train_dir, val_dir, test_dir)
    model = create_fit_model(train_generator_augm, val_generator)
    # model = keras.models.load_model('/root/autodl-tmp/capstone/model/xception_v3_best.keras')
    test_evaluation(model, test_generator)
    convert_to_tflite(model)
    
if __name__ == "__main__":
    main()