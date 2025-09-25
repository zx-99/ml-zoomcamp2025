import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_recall_curve, classification_report, auc
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    test_dir = "/root/autodl-tmp/fire_data/test"
    test_datagen = ImageDataGenerator(rescale = 1./255)

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                        shuffle = False,
                                                        target_size = (128,128),
                                                        batch_size = 32,
                                                        class_mode = 'binary')
    model = keras.models.load_model('/root/autodl-tmp/capstone/model/xception_v3_best.keras')
    
    test_generator.reset()
    test_results = model.evaluate(test_generator, verbose = 1)

    print("\nTest Result:")
    print(f"Loss: {test_results[0]:.3f}")
    print(f"Recall: {test_results[1]:.3f}")
    print(f"PR AUC: {test_results[2]:.3f}")
    
    #按照生成器的数据顺序获取模型的预测结果 (probability)
    y_pred_probs = model.predict(test_generator).flatten()

    # 获取真实标签
    y_true = test_generator.classes
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
    
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold based on PR curve: {optimal_threshold:.3f}")
    
    y_pred_classes = (y_pred_probs > optimal_threshold).astype(int)
    
    report = classification_report(y_true, y_pred_classes, target_names=["No Wildfire", "Wildfire"])
    print("\nClassification Report:")
    print(report)
    
if __name__ == "__main__":
    main()