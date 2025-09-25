from flask import Flask, redirect, request, jsonify
from keras_image_helper import create_preprocessor
from flasgger import Swagger
import tflite_runtime.interpreter as tflite
from PIL import Image
import time
import os
import requests
from datetime import datetime
import numpy as np

# 配置常量
PORT = int(os.environ.get('PORT', 9696))

class TokenBucket:
    """令牌桶算法实现限流"""
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # 每秒补充的令牌数
        self.last_refill = time.time()

    def refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now

    def take_token(self):
        """获取令牌"""
        self.refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


# 初始化令牌桶
bucket = TokenBucket(capacity=10, refill_rate=3)

def predict_wildfire(url):
    # 下载图像
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Failed to retrieve the image. Status code:", response.status_code)
        return None

    # 临时文件处理
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/tmp/downloaded_image_{timestamp}.jpg"  # 使用/tmp目录

    with open(filename, "wb") as file:
        file.write(response.content)
    # print(f"[DEBUG] Image saved to {filename}")

    # 验证图像有效性
    try:
        Image.open(filename).verify()
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

    # 加载模型
    interpreter = tflite.Interpreter(model_path='model/best_model.tflite')
    interpreter.allocate_tensors()
    # print("[DEBUG] Model loaded successfully")

    # 图像预处理
    preprocessor = create_preprocessor('xception', target_size=(128,128))
    X = preprocessor.from_path(filename)
    # print("[DEBUG] Preprocessing completed")
    
    # 获取输入输出张量信息
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 模型推理
    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 清理临时文件
    # os.remove(filename)
    return dict(zip( ['probability'], output_data[0].tolist() ))


# Flask应用初始化
app = Flask(__name__)
swagger = Swagger(app)


@app.before_request
def rate_limiter():
    """限流中间件"""
    if not bucket.take_token():
        return jsonify({"detail": "Rate limit exceeded"}), 429


@app.route("/")
def index():
    return redirect("/apidocs/")


@app.route("/predict", methods=['POST'])
def predict():
 
    # 字典
    data = request.json
    # 从字典中获取对应key的value
    query = data.get('query')
    
    if not query:
        return jsonify({"detail": "Query parameter is required"}), 400
    
    try:
        answer = predict_wildfire(query)
        return jsonify(answer), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)
