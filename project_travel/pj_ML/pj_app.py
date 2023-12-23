# # index 화면에서 파일 선택 후 예측 버튼을 클릭하면
# # 어떤 이미지 인지 분류하여 화면에 출력
# from flask import Flask, render_template, request, jsonify
# # import tensorflow as tf
# from PIL import Image
# import numpy as np
# import joblib
# import os
# from werkzeug.utils import secure_filename
# model = joblib.load('decision_tree_model.pkl')
# app = Flask(__name__)
# # 모델 불러오기
# # model = tf.keras.models.load_model('decision_tree_model.pkl')
# labels=['VISIT_TYPE','MOVING','GENDER','AGE','MARR','자연_도시','숙박_당일','비싼숙소_저렴한숙소','휴양OR휴식_체험','숨은여행지_유명여행지','비촬영여행지_사진촬영여행지','동반자수', 'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2', 'TRAVEL_MOTIVE_3', 'TRAVEL_COMPANIONS_NUM']
#
# @app.route('/')
# def index():
#     return render_template('index.html')
# @app.route('/upload',methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return jsonify({'error':'no file upload'}),400
#     f=request.files['image']
#     if f.filename=='':
#         return jsonify({'error':'no file selected'}),400
#     image=Image.open(f)
#     image=image.resize((224,224))
#     image_array=np.array(image)
#     # resahpe 함수와 같이
#     image_array=image_array.reshap(1,*image_array.shape)
#    # image_array=image_array[tf.newaxis, ...]
#     pred=model.predict(image_array)
#     pred_class=np.argmax(pred,axis=-1)[0]
#     return jsonify({'prediction': labels[pred_class]})
#
# if __name__ == '__main__':
#     app.run(debug=True)

######################################################
###################################################
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
app = Flask(__name__)

# Load the Decision Tree model
model = joblib.load('decision_tree_model.pkl')
@app.route('/')
def index():
    return render_template('index2.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Ensure that the incoming data matches the model's input features
        if set(data.keys()) == {'VISIT_TYPE', 'MOVING', 'GENDER', 'AGE', 'MARR', '자연_도시', '숙박_당일', '비싼숙소_저렴한숙소', '휴양OR휴식_체험', '숨은여행지_유명여행지', '비촬영여행지_사진촬영여행지', '동반자수', 'TRAVEL_MOTIVE_1', 'TRAVEL_COMPANIONS_NUM'}:
            input_data = pd.DataFrame(data, index=[0])
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)
            response = {
                'prediction': prediction[0],
                'probabilities': probability.tolist()
            }
            return jsonify(response)
        else:
            return jsonify({'error': 'Invalid input data format'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='192.168.0.22')
