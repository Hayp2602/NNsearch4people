import os
import cv2
import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Создаем папку "static/uploads", если она не существует
upload_folder = 'static/uploads'
os.makedirs(upload_folder, exist_ok=True)

# Отображение главной страницы
@app.route('/')
def index():
    return render_template('index.html')

# Обработка загрузки файла
@app.route('/upload', methods=['POST'])
def upload():
    # Проверяем, был ли файл отправлен в запросе
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']

    # Проверяем, был ли выбран файл
    if file.filename == '':
        return redirect('/')

    # Сохраняем файл в папку static/uploads
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Ваш код для обработки файла (поиск, анализ и т.д.)
    prediction = predict_image(file_path)

    return render_template('index.html', filename=file.filename, prediction=prediction)

def predict_image(file_path):
    model = torch.hub.load('yoltv5/yoltv5/yolov5/','custom', path='model/best.pt', source='local')
    frame = cv2.imread(file_path)
    detections = model(frame[..., ::-1])
    print(detections)
    results = detections.pandas().xyxy[0].to_dict(orient="records")
    for result in results:
        con = result['confidence']
        cs = result['class']
        x1 = int(result['xmin'])
        y1 = int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = int(result['ymax'])
        # Рисуем прямоугольник на изображении
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Сохраняем изображение с прямоугольниками
    cv2.imwrite(file_path, frame)

    if len(results) > 0:
        return "Люди обнаружены"
    else:
        return "Люди не обнаружены"

if __name__ == '__main__':
    app.run(debug=True)