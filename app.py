import os
from flask import Flask, render_template, request

app = Flask(__name__)

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
    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    # Ваш код для обработки файла (поиск, анализ и т.д.)

    # Пример использования нейронной сети (заглушка, замените на свой код)
    prediction = predict_image(file_path)

    return render_template('index.html', filename=file.filename, prediction=prediction)

# Пример заглушки для использования нейронной сети (замените на свой код)
def predict_image(file_path):
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    img = Image.open(file_path).resize((224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions.numpy())
    return decoded_predictions[0][0][1]

if __name__ == '__main__':
    app.run(debug=True)
