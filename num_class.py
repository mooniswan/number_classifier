import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tkinter import *
from PIL import Image, ImageDraw, ImageOps

# MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0

# 라벨 이진화
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# 신경망 모델 정의
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Tkinter 설정
class App:
    def __init__(self, master):
        self.master = master
        self.master.title("MNIST Digit Classifier")
        
        self.canvas = Canvas(master, width=200, height=200, bg="white")
        self.canvas.pack()
        
        self.button_frame = Frame(master)
        self.button_frame.pack()
        
        self.clear_button = Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=LEFT)
        
        self.predict_button = Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=RIGHT)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<ButtonRelease-1>", self.save_image)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        r = 5  # 반지름
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill="black")

    def save_image(self, event):
        self.image.save("digit.png")

    def predict_digit(self):
        self.image = self.image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(self.image)
        img_array = 255 - img_array  
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1).astype('float32') 
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        self.show_prediction(predicted_digit)

    def show_prediction(self, digit):
        result_window = Toplevel(self.master)
        result_window.title("Prediction Result")
        Label(result_window, text=f"Predicted Digit: {digit}", font=("Helvetica", 24)).pack()

root = Tk()
app = App(root)
root.mainloop()