from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
model1 = load_model('mnist_model.h5',compile=False)
print(model1.summary())
img = cv2.imread('uploads/image-4328.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(28,28))
plt.imshow(img)
plt.show()
img = img.reshape(-1, 28, 28, 1)
img = img/255.0
print(img.shape)
predict = model1.predict(img)
print(predict)
print(predict.argmax())
