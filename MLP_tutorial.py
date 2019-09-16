import numpy as np
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(5)

# 데이터셋 로드
dataset = np.loadtxt("C:\\Users\\dlgus\Documents\\learn_keras\\resource\\pima-indians-diabetes.csv", delimiter=",")

# 훈련, 검증, 평가 변수 생성
x_train = dataset[:700, 0:8]
y_train = dataset[:700, 8]
x_val = dataset[600:700, 0:8]
y_val = dataset[600:700, 8]
x_test = dataset[700:, 0:8]
y_test = dataset[700:, 8]

# 모델 구성
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 학습과정 설정
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
model.fit(x_train, y_train, epochs=1500, batch_size=64, validation_data=(x_val, y_val))

# 모델 평가
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))