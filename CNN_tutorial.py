import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# random seed 고정
np.random.seed(3)

# 훈련용 제너레이터
train_datagen = ImageDataGenerator(rescale=1. / 255)

# 이미지 경로, 이미지 크기(자동 리사이징), 배치 크기, 분류 방식
train_generator = train_datagen.flow_from_directory(
    '/home/hyeonjulee/PycharmProjects/learn_keras/resource/handwriting_shape/train',
    target_size=(24, 24),
    batch_size=3,
    class_mode='categorical'
)

# 검증용 제너레이터
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 이미지 경로, 이미지 크기(자동 리사이징), 배치 크기, 분류 방식
test_generator = train_datagen.flow_from_directory(
    '/home/hyeonjulee/PycharmProjects/learn_keras/resource/handwriting_shape/test',
    target_size=(24, 24),
    batch_size=3,
    class_mode='categorical'
)

model = Sequential()
# 필터 수 32개, 필터 크기 3 x 3, 활성화 함수 relu, 입력 이미지 크기 24 x 24, 입력 이미지 채널 3개,
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 3)))
# 필터 수 64개, 필터 크기 3 x 3, 활성화 함수 relu
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=15, epochs=50,
                    validation_data=test_generator, validation_steps=5)

print("-----평가-----")
scores = model.evaluate_generator(test_generator, steps=5)
print("{}: {}".format(model.metrics_names[1],scores[1]*100))

print("-----예측-----")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)