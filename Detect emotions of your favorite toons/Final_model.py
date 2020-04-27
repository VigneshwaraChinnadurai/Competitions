    
import keras 
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()  
model.add(Conv2D(16, (3, 3), input_shape = (128, 128, 3),kernel_initializer='normal', activation='relu'))
model.add(Conv2D(16, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(32, (3, 3), border_mode='valid',activation='relu'))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), border_mode='valid',activation='relu'))
model.add(Conv2D(128, (3, 3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), border_mode='valid',activation='relu'))
model.add(Conv2D(128, (3, 3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))

ada = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
sgd = keras.optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="mean_squared_error", optimizer=sgd, metrics=['accuracy'])
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(shear_range = 0.2,
                                   channel_shift_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range=10,
                                   validation_split=0.9,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(validation_split=0.9)

training_set = train_datagen.flow_from_directory('Train',
                                                 target_size = (128, 128),
                                                 batch_size = 16,
                                                 shuffle=True,
                                                 seed=101,
                                                 #save_to_dir='Augumented/Train',
                                                 #save_format='jpeg',
                                                 interpolation='nearest',
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Test',
                                            target_size = (128, 128),
                                            batch_size = 16,
                                            shuffle=True,
                                            seed=101,
                                            #save_to_dir='Augumented/Test',
                                            #save_format='jpeg',
                                            interpolation='nearest',
                                            class_mode = 'categorical')

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)

model.fit_generator(training_set,
                    steps_per_epoch = 228,
                    epochs = 100,
                    validation_data = test_set,
                    validation_steps = 70,
                    class_weight={0:5,4:9,2:15,3:15,1:15},
                    callbacks=[es,mc])

from keras.models import load_model
saved_model = load_model('best_model.h5')

from sklearn.externals import joblib
joblib.dump(model, 'Detect_emotions_of_your_favorite_cartoon.joblib')  

d=training_set.class_indices
d= {v:k for k,v in d.items()}   


import numpy as np
import pandas as pd
from keras.preprocessing import image
test_predict=pd.read_csv('Test.csv')
for n in test_predict['Frame_ID']:
    test_image = image.load_img('predict/'+n, target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = saved_model.predict(test_image)
    training_set.class_indices
    if result[0][0] >= 0.5:
        prediction = 'Unknown'
    elif result[0][1] >= 0.5:
        prediction = 'angry'
    elif result[0][2] >= 0.5:
        prediction = 'happy'
    elif result[0][3] >= 0.5:
        prediction = 'sad'
    else:
        prediction = 'surprised'
    test_predict.loc[test_predict['Frame_ID']==n,'Results']=prediction
    
test_predict.to_csv(r'Test_final.csv')