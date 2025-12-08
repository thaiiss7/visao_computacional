# import cv2
# import os

# f = './Data/Test'
# folders = [(os.path.join(f, nome), os.listdir((os.path.join(f, nome)))[0]) for nome in os.listdir(f)]
# for folder, arc in folders:
#     img = cv2.imread(folder + '/' + arc)
#     cv2.imwrite('./SimpleTest/' + arc.split('_')[1][0] + '.png', img)


import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import utils

images = [os.path.join('.\\SimpleTest', img) for img in os.listdir('.\\SimpleTest')]
model = load_model('model.keras')

for image in images:
    img = image
    img = utils.load_img(img, target_size=(28, 28))

    img_array = utils.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)

    predict_value = os.path.splitext(os.path.basename(images[np.argmax(pred)]))[0]
    real_value = os.path.splitext(os.path.basename(image))[0]
    print('Predict: ', predict_value, 'Real: ', real_value, '\t-> ', predict_value == real_value) 
    
