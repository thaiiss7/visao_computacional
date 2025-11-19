from draw_network import Network
from sklearn.model_selection import train_test_split
import numpy as np

train_val_images = './DATASET/train_images.npy'
train_val_labels = './DATASET/train_labels.npy' 

x = np.load(train_val_images)
y = np.load(train_val_labels)

x = x.reshape(x.shape[0], -1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

Net = Network(outputs=10, features=784, lr=0.05, neuron_layers=[16,8], line_limit=28, scaler=255)
Net.fit(x, y, 1)