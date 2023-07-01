from keras.models import Model
from keras.models import model_from_json
import tensorflow as tf
from main import X_testcnn, Y_test
from keras.models import Sequential
import keras

# loading json and creating model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                     optimizer="adam", metrics=['accuracy'])
score = loaded_model.evaluate(X_testcnn, Y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
