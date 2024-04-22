import numpy as np
import pickle
import tensorflow.lite as tflite


# Load the trained model file
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
tokenizer = None
with open("tokenizer", "rb") as f:
    tokenizer = pickle.load(f)
while True:
    sequences = [tokenizer.texts_to_sequences(
        [input(">")])]

    interpreter.set_tensor(interpreter.get_input_details()[
        0]['index'], np.array(
        [
            [
                [[
                    float(z)
                    for z in y
                ][0]]
                for y in x
            ]
            for x in sequences
        ], dtype="float32")[0])
    interpreter.invoke()
    prediction = np.argmax(interpreter.get_tensor(
        interpreter.get_output_details()[0]['index'])[0])
    print("positive" if prediction else "negative")
