from flask import Flask, request, jsonify
#from multiprocessing import Process, Value
from threading import Thread, Event
import time
import tensorflow as tf
import numpy as np
import pathlib
import os
from PIL import Image
import queue

def run_flask(model_queue):
    app = Flask(__name__)
 
    UPLOAD_FOLDER = '/home/pi/funcx-hierarchy-image-testing/animal_model_deployment'
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', '.tflite'}

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    @app.route('/update_model', methods = ["POST"])
    def update_model():
        #data = request.values
        file = request.files['file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        #print(data)
        print(file)
        
        model_queue.put('putting sign into queue')
        
        print(f'FLASK: {model_queue.empty()}')
        # get the model
        # reload it

        return 'model updated'

    @app.route("/test", methods=["GET"])
    def test():
        #print("Successfully received a GET request!")
        print(model_queue.empty())
        return f"queue status: {model_queue.empty()}"
    
    app.run(host='0.0.0.0', port=8090, debug=True, use_reloader=False)


def load_model():
    # Specify the TensorFlow model and labels
    script_dir = pathlib.Path(__file__).parent.absolute()
    model_file = os.path.join(script_dir, 'animal_model.tflite')
    #label_file = os.path.join(script_dir, 'human_labels.txt')
    labels = ['bear', 'elephant', 'leopard', 'lion', 'wolf']

    # Initialize the TF interpreter
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('model loaded')
    return interpreter, input_details, output_details, labels

def model_runner(img, interpreter, input_details, output_details, labels):
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Print the result
    classification = np.argmax(output_data[0])
    score = np.max(output_data[0])
    label = labels[classification]
    string_response = f"{label}:{score}"

    return string_response

def process_image(img, image_x_size=224, image_y_size=224, depth=3):
    img = img.resize((image_x_size, image_y_size))
    img = np.asarray(img)
    img = img / 255
    img = np.array([img], dtype='float32')
    return img

def model_process(model_queue):

    
    interpreter, input_details, output_details, labels = load_model()
    print('loaded the model')
    while True:
        print(model_queue.empty())
        if not model_queue.empty():
            popped = model_queue.get()
            print(popped)
            print(model_queue.empty())
            interpreter, input_details, output_details, labels = load_model()
            

        img = Image.open('test.jpg')
        img = process_image(img)
        string_response = model_runner(img, interpreter, input_details, output_details, labels)
        print(string_response)
        time.sleep(5)

if __name__ == "__main__":
    model_queue = queue.Queue()
    #recording_on = Value('b', True)
    p1 = Thread(target=model_process, args=(model_queue,))
    p2 = Thread(target=run_flask, args=(model_queue,))

    p1.start()  
    p2.start()
    
    p1.join()
    p2.join()
    
    while True:
        try:
            sleep(1)
        except KeyboardInterrupt:
            event.set()
            break
    event.set()
