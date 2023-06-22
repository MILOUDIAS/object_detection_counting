from PIL import Image
import tflite_runtime.interpreter as tflite

def make_interpreter(model_file):
    return tflite.Interpreter(model_path=model_file)

def set_input(interpreter, image, resample=Image.NEAREST):
    """Copies data to input tensor."""
    image = image.resize((input_image_size(interpreter)[0:2]), resample)
    input_tensor(interpreter)[:, :] = image

def input_image_size(interpreter):
    """Returns input image size as (width, height, channels) tuple."""
    _, height, width, channels = interpreter.get_input_details()[0]['shape']
    return width, height, channels

def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]

def load_labels(path):

  with open(path, 'r') as f:
    labels = f.readlines()

  labels = [label.strip() for label in labels]

  return labels


import os
def load_model(model_dir,model, lbl):
    
    print('Loading from directory: {} '.format(model_dir))
    print('Loading Model: {} '.format(model))
    print('Loading Labels: {} '.format(lbl))
    
    model_path=os.path.join(model_dir,model)
    labels_path=os.path.join(model_dir,lbl)
    
    interpreter = tflite.Interpreter(model_path)
    
    interpreter.allocate_tensors()
    
    labels = load_labels(labels_path)
    
    return interpreter, labels
    

