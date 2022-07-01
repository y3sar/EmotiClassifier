import torch
import gradio as gr
from model import EmotiClassifier

predictor = EmotiClassifier()

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

predictor.load_state_dict(torch.load('emoticlassifier-64acc-1_250loss.pth'))

def classify(image):
    
    torch_image = torch.Tensor(image)
    torch_image = torch_image.view(1, 1, torch_image.shape[0], torch_image.shape[1])
    
    
    pred = predictor(torch_image)
    
    label = torch.argmax(pred)
    
    pred_class = label.item()

    return labels[pred_class]

webcam = gr.Image(source='webcam',  shape=(48, 48), image_mode='L')

interface = gr.Interface(fn=classify, inputs=webcam, outputs='text', title="Emotion Classifier", description="<h4>Smiling is a natural anti depressant.\n Smile at this AI and see if it can recognize your emotions.</br> Put your face close to your camera so that your face takes up the whole screen for better result. </br> Click the camera button to take a picture and then submit. </br> Be gentle with this AI when its wrong it is still a new born 😁</h4>" )
interface.launch();
