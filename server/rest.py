import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from mistralai.client import MistralClient

MISTRAL_KEY = 'i5XyYjVEp3LT7zvEVIva3K2mSTHXMqjb'
MISTRAL_CLIENT = MistralClient(api_key=MISTRAL_KEY)

# Initialize Flask app
app = Flask(__name__)


class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)  # Fully connected layer 1
        self.dropout1 = nn.Dropout(0.8)  # Dropout layer with 20% probability
        self.fc2 = nn.Linear(512, 256)  # Fully connected layer 2
        self.dropout2 = nn.Dropout(0.8)  # Dropout layer with 20% probability
        self.fc3 = nn.Linear(256, 1)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first fully connected layer
        x = self.dropout1(x)  # Apply dropout to the output of the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second fully connected layer
        x = self.dropout2(x)  # Apply dropout to the output of the second layer
        x = torch.sigmoid(self.fc3(x))  # Apply sigmoid activation to the output layer for binary classification
        return x


# Load your PyTorch model
model = BinaryClassifier(input_size=2048)
model.load_state_dict(torch.load(r'model/model_final.pt', map_location=torch.device('cpu')))
model.eval()


def get_embeddings(text):
    try:
        embed = MISTRAL_CLIENT.embeddings(model='mistral-embed', input=text[:20000]).data[0].embedding
    except:
        embed = MISTRAL_CLIENT.embeddings(model='mistral-embed', input=text[:15000]).data[0].embedding

    return embed


# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the JSON data from the request
    emb1 = get_embeddings(data['text1'])
    emb2 = get_embeddings(data['text2'])

    # Assuming the model requires both inputs concatenated or as a tuple
    combined_input = torch.tensor([emb1 + emb2])  # Adjust based on your model's requirements

    with torch.no_grad():
        output = model(combined_input)

    result = output.tolist()[0]  # Convert output to list for JSON serialization
    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)
