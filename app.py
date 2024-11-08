from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definir las transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10)
])

# Definir el Dataset personalizado
class ElectrodomesticosDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = os.listdir(root_dir)
        
        # Cargar imágenes y etiquetas
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_path):
                self.images.append(os.path.join(class_path, image_name))
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# Configurar el modelo
num_classes = 2  # Ejemplo: bueno/malo
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Función de entrenamiento
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Época {epoch+1}, Loss: {running_loss/len(train_loader)}')
        
        # Guardar el modelo después de cada época
        torch.save(model.state_dict(), f'modelo_epoca_{epoch+1}.pth')

def procesar_imagen(ruta_imagen):
    # Cargar y transformar la imagen
    imagen = Image.open(ruta_imagen).convert('RGB')
    imagen = transform(imagen).unsqueeze(0).to(device)
    
    # Realizar la predicción
    model.eval()
    with torch.no_grad():
        salida = model(imagen)
        _, prediccion = torch.max(salida, 1)
    
    # Convertir predicción a etiqueta
    etiquetas = ['malo', 'bueno']  # Ajusta según tus clases
    return etiquetas[prediccion.item()]

# Configuración de Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No se seleccionó ningún archivo.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Ningún archivo seleccionado.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Procesar la imagen y obtener predicción
        resultado = procesar_imagen(filepath)
        return jsonify({
            'message': 'Imagen procesada correctamente.',
            'resultado': resultado,
            'filename': filename
        }), 200

    return jsonify({'error': 'Tipo de archivo no permitido.'}), 400

# Ruta para iniciar el entrenamiento
@app.route('/entrenar', methods=['POST'])
def iniciar_entrenamiento():
    try:
        # Crear dataset y dataloader
        train_dataset = ElectrodomesticosDataset(
            root_dir='datos_entrenamiento',  # Carpeta con las imágenes de entrenamiento
            transform=transform
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Entrenar el modelo
        train_model(model, train_loader, criterion, optimizer)
        
        return jsonify({'message': 'Entrenamiento completado con éxito'}), 200
    except Exception as e:
        return jsonify({'error': f'Error durante el entrenamiento: {str(e)}'}), 500

if __name__ == '__main__':
    # Crear la carpeta de uploads si no existe
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Crear la carpeta de datos de entrenamiento si no existe
    os.makedirs('datos_entrenamiento', exist_ok=True)
    
    app.run(debug=True)