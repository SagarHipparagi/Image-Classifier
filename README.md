# AI Image Classifier

An AI-powered image classification web application using TensorFlow's MobileNetV2 model, deployed on Vercel.

## Features

- 🖼️ Image upload interface with drag-and-drop support
- 🤖 Real-time image classification using MobileNetV2
- 📊 Top 3 predictions with confidence scores
- 🚀 Serverless deployment ready for Vercel

## Local Development

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository or navigate to the project folder:
```bash
cd "Image classifier"
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Flask application:
```bash
python api/index.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Deployment on Vercel

### Prerequisites
- Vercel account (free at https://vercel.com)
- Git repository

### Steps

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo>
git push -u origin main
```

2. **Deploy to Vercel**
   - Go to https://vercel.com
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will automatically detect it as a Python project
   - Click "Deploy"

3. **Done!** Your image classifier is now live at `https://your-project.vercel.app`

## How to Use

1. Open the application in your browser
2. Click the upload area or drag and drop an image
3. Click "Classify Image" button
4. View the top 3 predictions with confidence scores

## API Endpoints

### GET /
Returns the web interface

### POST /api/classify
Classifies an uploaded image

**Request:**
- Form data with `image` file

**Response:**
```json
{
  "status": "success",
  "predictions": [
    {
      "label": "pug",
      "confidence": "95.23%"
    },
    {
      "label": "pug_dog",
      "confidence": "3.21%"
    },
    {
      "label": "bulldog",
      "confidence": "1.23%"
    }
  ]
}
```

## Technical Stack

- **Backend:** Flask (Python)
- **Model:** TensorFlow MobileNetV2
- **Frontend:** Vanilla JavaScript, HTML, CSS
- **Deployment:** Vercel Serverless Functions

## Notes

- First request may take a few seconds as the model loads
- Images are processed to 224x224 pixels for the model
- The model uses ImageNet pre-trained weights
- Maximum file size handled by Vercel: 25MB