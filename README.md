# 🌾 Smart Agriculture Assistant

An AI-powered multilingual farming advisor with voice support, document processing, and real-time weather integration.

### 🏗️ Architecture
![architecture](https://github.com/user-attachments/assets/a96a25d0-4cd3-4f3c-8f08-959a955f5cfe)
--
### 🚀 How to Run the Streamlit App

1️⃣ Install all required libraries
- pip install -r requirements.txt
 
2️⃣ Open 'app.py' and replace the placeholders:
-    Replace ➤ "Replace with your actual NVIDIA API key"
-         with ➤ your actual NVIDIA API key
-    Replace ➤ "Replace with your actual OpenWeatherMap API key"
-         with ➤ your actual OpenWeatherMap API key

- NVIDIA_API_KEY = "your_nvidia_api_key_here"
- WEATHER_API_KEY = "your_openweather_api_key_here"
  
3️⃣ Launch the app
- streamlit run app.py
--
## ✨ Features

### 🌍 Multilingual Support
- **English** 🇺🇸
- **Hindi** 🇮🇳 (हिंदी)
- **Gujarati** 🇮🇳 (ગુજરાતી)

### 🤖 AI-Powered Agents
- **Weather Agent**: Real-time weather data and forecasts
- **Agriculture Agent**: Crop recommendations and farming advice
- **PDF Agent**: Document analysis and knowledge extraction
- **Combined Agent**: Weather-based agricultural recommendations

### 🎤 Voice Assistant
- Voice input in multiple languages
- Text-to-speech output
- Hands-free interaction for farmers

### 📄 Document Processing
- PDF upload and text extraction
- Vector database storage using ChromaDB
- Document-based Q&A system
- Automatic summarization and insights

### 🌤️ Weather Integration
- Real-time weather data from OpenWeatherMap
- Location-based weather queries
- Weather-informed crop recommendations
- Quick access to major city weather

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: NVIDIA AI Endpoints (Llama3-8B), NVIDIAEmbeddings
- **Vector Database**: ChromaDB
- **Document Processing**: PyMuPDF (fitz)
- **Weather API**: OpenWeatherMap
- **Voice**: SpeechRecognition, pyttsx3
- **Text Processing**: LangChain

## 🚀 Usage

### Basic Chat
1. Select your preferred language from the dropdown
2. Type your agriculture-related question
3. Get AI-powered responses with relevant information

### Voice Interaction
1. Enable voice in the settings
2. Click "Start Recording" to speak your query
3. The system will process your speech and respond with voice output

### Document Upload
1. Upload PDF documents (max 10MB)
2. The system automatically extracts and indexes content
3. Ask questions about the uploaded documents

### Weather Queries
Ask weather-related questions like:
- "What's the weather in Mumbai?"
- "Weather forecast for Delhi"
- "Current temperature in Gandhinagar"

### Combined Queries
Get weather-informed agricultural advice:
- "What crops are suitable for Delhi in current weather?"
- "Should I plant rice in Mumbai this season?"

### Data Flow
1. **Input Processing**: Voice/text input with language detection
2. **Query Classification**: Route to appropriate specialized agent
3. **Context Gathering**: Fetch relevant data (weather, documents)
4. **AI Processing**: Generate contextual responses using NVIDIA LLM
5. **Output Generation**: Multilingual text and voice responses
