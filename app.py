import streamlit as st
import requests
import chromadb
import uuid
import fitz  
import re
import os
import logging
import pyttsx3
import threading
import tempfile
import base64
from io import BytesIO
import time
import speech_recognition as sr
from datetime import datetime, timedelta
from chromadb.api.types import EmbeddingFunction
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "your_nvidia_api_key_here")  # Replace with your actual NVIDIA API key
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "your_openweathermap_api_key_here")  # Replace with your actual OpenWeatherMap API key

# Language configurations
LANGUAGES = {
    'english': {
        'name': 'English',
        'code': 'en',
        'speech_code': 'en-US',
        'tts_voice': 'english',
        'flag': '🇺🇸'
    },
    'hindi': {
        'name': 'हिंदी',
        'code': 'hi',
        'speech_code': 'hi-IN',
        'tts_voice': 'hindi',
        'flag': '🇮🇳'
    },
    'gujarati': {
        'name': 'ગુજરાતી',
        'code': 'gu',
        'speech_code': 'gu-IN',
        'tts_voice': 'gujarati',
        'flag': '🇮🇳'
    }
}

# Initialize NVIDIA LLM and Embeddings with error handling
try:
    llm = ChatNVIDIA(model="meta/llama3-8b-instruct", api_key=NVIDIA_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize NVIDIA LLM: {str(e)}")
    st.stop()

class CustomNVIDIAEmbeddingFunction(EmbeddingFunction):
    def __init__(self, nvidia_embeddings):
        self.nvidia_embeddings = nvidia_embeddings

    def __call__(self, input):
        return self.nvidia_embeddings.embed_documents(input)

try:
    nvidia_embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", api_key=NVIDIA_API_KEY)
    embedding_function = CustomNVIDIAEmbeddingFunction(nvidia_embeddings)
    chroma_client = chromadb.Client()
except Exception as e:
    st.error(f"Failed to initialize embeddings: {str(e)}")
    st.stop()

# Initialize collections
try:
    collection = chroma_client.get_collection(name="agri_docs", embedding_function=embedding_function)
except:
    collection = chroma_client.create_collection(name="agri_docs", embedding_function=embedding_function)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'weather_cache' not in st.session_state:
    st.session_state.weather_cache = {}
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = []
if 'pdf_content' not in st.session_state:
    st.session_state.pdf_content = ""
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = 'english'
if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = False
if 'listening' not in st.session_state:
    st.session_state.listening = False
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

class VoiceAssistant:
    """Handles voice input/output in multiple languages"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = None
        self.initialize_tts()
    
    def initialize_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            # Set properties
            self.tts_engine.setProperty('rate', 150)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.9)  # Volume level
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            self.tts_engine = None
    
    def set_voice_language(self, language_code):
        """Set TTS voice based on language"""
        if not self.tts_engine:
            return
        
        try:
            voices = self.tts_engine.getProperty('voices')
            
            # Language-specific voice selection
            voice_keywords = {
                'hindi': ['hindi', 'hi', 'india'],
                'gujarati': ['gujarati', 'gu', 'india'],
                'english': ['english', 'en', 'us', 'uk']
            }
            
            selected_voice = None
            keywords = voice_keywords.get(language_code, ['english'])
            
            for voice in voices:
                voice_name = voice.name.lower()
                if any(keyword in voice_name for keyword in keywords):
                    selected_voice = voice.id
                    break
            
            if selected_voice:
                self.tts_engine.setProperty('voice', selected_voice)
            else:
                # Use default voice if specific language not found
                self.tts_engine.setProperty('voice', voices[0].id)
                
        except Exception as e:
            logger.error(f"Voice setting failed: {e}")
    
    def listen_for_speech(self, language_code='en-US', timeout=5):
        """Listen for speech input"""
        try:
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                
                # Recognize speech
                text = self.recognizer.recognize_google(audio, language=language_code)
                return text
                
        except sr.WaitTimeoutError:
            return "timeout"
        except sr.UnknownValueError:
            return "unclear"
        except sr.RequestError as e:
            return f"error: {str(e)}"
        except Exception as e:
            return f"error: {str(e)}"
    
    def speak_text(self, text, language_code='english'):
        """Convert text to speech"""
        if not self.tts_engine:
            return False
        
        try:
            # Set voice for language
            self.set_voice_language(language_code)
            
            # Speak the text
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return True
            
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return False

class MultilingualTranslator:
    """Handles translation and multilingual responses"""
    
    @staticmethod
    def get_system_prompt(language_code):
        """Get system prompt in specified language"""
        prompts = {
            'english': "You are a helpful agriculture assistant. Provide practical farming advice in English.",
            'hindi': "आप एक सहायक कृषि सलाहकार हैं। हिंदी में व्यावहारिक खेती की सलाह दें।",
            'gujarati': "તમે એક સહાયક કૃષિ સલાહકાર છો. ગુજરાતીમાં વ્યાવહારિક ખેતીની સલાહ આપો."
        }
        return prompts.get(language_code, prompts['english'])

# Initialize voice assistant
voice_assistant = VoiceAssistant()
translator = MultilingualTranslator()

class AgentRouter:
    """Routes queries to appropriate specialized agents"""
    
    @staticmethod
    def classify_query(query):
        """Classify query to determine which agent should handle it"""
        query_lower = query.lower()
        
        # Multi-language patterns
        weather_patterns = [
            'weather', 'temperature', 'rain', 'humidity', 'wind', 'climate',
            'forecast', 'today weather', 'current weather', 'hot', 'cold',
            'मौसम', 'तापमान', 'बारिश', 'आर्द्रता', 'हवा', 'जलवायु',
            'હવામાન', 'તાપમાન', 'વરસાદ', 'ભેજ', 'પવન', 'આબોહવા'
        ]
        
        agriculture_patterns = [
            'crop', 'farming', 'agriculture', 'kharif', 'rabi', 'zaid',
            'fertilizer', 'seed', 'harvest', 'soil', 'irrigation', 'fruit',
            'suitable', 'plant', 'grow', 'cultivation', 'sowing', 'season',
            'फसल', 'खेती', 'कृषि', 'खरीफ', 'रबी', 'जायद',
            'ખેતી', 'કૃષિ', 'પાક', 'બીજ', 'માટી', 'સિંચાઈ'
        ]
        
        pdf_patterns = [
            'document', 'pdf', 'file', 'uploaded', 'summarize', 'summary',
            'from the document', 'in the file', 'according to',
            'दस्तावेज़', 'फ़ाइल', 'सारांश',
            'દસ્તાવેજ', 'ફાઇલ', 'સારાંશ'
        ]
        
        has_weather = any(pattern in query_lower for pattern in weather_patterns)
        has_agriculture = any(pattern in query_lower for pattern in agriculture_patterns)
        has_pdf = any(pattern in query_lower for pattern in pdf_patterns)
        
        if has_weather and has_agriculture:
            return 'combined'
        elif has_pdf:
            return 'pdf'
        elif has_weather:
            return 'weather'
        else:
            return 'agriculture'

class WeatherAgent:
    """Specialized agent for weather-related queries"""
    
    @staticmethod
    def extract_location(query):
        """Extract location from query with improved global city detection"""
        query_lower = query.lower()
        
        # Multi-language location patterns
        patterns = [
            r'weather\s+(?:in|of|at)\s+([a-zA-Z\s]+?)(?:\s+what|\s+and|\s+\?|$)',
            r'weather\s+(?:in|of|at)\s+([a-zA-Z\s]+)',
            r'(?:in|at)\s+([a-zA-Z\s]+?)(?:\s+what|\s+and|\s+\?|$)',
            r'(?:in|at)\s+([a-zA-Z\s]+)',
            r'मौसम.*?में\s+([a-zA-Z\s]+)',
            r'હવામાન.*?માં\s+([a-zA-Z\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                location = match.group(1).strip()
                stop_words = ['today', 'tomorrow', 'current', 'now', 'suggest', 'me', 'to', 'which', 
                             'आज', 'कल', 'અાજે', 'કાલે']
                location_words = location.split()
                cleaned_words = [word for word in location_words if word not in stop_words and len(word) > 2]
                
                if cleaned_words:
                    cleaned_location = ' '.join(cleaned_words)
                    if len(cleaned_location) <= 30 and all(word.replace('-', '').replace(' ', '').isalpha() for word in cleaned_location.split()):
                        return cleaned_location.title()
        
        return "Gandhinagar"  # Default location
    
    @staticmethod
    def get_weather_data(location):
        """Fetch weather data for location"""
        if location in st.session_state.weather_cache:
            cached_data = st.session_state.weather_cache[location]
            # Check if cache is still valid (1 hour)
            if datetime.now() - cached_data.get('timestamp', datetime.min) < timedelta(hours=1):
                return cached_data['data']
        
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather_info = {
                'description': data["weather"][0]["description"].capitalize(),
                'temperature': data["main"]["temp"],
                'humidity': data["main"]["humidity"],
                'wind_speed': data["wind"]["speed"],
                'location': data["name"],
                'country': data["sys"]["country"] if "sys" in data and "country" in data["sys"] else ""
            }
            
            # Cache with timestamp
            st.session_state.weather_cache[location] = {
                'data': weather_info,
                'timestamp': datetime.now()
            }
            return weather_info
            
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def format_weather_response(weather_data, language_code='english'):
        """Format weather data into readable response"""
        if 'error' in weather_data:
            error_messages = {
                'english': f"⚠️ Sorry, I couldn't fetch weather data: {weather_data['error']}",
                'hindi': f"⚠️ क्षमा करें, मैं मौसम डेटा नहीं ला सका: {weather_data['error']}",
                'gujarati': f"⚠️ માફ કરશો, હું હવામાન ડેટા લાવી શક્યો નહીં: {weather_data['error']}"
            }
            return error_messages.get(language_code, error_messages['english'])
        
        location_display = weather_data['location']
        if weather_data.get('country'):
            location_display += f", {weather_data['country']}"
        
        templates = {
            'english': f"""🌤 **Weather in {location_display}:**
• Condition: {weather_data['description']}
• Temperature: {weather_data['temperature']}°C
• Humidity: {weather_data['humidity']}%
• Wind Speed: {weather_data['wind_speed']} m/s""",
            
            'hindi': f"""🌤 **{location_display} में मौसम:**
• स्थिति: {weather_data['description']}
• तापमान: {weather_data['temperature']}°C
• आर्द्रता: {weather_data['humidity']}%
• हवा की गति: {weather_data['wind_speed']} m/s""",
            
            'gujarati': f"""🌤 **{location_display} માં હવામાન:**
• સ્થિતિ: {weather_data['description']}
• તાપમાન: {weather_data['temperature']}°C
• ભેજ: {weather_data['humidity']}%
• પવનની ગતિ: {weather_data['wind_speed']} m/s"""
        }
        
        return templates.get(language_code, templates['english'])

class PDFAgent:
    """Specialized agent for PDF/document-based queries"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file):
        """Extract text from PDF with better handling"""
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if not text.strip():
                    text = page.get_text("dict")
                    page_text = ""
                    for block in text["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    page_text += span["text"] + " "
                    text = page_text
                
                if text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")
            
            doc.close()
            full_text = "\n\n".join(text_content)
            st.session_state.pdf_content = full_text
            return full_text
            
        except Exception as e:
            st.error(f"Error extracting PDF text: {str(e)}")
            return None
    
    @staticmethod
    def query_documents(question):
        """Query the document collection with improved search"""
        try:
            results = collection.query(query_texts=[question], n_results=5)
            context_parts = []
            
            if results["documents"] and results["documents"][0]:
                context_parts.extend(results["documents"][0])
            
            if st.session_state.pdf_content:
                question_words = question.lower().split()
                pdf_lines = st.session_state.pdf_content.split('\n')
                
                relevant_lines = []
                for line in pdf_lines:
                    if any(word in line.lower() for word in question_words):
                        relevant_lines.append(line.strip())
                
                if relevant_lines:
                    context_parts.extend(relevant_lines[:10])
            
            if context_parts:
                unique_context = list(dict.fromkeys(context_parts))
                return " ".join(unique_context)
            
            return None
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None
    
    @staticmethod
    def generate_response(question, context, conversation_history=None, language_code='english'):
        """Generate response based on document context with multilingual support"""
        if not context:
            error_messages = {
                'english': "❌ No relevant documents found. Please upload a PDF or text file first.",
                'hindi': "❌ कोई संबंधित दस्तावेज़ नहीं मिला। कृपया पहले एक PDF या टेक्स्ट फ़ाइल अपलोड करें।",
                'gujarati': "❌ કોઈ સંબંધિત દસ્તાવેજો મળ્યા નથી. કૃપા કરીને પહેલા PDF અથવા ટેક્સ્ટ ફાઇલ અપલોડ કરો."
            }
            return error_messages.get(language_code, error_messages['english'])
        
        system_prompt = translator.get_system_prompt(language_code)
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append(msg)
        
        messages.append({
            "role": "user", 
            "content": f"Context from document: {context[:3000]}\n\nQuestion: {question}"
        })
        
        try:
            response = llm.invoke(messages)
            prefixes = {
                'english': "📄 **Based on uploaded document:**\n",
                'hindi': "📄 **अपलोड किए गए दस्तावेज़ के आधार पर:**\n",
                'gujarati': "📄 **અપલોડ કરેલા દસ્તાવેજના આધારે:**\n"
            }
            prefix = prefixes.get(language_code, prefixes['english'])
            return f"{prefix}{response.content}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

class AgricultureAgent:
    """Specialized agent for general agriculture queries"""
    
    @staticmethod
    def generate_response(question, weather_context=None, conversation_history=None, language_code='english'):
        """Generate agriculture-focused response with multilingual support"""
        system_prompt = translator.get_system_prompt(language_code)
        
        if weather_context:
            location_info = AgricultureAgent.get_location_context(weather_context)
            if location_info:
                system_prompt += f" The user is asking about agriculture in relation to {location_info}. Provide location-specific advice."
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            for msg in conversation_history[-8:]:
                messages.append(msg)
        
        if weather_context:
            messages.append({"role": "system", "content": f"Current weather information: {weather_context}"})
        
        messages.append({"role": "user", "content": question})
        
        try:
            response = llm.invoke(messages)
            prefixes = {
                'english': "🌾 **Agriculture Assistant:**\n",
                'hindi': "🌾 **कृषि सहायक:**\n",
                'gujarati': "🌾 **કૃષિ સહાયક:**\n"
            }
            prefix = prefixes.get(language_code, prefixes['english'])
            return f"{prefix}{response.content}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    @staticmethod
    def get_location_context(weather_context):
        """Extract location information from weather context"""
        if not weather_context:
            return None
        
        lines = weather_context.split('\n')
        for line in lines:
            if 'Weather in' in line or 'मौसम' in line or 'હવામાન' in line:
                if 'Weather in' in line:
                    location_part = line.split('Weather in')[1].strip().rstrip(':')
                    return location_part
        return None

class GeneralChatbot:
    """Main chatbot that coordinates with specialized agents"""
    
    @staticmethod
    def update_conversation_context(user_query, bot_response):
        """Update conversation context for memory"""
        st.session_state.conversation_context.append({"role": "user", "content": user_query})
        st.session_state.conversation_context.append({"role": "assistant", "content": bot_response})
        
        if len(st.session_state.conversation_context) > 20:
            st.session_state.conversation_context = st.session_state.conversation_context[-20:]
    
    @staticmethod
    def process_query(query, language_code='english'):
        """Process query and route to appropriate agent with multilingual support"""
        agent_type = AgentRouter.classify_query(query)
        
        if agent_type == 'weather':
            location = WeatherAgent.extract_location(query)
            weather_data = WeatherAgent.get_weather_data(location)
            response = WeatherAgent.format_weather_response(weather_data, language_code)
            
        elif agent_type == 'pdf':
            context = PDFAgent.query_documents(query)
            response = PDFAgent.generate_response(query, context, st.session_state.conversation_context, language_code)
            
        elif agent_type == 'combined':
            location = WeatherAgent.extract_location(query)
            weather_data = WeatherAgent.get_weather_data(location)
            
            if 'error' in weather_data:
                agri_response = AgricultureAgent.generate_response(query, None, st.session_state.conversation_context, language_code)
                error_messages = {
                    'english': f"⚠️ Couldn't fetch weather for {location}, but here's agricultural advice:\n\n{agri_response}",
                    'hindi': f"⚠️ {location} के लिए मौसम नहीं मिल सका, लेकिन यहाँ कृषि सलाह है:\n\n{agri_response}",
                    'gujarati': f"⚠️ {location} માટે હવામાન મળી શક્યું નથી, પરંતુ અહીં કૃષિ સલાહ છે:\n\n{agri_response}"
                }
                response = error_messages.get(language_code, error_messages['english'])
            else:
                weather_info = WeatherAgent.format_weather_response(weather_data, language_code)
                agri_advice = AgricultureAgent.generate_response(query, weather_info, st.session_state.conversation_context, language_code)
                response = f"{weather_info}\n\n---\n\n{agri_advice}"
        
        else:
            weather_context = None
            if any(word in query.lower() for word in ['planting', 'sowing', 'irrigation', 'crop', 'season', 'suitable', 'grow']):
                potential_location = WeatherAgent.extract_location(query)
                weather_data = WeatherAgent.get_weather_data(potential_location)
                if 'error' not in weather_data:
                    weather_context = WeatherAgent.format_weather_response(weather_data, language_code)
            
            response = AgricultureAgent.generate_response(query, weather_context, st.session_state.conversation_context, language_code)
        
        GeneralChatbot.update_conversation_context(query, response)
        return response

# Helper functions for file processing
def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def store_in_vector_db(content, source):
    chunks = split_text(content)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source, "type": "file"} for _ in chunks]
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)

def generate_summary(text, language_code='english'):
    prompts = {
        'english': f"Summarize the following agricultural document in 100-150 words:\n\n{text[:2000]}",
        'hindi': f"निम्नलिखित कृषि दस्तावेज़ का 100-150 शब्दों में सारांश दें:\n\n{text[:2000]}",
        'gujarati': f"નીચેના કૃષિ દસ્તાવેજનો 100-150 શબ્દોમાં સારાંશ આપો:\n\n{text[:2000]}"
    }
    
    try:
        prompt = prompts.get(language_code, prompts['english'])
        return llm.invoke(prompt).content
    except:
        return "Summary generation failed."

def generate_insights(text, language_code='english'):
    prompts = {
        'english': f"Give 3-5 key agricultural insights from the following content:\n\n{text[:2000]}",
        'hindi': f"निम्नलिखित सामग्री से 3-5 मुख्य कृषि अंतर्दृष्टि दें:\n\n{text[:2000]}",
        'gujarati': f"નીચેની સામગ્રીમાંથી 3-5 મુખ્ય કૃષિ સૂઝ આપો:\n\n{text[:2000]}"
    }
    
    try:
        prompt = prompts.get(language_code, prompts['english'])
        return llm.invoke(prompt).content
    except:
        return "Insights generation failed."

def custom_css():
    """Apply custom CSS for professional UI"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Header Styles */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 20px 20px 20px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
    }
    
    /* Language Selector */
    .language-selector {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Language Selector Continued */
    .language-selector .stSelectbox > div > div {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        border: none;
        font-weight: 500;
    }
    
    /* Chat Container */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
    }
    
    /* Message Styles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 2px 10px rgba(240, 147, 251, 0.3);
    }
    
    /* File Upload Area */
    .upload-area {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        border: 2px dashed rgba(255,255,255,0.5);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(168, 237, 234, 0.4);
    }
    
    /* Voice Controls */
    .voice-controls {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(252, 182, 159, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input Field */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e1e5e9;
        padding: 0.75rem 1.5rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Custom animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def render_chat_message(message, is_user=False):
    """Render a chat message with proper styling"""
    if is_user:
        st.markdown(f'<div class="user-message fade-in">👤 **You:** {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message fade-in">🤖 **Assistant:** {message}</div>', unsafe_allow_html=True)

def handle_voice_input(language_config):
    """Handle voice input with proper error handling"""
    if st.session_state.is_recording:
        st.warning("🎤 Listening... Speak now!")
        
        # Listen for speech
        speech_code = language_config['speech_code']
        result = voice_assistant.listen_for_speech(speech_code, timeout=10)
        
        if result == "timeout":
            st.error("⏰ No speech detected. Please try again.")
        elif result == "unclear":
            st.error("🔇 Could not understand the audio. Please speak clearly.")
        elif result.startswith("error:"):
            st.error(f"❌ {result}")
        else:
            st.success(f"🎤 Heard: {result}")
            st.session_state.voice_input = result
            st.session_state.is_recording = False
            st.rerun()

def main():
    """Main Streamlit application"""
    # Apply custom CSS
    custom_css()
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🌾 Smart Agriculture Assistant</h1>
        <p class="header-subtitle">AI-Powered Multilingual Farming Advisor with Voice Support</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Language Selection
    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 🌍 Language / भाषा / ભાષા")
    
    with col2:
        selected_lang = st.selectbox(
            "Choose your preferred language:",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: f"{LANGUAGES[x]['flag']} {LANGUAGES[x]['name']}",
            index=list(LANGUAGES.keys()).index(st.session_state.selected_language),
            key="language_selector"
        )
        
        if selected_lang != st.session_state.selected_language:
            st.session_state.selected_language = selected_lang
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Current language configuration
    current_lang = LANGUAGES[st.session_state.selected_language]
    
    # Voice Controls
    st.markdown('<div class="voice-controls">', unsafe_allow_html=True)
    st.markdown("### 🎤 Voice Assistant")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        voice_enabled = st.toggle("Enable Voice", value=st.session_state.voice_enabled)
        st.session_state.voice_enabled = voice_enabled
    
    with col2:
        if voice_enabled and st.button("🎤 Start Recording"):
            st.session_state.is_recording = True
            st.rerun()
    
    with col3:
        if st.button("🔊 Test Voice"):
            if voice_enabled:
                test_messages = {
                    'english': "Hello! I'm your agriculture assistant.",
                    'hindi': "नमस्ते! मैं आपका कृषि सहायक हूँ।",
                    'gujarati': "નમસ્તે! હું તમારો કૃષિ સહાયક છું."
                }
                test_msg = test_messages.get(st.session_state.selected_language, test_messages['english'])
                if voice_assistant.speak_text(test_msg, st.session_state.selected_language):
                    st.success("🔊 Voice test successful!")
                else:
                    st.error("❌ Voice test failed!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle voice input
    if st.session_state.is_recording:
        handle_voice_input(current_lang)
    
    # File Upload Section
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Agricultural Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file (Max 10MB):",
        type=['pdf'],
        help="Upload agricultural documents, research papers, or guides"
    )
    
    if uploaded_file is not None:
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
            st.error("⚠️ File size exceeds 10MB limit!")
        else:
            with st.spinner("📖 Processing document..."):
                try:
                    # Extract text from PDF
                    extracted_text = PDFAgent.extract_text_from_pdf(uploaded_file)
                    
                    if extracted_text:
                        # Store in vector database
                        store_in_vector_db(extracted_text, uploaded_file.name)
                        
                        # Generate summary and insights
                        summary = generate_summary(extracted_text, st.session_state.selected_language)
                        insights = generate_insights(extracted_text, st.session_state.selected_language)
                        
                        st.success("✅ Document processed successfully!")
                        
                        # Display summary and insights
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📋 Summary:**")
                            st.info(summary)
                        
                        with col2:
                            st.markdown("**💡 Key Insights:**")
                            st.info(insights)
                    else:
                        st.error("❌ Failed to extract text from PDF!")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat Interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("### 💬 Chat with Assistant")
    
    # Chat History
    if st.session_state.chat_history:
        st.markdown("**Chat History:**")
        for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history[-5:]):  # Show last 5 exchanges
            render_chat_message(user_msg, is_user=True)
            render_chat_message(bot_msg, is_user=False)
            st.markdown("---")
    
    # Input area
    user_input = None
    
    # Check for voice input
    if hasattr(st.session_state, 'voice_input') and st.session_state.voice_input:
        user_input = st.session_state.voice_input
        st.session_state.voice_input = None
    
    # Text input
    if not user_input:
        placeholder_texts = {
            'english': "Ask about crops, weather, farming techniques, or upload documents...",
            'hindi': "फसलों, मौसम, खेती की तकनीकों के बारे में पूछें या दस्तावेज़ अपलोड करें...",
            'gujarati': "પાક, હવામાન, ખેતીની તકનીકો વિશે પૂછો અથવા દસ્તાવેજો અપલોડ કરો..."
        }
        
        user_input = st.text_input(
            "Your question:",
            placeholder=placeholder_texts.get(st.session_state.selected_language, placeholder_texts['english']),
            key="user_input"
        )
    
    # Process input
    if user_input:
        with st.spinner("🤔 Processing your question..."):
            try:
                # Generate response
                response = GeneralChatbot.process_query(user_input, st.session_state.selected_language)
                
                # Add to chat history
                st.session_state.chat_history.append((user_input, response))
                
                # Keep only last 10 exchanges
                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[-10:]
                
                # Display response
                render_chat_message(user_input, is_user=True)
                render_chat_message(response, is_user=False)
                
                # Voice output if enabled
                if st.session_state.voice_enabled:
                    # Clean response for TTS (remove markdown and emojis)
                    clean_response = re.sub(r'[*#🌾🌤📄⚠️❌✅💬📁📋💡👤🤖]', '', response)
                    clean_response = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_response)  # Remove bold markdown
                    
                    if voice_assistant.speak_text(clean_response[:200], st.session_state.selected_language):
                        st.success("🔊 Response spoken!")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                logger.error(f"Query processing error: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar with additional features
    with st.sidebar:
        st.markdown("### ⚙︎ Settings")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.conversation_context = []
            st.success("Chat history cleared!")
        
        # Weather locations quick access
        st.markdown("### ☀️ Quick Weather")
        quick_locations = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Ahmedabad", "Gandhinagar"]
        
        for location in quick_locations:
            if st.button(f"🌤️ {location}"):
                weather_data = WeatherAgent.get_weather_data(location)
                weather_response = WeatherAgent.format_weather_response(weather_data, st.session_state.selected_language)
                st.session_state.chat_history.append((f"Weather in {location}", weather_response))
                st.rerun()
        

            # System status
        st.markdown("### 📊 System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(st.session_state.get('pdf_content', '').split('\n')) if st.session_state.get('pdf_content') else 0)
        with col2:
            st.metric("Chat History", len(st.session_state.chat_history))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Main application error: {e}")