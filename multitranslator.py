import streamlit as st
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import time
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="🌍 Multi-Language Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .translation-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .stat-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        flex: 1;
        margin: 0 0.5rem;
    }
    .success-msg {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .error-msg {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced caching for model loading
@st.cache_resource(show_spinner="🔄 Loading M2M100 model... This may take a moment.")
def load_translation_model():
    """Load and cache the M2M100 model and tokenizer"""
    model_name = "facebook/m2m100_418M"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

# Language mapping
@st.cache_data
def get_language_mapping():
    """Get comprehensive language mapping for M2M100"""
    return {
        "English": "en", "Hindi": "hi", "French": "fr", "German": "de", 
        "Spanish": "es", "Chinese (Simplified)": "zh", "Japanese": "ja", 
        "Korean": "ko", "Arabic": "ar", "Russian": "ru", "Portuguese": "pt", 
        "Italian": "it", "Dutch": "nl", "Swedish": "sv", "Norwegian": "no", 
        "Danish": "da", "Finnish": "fi", "Polish": "pl", "Czech": "cs", 
        "Hungarian": "hu", "Turkish": "tr", "Greek": "el", "Hebrew": "he", 
        "Thai": "th", "Vietnamese": "vi", "Indonesian": "id", "Malay": "ms", 
        "Tamil": "ta", "Telugu": "te", "Bengali": "bn", "Gujarati": "gu", 
        "Marathi": "mr", "Punjabi": "pa", "Urdu": "ur", "Swahili": "sw",
        "Amharic": "am", "Azerbaijani": "az", "Belarusian": "be", 
        "Bulgarian": "bg", "Catalan": "ca", "Croatian": "hr", "Estonian": "et",
        "Persian": "fa", "Georgian": "ka", "Kazakh": "kk", "Latvian": "lv",
        "Lithuanian": "lt", "Macedonian": "mk", "Mongolian": "mn", 
        "Romanian": "ro", "Serbian": "sr", "Slovak": "sk", "Slovenian": "sl",
        "Ukrainian": "uk", "Albanian": "sq", "Basque": "eu", "Galician": "gl"
    }

# Translation function with caching
@st.cache_data(show_spinner="🔄 Translating...", max_entries=100)
def translate_text(_tokenizer, _model, _device, text, src_lang, tgt_lang):
    """Translate text with caching for performance"""
    if not text.strip():
        return "⚠️ Please enter text to translate"
    
    if src_lang == tgt_lang:
        return f"ℹ️ Source and target languages are the same: {text}"
    
    try:
        # Set source language
        _tokenizer.src_lang = src_lang
        
        # Tokenize input
        encoded = _tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(_device)
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = _model.generate(
                **encoded,
                forced_bos_token_id=_tokenizer.get_lang_id(tgt_lang),
                max_length=512,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode translation
        translated = _tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )[0]
        
        return translated
        
    except Exception as e:
        return f"❌ Translation error: {str(e)}"

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'translation_history' not in st.session_state:
        st.session_state.translation_history = []
    
    if 'translation_count' not in st.session_state:
        st.session_state.translation_count = 0
    
    if 'favorite_languages' not in st.session_state:
        st.session_state.favorite_languages = {'source': 'English', 'target': 'Hindi'}

# Main application
def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Advanced Multi-Language Translator</h1>
        <p>Powered by Facebook's M2M100 - Supporting 50+ Languages with Direct Translation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model (cached)
    tokenizer, model, device = load_translation_model()
    
    if tokenizer is None or model is None:
        st.error("❌ Failed to load translation model. Please refresh the page.")
        return
    
    # Get language mapping
    language_map = get_language_mapping()
    language_options = list(language_map.keys())
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Translation Settings")
        
        # Language selection
        st.subheader("🔤 Language Selection")
        source_lang = st.selectbox(
            "Source Language",
            language_options,
            index=language_options.index(st.session_state.favorite_languages['source']),
            key="source_lang_select"
        )
        
        target_lang = st.selectbox(
            "Target Language", 
            language_options,
            index=language_options.index(st.session_state.favorite_languages['target']),
            key="target_lang_select"
        )
        
        # Quick swap button
        if st.button("🔄 Swap Languages", use_container_width=True):
            st.session_state.favorite_languages['source'], st.session_state.favorite_languages['target'] = target_lang, source_lang
            st.rerun()
        
        # Save as favorites
        if st.button("⭐ Save as Favorites", use_container_width=True):
            st.session_state.favorite_languages = {'source': source_lang, 'target': target_lang}
            st.success("✅ Languages saved as favorites!")
        
        # Statistics
        st.subheader("📊 Usage Statistics")
        st.metric("Total Translations", st.session_state.translation_count)
        st.metric("Supported Languages", len(language_options))
        st.metric("Model", "M2M100 (418M)")
        
        # Device info
        device_info = "🚀 GPU" if torch.cuda.is_available() else "💻 CPU"
        st.metric("Processing Device", device_info)
    
    # Main translation interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"📝 Input ({source_lang})")
        input_text = st.text_area(
            "Enter text to translate:",
            height=200,
            placeholder="Type your text here...",
            key="input_text"
        )
        
        # Character count
        char_count = len(input_text) if input_text else 0
        st.caption(f"Characters: {char_count}/512")
    
    with col2:
        st.subheader(f"🎯 Output ({target_lang})")
        
        # Translation button
        if st.button("🔄 Translate", type="primary", use_container_width=True):
            if input_text.strip():
                with st.spinner("Translating..."):
                    start_time = time.time()
                    
                    # Perform translation
                    result = translate_text(
                        tokenizer, model, device, 
                        input_text, 
                        language_map[source_lang], 
                        language_map[target_lang]
                    )
                    
                    translation_time = time.time() - start_time
                    
                    # Display result
                    st.markdown(f"""
                    <div class="translation-box">
                        <h4 style="color: black;">Translation Result:</h4>
                        <p style="font-size: 1.1em; line-height: 1.5; color: black;">{result}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Success metrics
                    if not result.startswith(("❌", "⚠️", "ℹ️")):
                        st.session_state.translation_count += 1
                        
                        # Add to history
                        history_entry = {
                            'input': input_text,
                            'output': result,
                            'source': source_lang,
                            'target': target_lang,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'time_taken': f"{translation_time:.2f}s"
                        }
                        st.session_state.translation_history.insert(0, history_entry)
                        
                        # Keep only last 10 entries
                        if len(st.session_state.translation_history) > 10:
                            st.session_state.translation_history = st.session_state.translation_history[:10]
                        
                        # Performance metrics
                        col_metric1, col_metric2 = st.columns(2)
                        with col_metric1:
                            st.metric("Translation Time", f"{translation_time:.2f}s")
                        with col_metric2:
                            st.metric("Words Translated", len(input_text.split()))
            else:
                st.warning("⚠️ Please enter text to translate")
    
    # Translation History
    if st.session_state.translation_history:
        st.subheader("📜 Recent Translations")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.translation_history = []
            st.rerun()
        
        # Display history
        for i, entry in enumerate(st.session_state.translation_history[:5]):
            with st.expander(f"🔄 {entry['source']} → {entry['target']} ({entry['timestamp']})"):
                col_hist1, col_hist2 = st.columns(2)
                with col_hist1:
                    st.write(f"**Input ({entry['source']}):**")
                    st.info(entry['input'])
                with col_hist2:
                    st.write(f"**Output ({entry['target']}):**")
                    st.success(entry['output'])
                st.caption(f"⏱️ Translation time: {entry['time_taken']}")

# Footer
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <p>🚀 Built with Streamlit | Powered by Facebook M2M100 | 
        <a href="#" style="color: #667eea;">Portfolio Project</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()
