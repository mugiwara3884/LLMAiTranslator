import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from gtts import gTTS
import io 


# --- Set page config ---
st.set_page_config(page_title="Polyglot LLM Translator", page_icon="🌎", layout="centered")

# --- Load model and tokenizer with caching for performance ---
@st.cache_resource()
def load_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50Tokenizer.from_pretrained(model_name, use_fast=False)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# --- Language code mapping ---
lang_code_map = {
    'English': 'en_XX',
    'Hindi': 'hi_IN',
    'French': 'fr_XX',
    'Spanish': 'es_XX',
    'Russian': 'ru_RU',
    'Chinese (Simplified)': 'zh_CN',
    'German': 'de_DE',
    'Italian': 'it_IT',
    'Arabic': 'ar_AR',
    'Portuguese': 'pt_XX',
    'Japanese': 'ja_XX',
    'Korean': 'ko_KR'
}

# gTTS language codes
gtts_lang_map = {
    'English': 'en',
    'Hindi': 'hi',
    'French': 'fr',
    'Spanish': 'es',
    'Russian': 'ru',
    'Chinese (Simplified)': 'zh-CN',
    'German': 'de',
    'Italian': 'it',
    'Arabic': 'ar',
    'Portuguese': 'pt',
    'Japanese': 'ja',
    'Korean': 'ko'
}

lang_list = list(lang_code_map.keys())

# --- UI ---
st.title("🌎 Polyglot LLM Translator")
st.caption("Professional multilingual translation using Facebook's MBART-50 model.")

# --- Initialize session state for translation history ---
if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []

st.subheader("🖋️ Enter text to translate:")
input_text = st.text_area("", height=150)

col1, col2 = st.columns(2)
with col1:
    source_language = st.selectbox("🛠️ Source Language", lang_list)
with col2:
    target_language = st.selectbox("🛠️ Target Language", lang_list, index=1)

if st.button("🌐 Translate", type="primary"):
    if input_text.strip() == "":
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner("Translating..."):
            tokenizer.src_lang = lang_code_map[source_language]
            encoded_input = tokenizer(input_text, return_tensors="pt")
            generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.lang_code_to_id[lang_code_map[target_language]])
            translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            st.success("✅ Translation:")
            st.write(translated_text)

            # Save translation history to session state
            st.session_state.translation_history.append((input_text, translated_text))

            # Text-to-Speech and Download
            tts_lang_code = gtts_lang_map.get(target_language, 'en')  # fallback to English if not found
            tts = gTTS(text=translated_text, lang=tts_lang_code)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

            st.download_button("📄 Download Translation", data=translated_text, file_name="translation.txt")
            st.audio(audio_bytes, format='audio/mp3')

# --- Show Translation History ---
with st.expander("📜 Translation History"):
    if st.session_state.translation_history:
        for src, tgt in st.session_state.translation_history[-5:][::-1]:
            st.write(f"**Input:** {src}")
            st.write(f"**Output:** {tgt}")
            st.markdown("---")
    else:
        st.info("No translations yet.")
