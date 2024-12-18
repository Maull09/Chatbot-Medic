import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

st.title("Chatbot Medis")

# Cache untuk memuat model dan tokenizer
@st.cache_resource  # Gunakan cache_resource untuk objek besar seperti model
def load_model():
    model_path = "maull04/biogpt_finetuning"  # Sesuaikan dengan path folder model Anda
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Menambahkan pad_token jika tidak ada
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return text_gen_pipeline

# Memuat pipeline menggunakan cache
text_gen_pipeline = load_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Masukkan pertanyaan medis Anda di sini..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response from the model
    with st.chat_message("assistant"):
        with st.spinner("Mengetik..."):
            # Format prompt for model
            formatted_prompt = (
                "Instruction: If you are a doctor, please answer the medical questions based on the patient's description.\n"
                f"Input: {user_input}\nResponse:"
            )
            
            # Generate response
            response = text_gen_pipeline(formatted_prompt, max_length=256, do_sample=True, num_return_sequences=1, truncation=False)
            generated_text = response[0]['generated_text'].split("Response:")[-1].strip()
            
            # Display assistant's response in chat message container
            st.markdown(generated_text)
            
            # Add assistant's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": generated_text})
