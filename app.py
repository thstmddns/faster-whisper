import os
import sys
import gradio as gr
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import re
import nltk
from nltk.corpus import stopwords
from rapidfuzz.distance import Levenshtein
# import openai
import random

# Ensure stopwords are available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

load_dotenv()

openai_key = os.getenv("OPENAI_KEY")
# 만약 openapi key가 없다면 아래에 추가 

if openai_key is None or openai_key == "":
    sys.exit("Please Provide Your OpenAI API Key")

# openai.api_key = openai_key

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def transcript(audio_file_path):
    segments, info = model.transcribe(audio_file_path, beam_size=5)
    transcription = " ".join([segment.text for segment in segments])
    return transcription

def calculate_similarity(transcription, reference_script):
    # Preprocess both texts
    transcription_processed = preprocess_text(transcription)
    reference_script_processed = preprocess_text(reference_script)
    
    # Calculate Levenshtein similarity
    distance = Levenshtein.distance(transcription_processed, reference_script_processed)
    max_len = max(len(transcription_processed), len(reference_script_processed))
    similarity_score = 1 - (distance / max_len)
    
    return similarity_score

def generate_script():
    prompt = "Create a simple sentence in Korean for language learning purposes."
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # return response['choices'][0]['message']['content'].strip()
    # Since OpenAI API is not being used, let's return a placeholder script
    return "이것은 한국어 학습을 위한 간단한 문장입니다."

def get_random_script():
    return generate_script()

def transcript_with_similarity(audio_file, reference_script):
    transcription = transcript(audio_file)
    similarity_score = calculate_similarity(transcription, reference_script)
    result = f"Transcription: {transcription}\nReference Script: {reference_script}\n\nSimilarity with reference script: {similarity_score:.2f}"
    return result

with gr.Blocks() as prototype:
    gr.Markdown('손선생과 함께하는 한국어 공부시간')

    with gr.Row():
        script_text = gr.Textbox(label="Script", interactive=False)
        new_script_button = gr.Button("Generate New Script")
    
    with gr.Row():
        audio = gr.Audio(sources=["microphone"], type="filepath", streaming=True)
        file = gr.UploadButton(file_types=[".mp3", ".wav"], label="Select File", type="filepath")
    
    evaluate_button = gr.Button("Evaluate Similarity")
    output_text = gr.Textbox(label="Output Text")

    def update_script():
        script = get_random_script()
        return script

    new_script_button.click(fn=update_script, outputs=script_text)
    evaluate_button.click(fn=transcript_with_similarity, inputs=[audio, script_text], outputs=output_text)
    file.upload(fn=transcript_with_similarity, inputs=[file, script_text], outputs=output_text)

    script_text.value = get_random_script()

if __name__ == "__main__":
    prototype.launch()
