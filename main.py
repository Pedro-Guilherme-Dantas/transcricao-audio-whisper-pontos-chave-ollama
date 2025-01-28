from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import whisper

def transcription(audio_path):
    model = whisper.load_model("large")
    text = model.transcribe(audio_path)['text']
    return text

def resume_text(text):
    resume_model = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(resume_model)
    bart_model = BartForConditionalGeneration.from_pretrained(resume_model)
    inputs = tokenizer([text], return_tensors="pt", max_length=1024, truncation=True)
    ids = bart_model.generate(inputs["input_ids"], max_length=200, num_beams=4, early_stopping=True)
    resume = tokenizer.decode(ids[0], skip_special_tokens=True)
    return resume

def main():
    transcription_text = transcription("audio.mp3")
    print(f"Transcrição: {transcription_text}")

    resume = resume_text(transcription_text)
    print(f"Resumo: {resume}")

if __name__ == "__main__":
    main()
