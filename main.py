import os
import yt_dlp
import whisper
import torch
import re
from datetime import timedelta
from transformers import pipeline
from google.colab import drive

# Mount Google Drive
drive.mount('/content/gdrive')

# Set save path
SAVE_PATH = "/content/gdrive/MyDrive/YouTubeSummarizer/"
os.makedirs(SAVE_PATH, exist_ok=True)

# Path to cookies.txt file (exported via Get cookies.txt LOCALLY)
COOKIES_FILE = "/content/gdrive/MyDrive/cookies.txt"

# Initialize summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define video profile presets
VIDEO_PRESETS = {
    "1": {"name": "10 minutes", "max_length": 100, "chunk_limit": 512},
    "2": {"name": "30 minutes", "max_length": 150, "chunk_limit": 768},
    "3": {"name": "1 hour", "max_length": 250, "chunk_limit": 896},
    "4": {"name": "2+ hours", "max_length": 350, "chunk_limit": 1024}
}

# Download audio from YouTube using cookies
def download_audio(youtube_url):
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'cookies': COOKIES_FILE}) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_title = info.get('title', 'audio')
    except Exception as e:
        print("Error fetching info:", e)
        return None, None

    video_title = re.sub(r'[^\w\s-]', '', video_title).strip().replace(" ", "_")
    output_path = os.path.join(SAVE_PATH, f"{video_title}.mp3")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path.replace('.mp3', ''),
        'cookies': COOKIES_FILE,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except Exception as e:
        print("Download failed:", e)
        return None, None

    if os.path.exists(output_path + ".mp3"):
        os.rename(output_path + ".mp3", output_path)

    return output_path, video_title

# Transcribe using Whisper
def transcribe_audio(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model = whisper.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu")
    result = model.transcribe(audio_path)
    return result["segments"]

# Summarize transcription
def summarize_transcription(segments, max_length, chunk_token_limit):
    summarized_sections = []
    chunk_text = ""
    chunk_start_time = None

    for segment in segments:
        start_time = str(timedelta(seconds=int(segment["start"])))
        text = segment["text"]

        if len(chunk_text) + len(text) > chunk_token_limit:
            input_len = len(chunk_text.split())
            adjusted_max_len = min(max_length, int(input_len * 0.7))
            summarized_text = summarizer(chunk_text, max_length=adjusted_max_len, min_length=50, do_sample=False)[0]["summary_text"]
            summarized_sections.append(f"[{chunk_start_time}] {summarized_text}")
            chunk_text = text
            chunk_start_time = start_time
        else:
            if not chunk_text:
                chunk_start_time = start_time
            chunk_text += " " + text

    if chunk_text:
        input_len = len(chunk_text.split())
        adjusted_max_len = min(max_length, int(input_len * 0.7))
        summarized_text = summarizer(chunk_text, max_length=adjusted_max_len, min_length=50, do_sample=False)[0]["summary_text"]
        summarized_sections.append(f"[{chunk_start_time}] {summarized_text}")

    return "\n".join(summarized_sections)

# Full processing pipeline
def summarize_youtube_video(youtube_url, max_length, chunk_token_limit):
    print("Downloading audio...")
    audio_path, video_title = download_audio(youtube_url)
    if not audio_path:
        return

    print(f"Audio downloaded: {audio_path}")

    print("Transcribing audio...")
    try:
        segments = transcribe_audio(audio_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    transcription_path = os.path.join(SAVE_PATH, f"{video_title}_transcription.txt")
    with open(transcription_path, "w") as f:
        for segment in segments:
            f.write(f"[{str(timedelta(seconds=int(segment['start'])))}] {segment['text']}\n")

    print("Summarizing...")
    summary = summarize_transcription(segments, max_length, chunk_token_limit)

    summary_path = os.path.join(SAVE_PATH, f"{video_title}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"Summary saved to: {summary_path}")
    print(f"Transcription saved to: {transcription_path}")
    print("\nPreview of summary:\n")
    print(summary[:1000])

# User input loop
num_videos = int(input("How many YouTube videos do you want to summarize? "))
video_urls = []
video_profiles = []

for i in range(num_videos):
    url = input(f"Enter YouTube URL {i+1}: ")
    print("Choose video length:")
    for key, preset in VIDEO_PRESETS.items():
        print(f"{key}: {preset['name']}")
    choice = input(f"Select a category for video {i+1}: ")
    while choice not in VIDEO_PRESETS:
        choice = input("Invalid. Select 1, 2, 3, or 4: ")

    video_urls.append(url)
    video_profiles.append(VIDEO_PRESETS[choice])

for i, url in enumerate(video_urls):
    profile = video_profiles[i]
    summarize_youtube_video(url, profile["max_length"], profile["chunk_limit"])

print("All videos processed.")
