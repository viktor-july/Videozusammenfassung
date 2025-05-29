# Videozusammenfassung (YouTube Video Summarizer)

This project downloads YouTube video audio, transcribes it using Whisper, and summarizes the transcription using a BART summarizer.

---

## How it works

1. Mounts Google Drive to save files.
2. Downloads audio from YouTube using `yt-dlp` with cookies support.
3. Transcribes audio with Whisper medium model.
4. Summarizes the transcription in chunks.
5. Saves audio, full transcription, and summary (with timestamps) in your Google Drive.

---

## Usage

- Install all necessary packages and mount a drive
- Run `main.py` in Google Colab.
- Enter the number of videos you want to summarize.
- Paste the URLs.
- Select the video length category (10m, 30m, 1h, 2h+).
- Wait for the process to finish.
- Check the saved files in your Google Drive folder `/YouTubeSummarizer/`.

---

## Notes

- You need a cookies.txt file exported from your browser to download some videos (for videos requiring sign-in).
- Large videos can take time and storage.
- Summary length and chunk size are set by video length categories.
- The code uses open source models: Whisper by OpenAI and BART by Facebook.
- Better to use GPU in runtime, though if you have more time, it's better to use CPU, especially without a subscription.

---

## What’s next?

- Add more video length presets.
- Improve chunking and summarization.


