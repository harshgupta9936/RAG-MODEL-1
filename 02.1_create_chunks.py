import whisper
import json
import os

audios=os.listdir("audios")

model = whisper.load_model("base").to("cuda")

for audio in audios:
    number = audio.split("_")[0]
    title = audio.split("_")[1].split(".mp3")[0]
    # print(number, title)
    result = model.transcribe(audio= f"audios/{audio}",
                          language="en",
                          task = "transcribe",
                          word_timestamps = False,
                          fp16=True)

    chunks=[]
    for segment in result["segments"]:
        chunks.append({"number": number, "title" : title, "start" : segment["start"], "end" : segment["end"], "text" : segment["text"]})
    
    chunks_with_metadata = {"chunks": chunks, "text": result["text"]}    

    with open(f"jsons/{audio}.json", "w") as f:
        json.dump(chunks_with_metadata, f, indent =4)