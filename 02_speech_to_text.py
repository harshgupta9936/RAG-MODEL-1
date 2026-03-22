import whisper
import json

model = whisper.load_model("base").to("cuda")

result = model.transcribe(audio= "audios/01_ChatGPT for Data Analytics.mp3",
                          language="en",
                          task = "transcribe",
                          word_timestamps = False,
                          fp16=True)
# print(result["text"])
# with open("output.json", "w") as f:
#     json.dump(result, f, indent=4)


# print(result["segments"])
chunks=[]
for segment in result["segments"]:
    chunks.append({"start" : segment["start"], "end" : segment["end"], "text" : segment["text"]})

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks, f, indent =4)