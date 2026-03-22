import os
import json
import math

n=5

for filename in os.listdir("jsons"):
    if filename.endswith(".json"):
        file_path = os.path.join("jsons", filename)
        with open (file_path, "r", encoding ="utf-8") as f:
            data= json.load(f)
            new_chunks=[]
            num_chunks=len(data['chunks'])
            num_groups = math.ceil(num_chunks/n)

            for i in range (num_groups):
                start_indx = i*n
                end_indx = min((i+1)*n, num_chunks)

                chunk_grp = data['chunks'][start_indx : end_indx]

                new_chunks.append(
                    {
                        "number" : data['chunks'][0]['number'],
                        "title" : data['chunks'][0]['title'],
                        "start" : chunk_grp[0]['start'],
                        "end" : chunk_grp[-1]['end'],
                        "text" : " ".join(c['text'] for c in chunk_grp)
                    }
                )
            
            # save file without double .json
            os.makedirs("new_jsons", exist_ok=True)
            with open(os.path.join("new_jsons", filename), "w", encoding="utf-8") as json_file:
                json.dump({"chunks": new_chunks, "text" : data["text"]}, json_file, indent=4)