import csv
import json

data = []
with open('song_describer-nosinging.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            "path": row["path"],
            "caption_id": int(row["caption_id"]),
            "caption": row["caption"]
        })

with open('captions.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
