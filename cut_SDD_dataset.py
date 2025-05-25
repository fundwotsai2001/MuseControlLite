import json

# Load the full list of 586 dictionaries
with open('/home/b06611012/fundwotsai/MuseControlLite_v2/SDD_nosinging_full_conditions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Compute half length
half = len(data) // 2

# Build new list of 293 dictionaries
new_list = []
for i in range(half):
    first = data[i]
    second = data[i + half]
    combined = {
        "path": first['path'],
        "caption_id": first['caption_id'],
        'caption': first['caption'],
        'dynamics_path': second['dynamics_path'],
        'melody_path': second['melody_path'],
        'rhythm_path': second['rhythm_path']
    }
    new_list.append(combined)

# Save to a new JSON file
output_path = '/home/b06611012/fundwotsai/MuseControlLite_v2/SDD_nosinging_half_conditions.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(new_list, f, ensure_ascii=False, indent=2)

# Display the first few entries
for entry in new_list[:5]:
    print(entry)

print(f"\nGenerated {len(new_list)} dictionaries. Saved to {output_path}")
