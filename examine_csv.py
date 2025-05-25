import csv

file_path = '/home/b06611012/fundwotsai/MuseControlLite_v2/song_describer-nosinging_half.csv'
with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)                   # skip header row
    data_line_count = sum(1 for row in reader if row)
print(f'Data rows (excluding header): {data_line_count}')
