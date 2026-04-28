from pathlib import Path

lab_dir = Path(r"D:\reidbise-yolov12\datasets\person_det\labels\val")
txts = sorted(lab_dir.glob("*.txt"))

print("txt files:", len(txts))

valid_files = 0
valid_lines = 0
bad_examples = []

for f in txts:
    ok_file = False
    lines = f.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    for ln in lines:
        parts = ln.strip().replace(",", " ").split()
        if len(parts) != 5:
            bad_examples.append((f.name, ln))
            continue
        try:
            c = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                bad_examples.append((f.name, ln))
                continue
            ok_file = True
            valid_lines += 1
        except:
            bad_examples.append((f.name, ln))
            continue
    if ok_file:
        valid_files += 1

print("valid_files:", valid_files)
print("valid_lines:", valid_lines)
print("bad_examples_count:", len(bad_examples))
print("bad_examples_top10:")
for x in bad_examples[:10]:
    print(x)