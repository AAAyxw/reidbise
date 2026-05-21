# -*- coding: utf-8 -*-
import zipfile
import xml.etree.ElementTree as ET
import sys

WNS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_text(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    paras = []
    for p in root.iter(WNS + "p"):
        texts = [t.text for t in p.iter(WNS + "t") if t.text]
        if texts:
            paras.append("".join(texts))
    return "\n".join(paras)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "开题报告"
    path = rf"d:\我的论文项目\{name}.docx"
    out = rf"d:\reidbise-yolov12\_docx_{name}.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(docx_text(path))
    print(out)
