import fitz
import os

DATA_DIR = "data"

for pdf in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, pdf)

    doc = fitz.open(path)

    text = ""

    for page in doc:
        text += page.get_text()

    print(pdf)
    print(len(text))