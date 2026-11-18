
import fitz
from pathlib import Path

DATA_PATH = Path("data/raw_documents")

def load_pdfs():

    documents = []

    for pdf_file in DATA_PATH.glob("*.pdf"):

        pdf = fitz.open(pdf_file)

        for page_number, page in enumerate(pdf):

            text = page.get_text("text")

            if text.strip():

                documents.append({
                    "text": text,
                    "source": pdf_file.name,
                    "page": page_number + 1
                })

        pdf.close()

    return documents


if __name__ == "__main__":

    documents = load_pdfs()

    print("Total pages extracted:", len(documents))

    for doc in documents[:3]:

        print("\nSource:", doc["source"])
        print("Page:", doc["page"])
        print("Text:", doc["text"][:300])
