
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.document_loader import load_pdfs


def create_chunks():

    # Load extracted PDF pages
    documents = load_pdfs()

    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    # Group pages by source document
    documents_by_source = {}

    for doc in documents:
        documents_by_source.setdefault(doc["source"], []).append(doc)

    # Split each complete document instead of splitting pages independently
    for source, pages in documents_by_source.items():

        pages.sort(key=lambda x: x["page"])

        page_texts = [page["text"] for page in pages]
        full_text = "\n\n".join(page_texts)

        chunks = splitter.split_text(full_text)

        # Track the approximate page range covered by each chunk
        page_positions = []
        current_position = 0

        for page in pages:
            start = current_position
            end = start + len(page["text"])

            page_positions.append({
                "page": page["page"],
                "start": start,
                "end": end
            })

            current_position = end + 2  # Account for the separator

        # Create chunk metadata
        search_position = 0

        for chunk_index, chunk in enumerate(chunks):

            chunk_start = full_text.find(chunk, search_position)

            if chunk_start == -1:
                chunk_start = full_text.find(chunk)

            chunk_end = chunk_start + len(chunk)
            search_position = chunk_start + 1

            overlapping_pages = [
                page["page"]
                for page in page_positions
                if (
                    chunk_start < page["end"]
                    and chunk_end > page["start"]
                )
            ]

            all_chunks.append({
                "chunk_id": f"{source}_chunk_{chunk_index + 1}",
                "text": chunk,
                "source": source,
                "page": overlapping_pages[0] if overlapping_pages else pages[0]["page"],
                "pages": overlapping_pages
            })

    return all_chunks


if __name__ == "__main__":

    all_chunks = create_chunks()

    print("Total chunks:", len(all_chunks))

    for chunk in all_chunks[:3]:
        print("\nChunk ID:", chunk["chunk_id"])
        print("Source:", chunk["source"])
        print("Pages:", chunk["pages"])
        print("Text:", chunk["text"])
