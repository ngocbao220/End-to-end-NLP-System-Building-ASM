from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pprint import pprint

def metadata_func(record: dict, metadata: dict) -> dict:
    json_metadata = record.get("metadata", {})
    source_url = json_metadata.get("source")

    metadata["source"] = source_url
    return metadata

# Khởi tạo JSONLoader
# JSONLoader sẽ mặc định lấy đường dẫn file làm metadata, chúng ta ghi đè bằng hàm tự viết
loader = JSONLoader(
    file_path='data/document.json',
    jq_schema='.[]',
    content_key='content',
    metadata_func=metadata_func
)

# Load dữ liệu thành các Documents
documents = loader.load()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]

# Chunk văn bản
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

chunks = []
for doc in documents:
    # Chia nội dung Markdown
    header_splits = markdown_splitter.split_text(doc.page_content)
    # Header_splits trả về các Document mới, ta cần gán lại 'source' từ metadata gốc vào
    for split in header_splits:
        split.metadata.update(doc.metadata) # Giữ lại URL source từ JSON
        chunks.append(split)


# Kiểm tra kết quả
print(f"Số chunk được tạo ra: {len(chunks)}")
if chunks:
    print(chunks[:3])
    print("="*80)
    print("Ví dụ:")
    print(f"Nội dung: {chunks[1].page_content}")
    print(f"Metadata: {chunks[1].metadata}")
