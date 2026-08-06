import os
import json
import re
from pathlib import Path
from datetime import datetime
import config

# Try optional imports
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx  # python-docx
except ImportError:
    docx = None

try:
    from pypdf import PdfReader as _PyPdfReader  # fallback for damaged PDFs
except ImportError:
    _PyPdfReader = None

class DocumentIngester:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir or config.EVAL_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.data_dir / "document_index.json"
        self.documents_metadata = self.load_index()

    def load_index(self):
        """Load the JSON database index of ingested files."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_index(self):
        """Save the JSON database index."""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.documents_metadata, f, indent=4, ensure_ascii=False)

    def extract_text_from_pdf(self, file_path):
        """Extract text and page-level metadata from PDF using PyMuPDF.
        Falls back to multiple extraction modes for compressed/scanned PDFs.
        Falls back further to pypdf for partially corrupted PDFs.
        """
        if not fitz:
            raise ImportError("PyMuPDF (fitz) is not installed. Please run 'pip install pymupdf'.")
        
        doc = fitz.open(file_path)
        pages_content = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Primary extraction
            text = page.get_text()
            
            # Fallback 1: blocks method (better for columnar PDFs)
            if not text.strip():
                try:
                    blocks = page.get_text("blocks")
                    text = "\n".join(b[4] for b in blocks if isinstance(b[4], str))
                except Exception:
                    text = ""
            
            # Fallback 2: words method
            if not text.strip():
                try:
                    words = page.get_text("words")
                    text = " ".join(w[4] for w in words if isinstance(w[4], str))
                except Exception:
                    text = ""
            
            # Fallback 3: rawdict — extract span text
            if not text.strip():
                try:
                    raw = page.get_text("rawdict")
                    spans = []
                    for block in raw.get("blocks", []):
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                if span.get("text", "").strip():
                                    spans.append(span["text"])
                    text = " ".join(spans)
                except Exception:
                    text = ""
            
            if text.strip():
                pages_content.append({
                    "page_number": page_num + 1,
                    "text": text
                })
        
        # Fallback to pypdf if fitz got nothing (handles partially corrupted PDFs)
        if not pages_content and _PyPdfReader:
            try:
                print(f"  [Fallback] Trying pypdf for {Path(file_path).name} ...")
                reader = _PyPdfReader(str(file_path), strict=False)
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text() or ""
                        if text.strip():
                            pages_content.append({
                                "page_number": page_num + 1,
                                "text": text
                            })
                    except Exception:
                        pass
                if pages_content:
                    print(f"  [Fallback] pypdf recovered {len(pages_content)} pages.")
            except Exception as e:
                print(f"  [Fallback] pypdf also failed: {e}")
        
        return pages_content

    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX using python-docx."""
        if not docx:
            raise ImportError("python-docx is not installed. Please run 'pip install python-docx'.")
        
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return [{"page_number": 1, "text": text}] # DOCX doesn't have native easy page breaks

    def extract_text_from_txt(self, file_path):
        """Extract text from a plain TXT file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [{"page_number": 1, "text": text}]

    def recursive_character_chunker(self, text, chunk_size=800, chunk_overlap=150):
        """
        Split text recursively based on a list of separators.
        Attempts to split on paragraphs first, then lines, then sentences, then spaces.
        """
        separators = ["\n\n", "\n", ". ", " ", ""]
        
        def split_recursive(text_to_split, separators_list):
            if len(text_to_split) <= chunk_size:
                return [text_to_split]
            
            if not separators_list:
                # Force chunk by character count if we ran out of separators
                return [text_to_split[i:i+chunk_size] for i in range(0, len(text_to_split), chunk_size - chunk_overlap)]
            
            separator = separators_list[0]
            remaining_separators = separators_list[1:]
            
            # Split text on this separator
            if separator == "":
                splits = list(text_to_split)
            else:
                splits = text_to_split.split(separator)
            
            chunks = []
            current_chunk = ""
            
            for split in splits:
                # Add separator back if it's not the last element
                item = split + (separator if separator != "" else "")
                
                # Check if this single split item itself is larger than chunk_size
                if len(item) > chunk_size:
                    # Flush the current chunk if exists
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    # Recursively split the long item with remaining separators
                    sub_chunks = split_recursive(item, remaining_separators)
                    chunks.extend(sub_chunks)
                # Check if we can add to the current chunk
                elif len(current_chunk) + len(item) <= chunk_size:
                    current_chunk += item
                else:
                    # Flush the current chunk
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = item
            
            if current_chunk:
                chunks.append(current_chunk.strip())
                
            return chunks

        # Perform initial splits and then build overlapping chunks
        raw_chunks = split_recursive(text, separators)
        
        # Merge chunks with overlap
        final_chunks = []
        for i, chunk in enumerate(raw_chunks):
            if not chunk:
                continue
            # If it's the first chunk or fits, just append
            if not final_chunks:
                final_chunks.append(chunk)
            else:
                prev_chunk = final_chunks[-1]
                # Calculate how much overlap we can grab from previous chunk
                overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) >= chunk_overlap else prev_chunk
                # Form the new chunk
                if chunk not in prev_chunk: # avoid exact duplicate chunks
                    final_chunks.append(chunk)
                    
        return final_chunks

    def ingest_document(self, file_path, domain="General", chunk_size=800, chunk_overlap=150):
        """
        Ingest a document, split into chunks, extract metadata, and update index.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = path.suffix.lower()
        doc_id = re.sub(r'[^a-zA-Z0-9]', '_', path.stem) + "_" + str(int(datetime.now().timestamp()))
        
        # Extract content page by page
        if file_ext == ".pdf":
            pages = self.extract_text_from_pdf(path)
        elif file_ext == ".docx":
            pages = self.extract_text_from_docx(path)
        elif file_ext in [".txt", ".md"]:
            pages = self.extract_text_from_txt(path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        # Process into chunks
        all_chunks = []
        chunk_idx = 0
        total_chars = 0
        
        for page_data in pages:
            page_num = page_data["page_number"]
            page_text = page_data["text"]
            total_chars += len(page_text)
            
            page_chunks = self.recursive_character_chunker(page_text, chunk_size, chunk_overlap)
            
            for chunk_text in page_chunks:
                if not chunk_text.strip():
                    continue
                all_chunks.append({
                    "chunk_id": f"{doc_id}_c{chunk_idx}",
                    "text": chunk_text,
                    "metadata": {
                        "source": path.name,
                        "page": page_num,
                        "domain": domain,
                        "char_count": len(chunk_text),
                        "chunk_index": chunk_idx
                    }
                })
                chunk_idx += 1

        # Prepare document metadata entry
        doc_entry = {
            "doc_id": doc_id,
            "filename": path.name,
            "domain": domain,
            "filepath": str(path.resolve()),
            "ingested_at": datetime.now().isoformat(),
            "total_characters": total_chars,
            "total_chunks": len(all_chunks),
            "chunks": all_chunks
        }
        
        # Save chunks physically in a separate JSON file to avoid bloated index file
        chunks_file_path = self.data_dir / f"{doc_id}_chunks.json"
        with open(chunks_file_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=4, ensure_ascii=False)
            
        # Save reference metadata in the main index
        doc_entry_light = doc_entry.copy()
        doc_entry_light["chunks_file"] = str(chunks_file_path.name)
        del doc_entry_light["chunks"] # Remove full chunks from index to keep it lightweight
        
        self.documents_metadata[doc_id] = doc_entry_light
        self.save_index()
        
        print(f"Successfully ingested {path.name} ({len(all_chunks)} chunks, Domain: {domain})")
        return doc_id, doc_entry_light

    def get_all_chunks(self):
        """Retrieve all ingested chunks across all documents."""
        all_chunks = []
        for doc_id, meta in self.documents_metadata.items():
            chunks_file = self.data_dir / meta.get("chunks_file")
            if chunks_file.exists():
                with open(chunks_file, "r", encoding="utf-8") as f:
                    all_chunks.extend(json.load(f))
        return all_chunks

    def get_domains(self):
        """Get unique list of domains from ingested documents."""
        return list(set(meta["domain"] for meta in self.documents_metadata.values()))
