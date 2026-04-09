"""
MemoryOS — File Memory Manager
================================
Handles upload, processing, and memory-storage of user files:
  • Images   — Groq vision (llama-3.2-11b-vision-preview) → rich description → archival memory
  • PDFs     — pdfplumber text extraction → chunked archival memories
  • Audio    — Groq Whisper transcription → archival memory
  • Text/CSV/MD — direct text → archival memory

Files are persisted to uploads/{user_id}/ on disk.
An index.json per user tracks all file metadata.
"""

import base64
import json
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import config

UPLOAD_BASE = config.BASE_DIR / "uploads"

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
ALLOWED_DOC   = {".pdf", ".txt", ".md", ".csv", ".docx"}
ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".webm"}
ALLOWED_ALL   = ALLOWED_IMAGE | ALLOWED_DOC | ALLOWED_AUDIO

MAX_BYTES = 30 * 1024 * 1024  # 30 MB

MIME_MAP = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".mp3":  "audio/mpeg",
    ".wav":  "audio/wav",
    ".m4a":  "audio/m4a",
    ".ogg":  "audio/ogg",
    ".webm": "audio/webm",
}


class FileMemoryManager:

    def __init__(self, user_id: str, archival_manager):
        self.user_id = user_id
        self.archive = archival_manager
        self._dir    = UPLOAD_BASE / user_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._dir / "index.json"
        self._meta: dict = self._load_meta()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            try:
                return json.loads(self._meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_meta(self):
        self._meta_path.write_text(
            json.dumps(self._meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _safe_name(filename: str) -> str:
        name = Path(filename).name
        name = re.sub(r"[^\w\-. ]", "_", name)[:120]
        return name or "file"

    # ── Public API ────────────────────────────────────────────────────────────

    def store_file(self, filename: str, data: bytes, groq_key: str, current_turn: int) -> dict:
        """
        Validate, store, process, and index a user-uploaded file.
        Returns the metadata dict for the file.
        Raises ValueError for validation errors.
        """
        if len(data) > MAX_BYTES:
            mb = len(data) / 1024 / 1024
            raise ValueError(f"File too large ({mb:.1f} MB). Maximum is 30 MB.")

        safe_name = self._safe_name(filename)
        ext = Path(safe_name).suffix.lower()
        if not ext:
            raise ValueError("File has no extension.")
        if ext not in ALLOWED_ALL:
            raise ValueError(
                f"File type '{ext}' not supported. "
                f"Supported: images ({', '.join(sorted(ALLOWED_IMAGE))}), "
                f"docs ({', '.join(sorted(ALLOWED_DOC))}), "
                f"audio ({', '.join(sorted(ALLOWED_AUDIO))})."
            )

        file_id     = str(uuid.uuid4())
        stored_name = f"{file_id}{ext}"
        file_path   = self._dir / stored_name
        file_path.write_bytes(data)

        # Category + processing
        if ext in ALLOWED_IMAGE:
            category = "image"
            summary, memory_text = self._process_image(safe_name, data, ext, groq_key)
        elif ext == ".pdf":
            category = "pdf"
            summary, memory_text = self._process_pdf(file_path, safe_name)
        elif ext in ALLOWED_AUDIO:
            category = "audio"
            summary, memory_text = self._process_audio(safe_name, data, ext, groq_key)
        elif ext == ".docx":
            category = "document"
            summary, memory_text = self._process_docx(file_path, safe_name)
        else:
            category = "text"
            text      = data.decode("utf-8", errors="replace")
            summary   = text[:300]
            memory_text = f"[Text File] {safe_name}:\n{text}"

        # Store chunks to archival memory
        self._store_chunks(file_id, safe_name, category, memory_text, current_turn)

        meta = {
            "file_id":   file_id,
            "name":      safe_name,
            "ext":       ext,
            "category":  category,
            "size":      len(data),
            "stored_as": stored_name,
            "turn":      current_turn,
            "ts":        int(time.time()),
            "summary":   summary,
        }
        self._meta[file_id] = meta
        self._save_meta()
        return meta

    def list_files(self) -> list:
        return sorted(self._meta.values(), key=lambda x: x["ts"], reverse=True)

    def get_file_path(self, file_id: str) -> Optional[Path]:
        meta = self._meta.get(file_id)
        if not meta:
            return None
        p = self._dir / meta["stored_as"]
        return p if p.exists() else None

    def get_file_meta(self, file_id: str) -> Optional[dict]:
        return self._meta.get(file_id)

    def delete_file(self, file_id: str) -> bool:
        meta = self._meta.pop(file_id, None)
        if not meta:
            return False
        p = self._dir / meta["stored_as"]
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
        self._save_meta()
        return True

    # ── Processing ────────────────────────────────────────────────────────────

    def _process_image(self, name: str, data: bytes, ext: str, groq_key: str) -> tuple:
        try:
            from groq import Groq
            mime  = MIME_MAP.get(ext, "image/jpeg")
            b64   = base64.b64encode(data).decode()
            client = Groq(api_key=groq_key)
            completion = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in rich, personal detail as if writing a memory. "
                                "Include: what's shown, people (appearance, names if visible), "
                                "location cues, emotions/mood, visible text, memorable objects. "
                                "Write 3-5 sentences. This will be saved as a personal memory."
                            ),
                        },
                    ],
                }],
                max_tokens=500,
            )
            desc = completion.choices[0].message.content.strip()
        except Exception as ex:
            desc = f"Image uploaded: {name} (AI description unavailable: {ex})"

        memory_text = f"[Photo Memory] {name}: {desc}"
        return desc[:300], memory_text

    def _process_pdf(self, path: Path, name: str) -> tuple:
        full_text = ""
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                pages = []
                for i, page in enumerate(pdf.pages[:60]):
                    t = (page.extract_text() or "").strip()
                    if t:
                        pages.append(f"[Page {i+1}] {t}")
                full_text = "\n\n".join(pages)
        except ImportError:
            pass

        if not full_text:
            try:
                # Try pypdf (modern successor to PyPDF2)
                try:
                    from pypdf import PdfReader
                except ImportError:
                    from PyPDF2 import PdfReader  # type: ignore
                with open(path, "rb") as f:
                    reader = PdfReader(f)
                    texts = []
                    for i, page in enumerate(reader.pages[:60]):
                        t = (page.extract_text() or "").strip()
                        if t:
                            texts.append(f"[Page {i+1}] {t}")
                full_text = "\n\n".join(texts)
            except ImportError:
                full_text = "PDF uploaded — install 'pdfplumber' or 'pypdf' for text extraction"
            except Exception as ex:
                full_text = f"PDF uploaded (extraction error: {ex})"

        if not full_text:
            full_text = "PDF uploaded — no extractable text found (may be a scanned image PDF)"

        summary = (full_text[:300] + "…") if len(full_text) > 300 else full_text
        memory_text = f"[PDF Memory] {name}:\n{full_text}"
        return summary, memory_text

    def _process_docx(self, path: Path, name: str) -> tuple:
        try:
            import docx
            doc  = docx.Document(str(path))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as ex:
            text = f"Document extraction failed: {ex}"
        summary = text[:300]
        memory_text = f"[Document Memory] {name}:\n{text}"
        return summary, memory_text

    def _process_audio(self, name: str, data: bytes, ext: str, groq_key: str) -> tuple:
        try:
            from groq import Groq
            mime   = MIME_MAP.get(ext, "audio/mpeg")
            client = Groq(api_key=groq_key)
            transcription = client.audio.transcriptions.create(
                file=(name, data, mime),
                model="whisper-large-v3",
                response_format="text",
            )
            text = str(transcription).strip()
        except Exception as ex:
            text = f"Audio transcription failed: {ex}"
        summary = text[:300]
        memory_text = f"[Audio Memory] {name}: {text}"
        return summary, memory_text

    # ── Memory storage ────────────────────────────────────────────────────────

    def _store_chunks(
        self, file_id: str, name: str, category: str,
        full_text: str, turn: int, chunk_words: int = 700
    ):
        """Chunk text and store each piece as a semantic memory."""
        words  = full_text.split()
        chunks = []
        for i in range(0, max(1, len(words)), chunk_words):
            chunk = " ".join(words[i : i + chunk_words]).strip()
            if chunk:
                chunks.append(chunk)

        for idx, chunk in enumerate(chunks):
            label = f"[FILE:{category}:{name}]"
            if len(chunks) > 1:
                label += f"[part {idx+1}/{len(chunks)}]"
            self.archive.add_fact(
                content=f"{label} {chunk}",
                turn_number=turn,
                confidence=0.95,
            )
