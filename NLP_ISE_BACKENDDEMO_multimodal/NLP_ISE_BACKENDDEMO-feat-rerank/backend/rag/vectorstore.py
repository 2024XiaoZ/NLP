"""借鉴 demo 的 LangChain + FAISS 管线，管理本地向量库。"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
_VECTORSTORE: FAISS | None = None


def ensure_vectorstore() -> None:
    """在应用启动阶段预热向量库。"""

    get_vectorstore()


def get_vectorstore() -> FAISS:
    """获取内存中的向量库实例（懒加载）。"""

    global _VECTORSTORE
    if _VECTORSTORE is None:
        _VECTORSTORE = _load_or_build()
    return _VECTORSTORE


def _load_or_build() -> FAISS:
    settings = get_settings()
    store_dir = settings.indexes_dir / "faiss_md_index"
    store_dir.mkdir(parents=True, exist_ok=True)
    embeddings = _get_embeddings()
    index_file = store_dir / "index.faiss"
    docstore_file = store_dir / "docstore.pkl"
    if index_file.exists() and docstore_file.exists():
        logger.info(f"vectorstore.load: path={store_dir}")
        return FAISS.load_local(
            str(store_dir.resolve()),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    docs = _build_documents(settings.data_dir / "fictional_knowledge_base.md")
    if not docs:
        raise RuntimeError("知识库为空：请检查 storage/data/fictional_knowledge_base.md")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    for idx, doc in enumerate(split_docs):
        doc.metadata.setdefault("section", "未命名章节")
        doc.metadata["chunk_id"] = f"chunk-{idx:04d}"

    store = FAISS.from_documents(split_docs, embeddings)
    # 确保目录存在（保存前再次检查）
    store_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(store_dir.resolve()))
    logger.info(f"vectorstore.build_completed: path={store_dir}, chunks={len(split_docs)}")
    return store


def _build_documents(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"未找到知识库源文件：{path}")

    text = path.read_text(encoding="utf-8")
    sections = _split_by_heading(text)
    docs: list[Document] = []
    for title, body in sections:
        cleaned = body.strip()
        if not cleaned:
            continue
        docs.append(Document(page_content=cleaned, metadata={"section": title, "source": str(path)}))
    logger.info(f"vectorstore.docs_ready: sections={len(docs)}, path={path}")
    return docs


def _split_by_heading(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = "引言"
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines)))
                current_lines = []
            current_title = line.lstrip("#").strip() or current_title
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines)))
    return sections


@lru_cache(maxsize=1)
def _get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    if not settings.llm_api_key:
        raise RuntimeError("缺少 LLM_API_KEY，无法初始化向量库嵌入。")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.llm_api_key,
        base_url=str(settings.llm_base_url),
    )



