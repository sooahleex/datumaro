# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import re
from typing import Dict, List, Optional, Tuple

import fitz

from datumaro.components.annotation import AnnotationType, Categories, Tabular, TabularCategories
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Table, TableRow
from datumaro.util.os_util import find_files

PDF_EXTENSIONS = [
    "pdf",
]


class PdfDataBase(DatasetBase):
    NAME = "pdf"

    def __init__(
        self,
        path: str,
        ctx: Optional[ImportContext] = None,
        **kwargs,
    ) -> None:
        super().__init__(media_type=TableRow, ctx=ctx)

        self._infos = {"path": path}
        self._items, self._categories = self._parse(path, **kwargs)

    def _parse_document_structure(self, text):
        """
        Splits the document into sections based on a pattern that matches section titles.
        The pattern assumes sections are numbered and followed by a title starting with a capital letter (e.g., "1. Introduction").
        This helps extract logical sections of the document for further processing.
        """
        sections = re.split(r"\n\s*\d+\.\s*[A-Z][\w\s]+", text)  # Section number + title pattern
        return sections

    def _split_into_paragraphs(self, text):
        """
        Splits the text into paragraphs based on empty lines.
        """
        paragraphs = text.split("\n\n")
        return paragraphs

    def _split_into_chunks(self, text, max_chunk_size=300):
        """
        Splits the text into chunks of a given size.
        """
        chunks = [text[i : i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        return chunks

    def _create_overlap_chunks(self, chunks, overlap_ratio=0.2):
        """
        Creates overlapped chunks based on the given overlap ratio.
        """
        overlap_size = int(
            len(chunks[0]) * overlap_ratio
        )  # Calculate overlap ratio based on the size of the first chunk
        overlapped_chunks = []

        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Create a new chunk including the overlapping part of the current and next chunks
            overlap_chunk = current_chunk + next_chunk[:overlap_size]
            overlapped_chunks.append(overlap_chunk)
        overlapped_chunks.append(chunks[-1])  # Add the last chunk as is
        return overlapped_chunks

    def _parse(
        self,
        path: str,
        **kwargs,
    ) -> Tuple[List[DatasetItem], Dict[AnnotationType, Categories]]:
        items: List[DatasetItem] = []
        categories: TabularCategories = TabularCategories()

        doc = fitz.open(path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()

        sections = self._parse_document_structure(text)

        all_chunks = []
        for section in sections:
            paragraphs = self._split_into_paragraphs(section)
            all_chunks.extend(paragraphs)

        final_chunks = []
        for chunk in all_chunks:
            small_chunks = self._split_into_chunks(chunk, max_chunk_size=300)
            final_chunks.extend(small_chunks)

        overlapped_chunks = self._create_overlap_chunks(final_chunks, overlap_ratio=0.2)
        list_of_dicts = {
            "chunk_id": range(0, len(overlapped_chunks)),
            "chunk_text": overlapped_chunks,
        }

        # Convert to table
        table = Table.from_list(list_of_dicts)

    def categories(self):
        return self._categories

    def __iter__(self):
        yield from self._items


class PdfDataImporter(Importer):
    """
    Import a tabular dataset.
    Each '.csv' file is regarded as a subset.
    """

    NAME = "tabular"

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        return parser

    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            ext = osp.splitext(path)[1][1:]  # exclude "."
            if ext in PDF_EXTENSIONS:
                return [{"url": path, "format": PdfDataBase.NAME}]
        else:
            for _ in find_files(path, PDF_EXTENSIONS):  # find 1 depth only.
                return [{"url": path, "format": PdfDataBase.NAME}]
        return []

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return list({f".{ext}" for ext in PDF_EXTENSIONS})
