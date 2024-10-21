# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pytest

from datumaro.components.annotation import AnnotationType, TabularCategories
from datumaro.components.dataset import Dataset
from datumaro.plugins.data_formats.tabular import *

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path


@pytest.fixture()
def fxt_pdf_root():
    yield get_test_asset_path("pdf_dataset")


@pytest.fixture()
def fxt_epstein(fxt_pdf_root):
    path = osp.join(fxt_pdf_root, "Epstein_documents.pdf")
    yield Dataset.import_from(path, "pdf")


@pytest.mark.new
class PdfImporterTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_tabular_file(self, fxt_epstein) -> None:
        dataset: Type[Dataset] = fxt_epstein
        expected_categories = {AnnotationType.tabular: TabularCategories.from_iterable([])}
        expected_subset = "electricity"

        assert dataset.categories() == expected_categories
        assert len(dataset) == 100
        assert set(dataset.subsets()) == {expected_subset}

        for idx, item in enumerate(dataset):
            assert idx == item.media.index
            assert len(item.annotations) == 0
