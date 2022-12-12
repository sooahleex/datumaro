import os.path as osp
from functools import partial
from unittest import TestCase

import numpy as np

<<<<<<< HEAD
from datumaro.components.annotation import AnnotationType, Caption, Label, LabelCategories
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.media import Image
from datumaro.components.model_inference import hash_inference
from datumaro.components.searcher import Searcher
from datumaro.plugins.datumaro_format.converter import DatumaroConverter
=======
import datumaro.plugins.data_formats.voc.format as VOC
from datumaro.components.annotation import AnnotationType, Bbox, Caption, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.media import Image
from datumaro.components.model_inference import hash_inference
from datumaro.components.searcher import Searcher
from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter
from datumaro.plugins.data_formats.voc.exporter import VocExporter
>>>>>>> data_searcher
from datumaro.util.image import load_image
from datumaro.util.test_utils import TestDir

from .requirements import Requirements, mark_requirement

<<<<<<< HEAD
DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), "assets", "datumaro_dataset")
=======

class TestExtractorBase(DatasetBase):
    def _label(self, voc_label):
        return self.categories()[AnnotationType.label].find(voc_label)[0]

    def categories(self):
        return VOC.make_voc_categories()
>>>>>>> data_searcher


class SearcherTest(TestCase):
    @property
    def test_dataset(self):
        train_img = np.full((5, 5, 3), 255, dtype=np.uint8)
        train_img[2, :] = 0
        test_img = np.full((5, 5, 3), 0, dtype=np.uint8)
        test_img[2, :] = 255

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image(data=train_img),
                    annotations=[Label(1, id=1), Caption("cat")],
                ),
                DatasetItem(
                    id=3,
                    subset="test",
                    media=Image(data=test_img),
                    annotations=[Label(2, id=2), Caption("dog")],
                ),
            ]
        )
<<<<<<< HEAD
        with TestDir() as test_dir:
            converter = partial(DatumaroConverter.convert, save_media=True)
            converter(dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro", save_hash=True)
        return imported_dataset
=======
        return dataset
>>>>>>> data_searcher

    @property
    def test_coco_dataset(self):
        SEARCHER_DATASET_DIR = osp.join(osp.dirname(__file__), "assets", "searcher")
        train_img1 = load_image(osp.join(SEARCHER_DATASET_DIR, "000000094852.jpg"))
        train_img2 = load_image(osp.join(SEARCHER_DATASET_DIR, "000000475779.jpg"))
        test_img = load_image(osp.join(SEARCHER_DATASET_DIR, "000000572517.jpg"))

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=train_img1),
                    annotations=[Label(1, id=1), Caption("elephant")],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image(data=train_img2),
                    annotations=[Label(1, id=1), Caption("elephant")],
                ),
                DatasetItem(
                    id=3,
                    subset="test",
                    media=Image(data=test_img),
                    annotations=[Label(2, id=2), Caption("bear")],
                ),
            ]
        )
<<<<<<< HEAD
        with TestDir() as test_dir:
            converter = partial(DatumaroConverter.convert, save_media=True)
            converter(dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro", save_hash=True)
        return imported_dataset
=======
        return dataset
>>>>>>> data_searcher

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inference(self):
        null_image = DatasetItem(
            id="null_img",
            subset="train",
            media=Image(data=np.zeros((3, 4, 3))),
            annotations=[Label(0)],
        )

        hash_key = hash_inference(null_image.media)
        null_hash_key = null_image.set_hash_key

        self.assertEqual(hash_key, null_hash_key)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inference_img_query(self):
<<<<<<< HEAD
        for i, item in enumerate(self.test_dataset):
            if i == 1:
                query = item
        seacher = Searcher(self.test_dataset)
        result = seacher.search_topk(query, topk=2)
        self.assertEqual(query.subset, result[1].subset)
        self.assertEqual(query.hash_key, result[1].hash_key)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inference_txt_query(self):
        seacher = Searcher(self.test_coco_dataset)
        result = seacher.search_topk("elephant", topk=2)
        self.assertEqual(result[0].subset, result[0].subset)
        self.assertEqual(result[0].hash_key, result[0].hash_key)
=======
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro", save_hash=True)
            for i, item in enumerate(imported_dataset):
                if i == 1:
                    query = item
            searcher = Searcher(imported_dataset)
            result = searcher.search_topk(query, topk=2)

            self.assertEqual(query.subset, result[0].subset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inference_txt_query(self):
        with TestDir() as test_dir:
            converter = partial(DatumaroExporter.convert, save_media=True)
            converter(self.test_coco_dataset, test_dir)
            imported_dataset = Dataset.import_from(test_dir, "datumaro", save_hash=True)
            searcher = Searcher(imported_dataset)
            result = searcher.search_topk("elephant", topk=2)

            self.assertEqual(result[0].subset, result[1].subset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_query_data_none(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="frame1",
                            subset="train",
                            media=Image(path="frame1.jpg"),
                            annotations=[
                                Bbox(
                                    1.0,
                                    2.0,
                                    3.0,
                                    4.0,
                                    attributes={
                                        "difficult": False,
                                        "truncated": False,
                                        "occluded": False,
                                    },
                                    id=1,
                                    label=0,
                                    group=1,
                                )
                            ],
                        ),
                        DatasetItem(
                            id="frame2",
                            subset="train",
                            media=Image(path="frame2.jpg"),
                            annotations=[
                                Bbox(
                                    1.0,
                                    2.0,
                                    3.0,
                                    4.0,
                                    attributes={
                                        "difficult": False,
                                        "truncated": False,
                                        "occluded": False,
                                    },
                                    id=2,
                                    label=0,
                                    group=1,
                                )
                            ],
                        ),
                        DatasetItem(
                            id="frame3",
                            subset="test",
                            media=Image(path="frame3.jpg"),
                            annotations=[
                                Bbox(
                                    1.0,
                                    2.0,
                                    3.0,
                                    4.0,
                                    attributes={
                                        "difficult": False,
                                        "truncated": False,
                                        "occluded": False,
                                    },
                                    id=3,
                                    label=1,
                                    group=2,
                                )
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            converter = partial(VocExporter.convert, label_map="voc")
            converter(TestExtractor(), test_dir)
            imported_dataset = Dataset.import_from(test_dir, "voc", save_hash=True)
            for i, item in enumerate(imported_dataset):
                if i == 0:
                    query = item
            searcher = Searcher(imported_dataset)

            self.assertEqual(query.image.data, None)
            with self.assertRaises(Exception):
                searcher.search_topk(query, topk=2)
>>>>>>> data_searcher
