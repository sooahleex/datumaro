# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import shutil
from unittest.case import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories, Mask
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.open_images import OpenImagesExporter, OpenImagesImporter
from datumaro.util.test_utils import TestDir, compare_datasets, get_hash_key

from tests.requirements import Requirements, mark_requirement


class OpenImagesFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_274)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a", subset="train", annotations=[Label(0, attributes={"score": 0.7})]
                ),
                DatasetItem(
                    id="b",
                    subset="train",
                    media=Image(data=np.zeros((8, 8, 3))),
                    annotations=[
                        Label(1),
                        Label(2, attributes={"score": 0}),
                        Bbox(label=0, x=4, y=3, w=2, h=3),
                        Bbox(
                            label=1,
                            x=2,
                            y=3,
                            w=6,
                            h=1,
                            group=1,
                            attributes={
                                "score": 0.7,
                                "occluded": True,
                                "truncated": False,
                                "is_group_of": True,
                                "is_depiction": False,
                                "is_inside": False,
                            },
                        ),
                        Mask(label=0, image=np.eye(8)),
                        Mask(
                            label=1,
                            image=np.ones((8, 8)),
                            group=1,
                            attributes={
                                "box_id": "00000000",
                                "predicted_iou": 0.75,
                            },
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        "/m/0",
                        ("/m/1", "/m/0"),
                        "/m/2",
                    ]
                ),
            },
        )

        expected_dataset = Dataset.from_extractors(source_dataset)
        expected_dataset.put(
            DatasetItem(
                id="b",
                subset="train",
                media=Image(data=np.zeros((8, 8, 3))),
                annotations=[
                    # the converter assumes that annotations without a score
                    # have a score of 100%
                    Label(1, attributes={"score": 1}),
                    Label(2, attributes={"score": 0}),
                    # Box group numbers are reassigned sequentially.
                    Bbox(label=0, x=4, y=3, w=2, h=3, group=1, attributes={"score": 1}),
                    Bbox(
                        label=1,
                        x=2,
                        y=3,
                        w=6,
                        h=1,
                        group=2,
                        attributes={
                            "score": 0.7,
                            "occluded": True,
                            "truncated": False,
                            "is_group_of": True,
                            "is_depiction": False,
                            "is_inside": False,
                        },
                    ),
                    # Box IDs are autogenerated for masks that don't have them.
                    # Group numbers are assigned to match the corresponding boxes,
                    # if any.
                    Mask(
                        label=0,
                        image=np.eye(8),
                        attributes={
                            "box_id": "00000001",
                        },
                    ),
                    Mask(
                        label=1,
                        image=np.ones((8, 8)),
                        group=2,
                        attributes={
                            "box_id": "00000000",
                            "predicted_iou": 0.75,
                        },
                    ),
                ],
            ),
        )

        with TestDir() as test_dir:
            OpenImagesExporter.convert(source_dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "open_images")

            compare_datasets(self, expected_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_274)
    def test_can_save_and_load_with_no_subsets(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="a", annotations=[Label(0, attributes={"score": 0.7})]),
            ],
            categories=["/m/0"],
        )

        with TestDir() as test_dir:
            OpenImagesExporter.convert(source_dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "open_images")

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_274)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id="a/1", media=Image(path="a/1.JPEG", data=np.zeros((4, 3, 3)))),
                DatasetItem(
                    id="b/c/d/2", media=Image(path="b/c/d/2.bmp", data=np.zeros((3, 4, 3)))
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            OpenImagesExporter.convert(dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "open_images")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_274)
    def test_inplace_save_writes_only_updated_data(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "a",
                    subset="modified",
                    media=Image(data=np.ones((2, 1, 3))),
                    annotations=[
                        Label(0, attributes={"score": 1}),
                        Bbox(0, 0, 1, 2, label=0),
                        Mask(label=0, image=np.ones((2, 1))),
                    ],
                ),
                DatasetItem(
                    "b",
                    subset="modified",
                    media=Image(data=np.ones((2, 1, 3))),
                    annotations=[
                        Label(1, attributes={"score": 1}),
                    ],
                ),
                DatasetItem(
                    "c",
                    subset="removed",
                    media=Image(data=np.ones((3, 2, 3))),
                    annotations=[Label(2, attributes={"score": 1})],
                ),
                DatasetItem(
                    "d",
                    subset="unmodified",
                    media=Image(data=np.ones((4, 3, 3))),
                    annotations=[Label(3, attributes={"score": 1})],
                ),
            ],
            categories=["/m/0", "/m/1", "/m/2", "/m/3"],
        )

        with TestDir() as path:
            dataset.export(path, "open_images", save_media=True)

            dataset.put(
                DatasetItem(
                    "e",
                    subset="new",
                    media=Image(data=np.ones((5, 4, 3))),
                    annotations=[Label(1, attributes={"score": 1})],
                )
            )
            dataset.remove("c", subset="removed")
            del dataset.get("a", subset="modified").annotations[1:3]
            dataset.save(save_media=True)

            self.assertEqual(
                {
                    "bbox_labels_600_hierarchy.json",
                    "class-descriptions.csv",
                    "modified-annotations-human-imagelabels.csv",
                    "modified-images-with-rotation.csv",
                    "new-annotations-human-imagelabels.csv",
                    "new-images-with-rotation.csv",
                    "unmodified-annotations-human-imagelabels.csv",
                    "unmodified-images-with-rotation.csv",
                },
                set(os.listdir(osp.join(path, "annotations"))),
            )

            expected_images = {f"{id}.jpg" for id in ["a", "b", "d", "e"]}

            actual_images = {
                file_name
                for _, _, file_names in os.walk(osp.join(path, "images"))
                for file_name in file_names
            }

            self.assertEqual(actual_images, expected_images)

            dataset_reloaded = Dataset.import_from(path, "open_images")
            compare_datasets(self, dataset, dataset_reloaded, require_media=True)

    @mark_requirement(Requirements.DATUM_BUG_466)
    def test_can_save_and_load_without_saving_images(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    media=Image(data=np.ones((5, 5, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=0, group=1, attributes={"score": 1.0}),
                        Mask(
                            label=1,
                            group=0,
                            image=np.ones((5, 5)),
                            attributes={"box_id": "00000000"},
                        ),
                    ],
                )
            ],
            categories=["label_0", "label_1"],
        )

        with TestDir() as test_dir:
            OpenImagesExporter.convert(dataset, test_dir)

            parsed_dataset = Dataset.import_from(test_dir, "open_images")

            compare_datasets(self, dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_274)
    def test_can_save_and_load_with_meta_file(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    media=Image(data=np.ones((5, 5, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=0, group=1, attributes={"score": 1.0}),
                        Mask(
                            label=1,
                            group=0,
                            image=np.ones((5, 5)),
                            attributes={"box_id": "00000000"},
                        ),
                    ],
                )
            ],
            categories=["label_0", "label_1"],
        )

        with TestDir() as test_dir:
            OpenImagesExporter.convert(dataset, test_dir, save_media=True, save_dataset_meta=True)

            parsed_dataset = Dataset.import_from(test_dir, "open_images")

            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))
            compare_datasets(self, dataset, parsed_dataset, require_media=True)


ASSETS_DIR = osp.join(osp.dirname(__file__), "assets")

DUMMY_DATASET_DIR_V6 = osp.join(ASSETS_DIR, "open_images_dataset/v6")
DUMMY_DATASET_DIR_V5 = osp.join(ASSETS_DIR, "open_images_dataset/v5")


class OpenImagesImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_274)
    def test_can_import_v6(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="train",
                    media=Image(data=np.zeros((8, 6, 3))),
                    annotations=[Label(label=0, attributes={"score": 1})],
                ),
                DatasetItem(
                    id="b",
                    subset="train",
                    media=Image(data=np.zeros((2, 8, 3))),
                    annotations=[
                        Label(label=0, attributes={"score": 0}),
                        Bbox(label=0, x=1.6, y=0.6, w=6.4, h=0.4, group=1, attributes={"score": 1}),
                        Mask(
                            label=0,
                            image=np.hstack((np.ones((2, 2)), np.zeros((2, 6)))),
                            group=1,
                            attributes={
                                "box_id": "01234567",
                                "predicted_iou": 0.5,
                            },
                        ),
                    ],
                ),
                DatasetItem(
                    id="c",
                    subset="test",
                    media=Image(data=np.ones((10, 5, 3))),
                    annotations=[
                        Label(label=1, attributes={"score": 1}),
                        Label(label=3, attributes={"score": 1}),
                        Bbox(
                            label=3,
                            x=3.5,
                            y=0,
                            w=0.5,
                            h=5,
                            group=1,
                            attributes={
                                "score": 0.7,
                                "occluded": True,
                                "truncated": False,
                                "is_group_of": True,
                                "is_depiction": False,
                                "is_inside": False,
                            },
                        ),
                    ],
                ),
                DatasetItem(
                    id="d",
                    subset="validation",
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        # The hierarchy file in the test dataset also includes a fake label
                        # /m/x that is set to be /m/0's parent. This is to mimic the real
                        # Open Images dataset, that assigns a nonexistent label as a parent
                        # to all labels that don't have one.
                        "/m/0",
                        ("/m/1", "/m/0"),
                        "/m/2",
                        "/m/3",
                    ]
                ),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR_V6, "open_images")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_274)
    def test_can_import_v5(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="aa", subset="train", media=Image(data=np.zeros((8, 6, 3)))),
                DatasetItem(id="cc", subset="test", media=Image(data=np.ones((10, 5, 3)))),
            ],
            categories=[
                "/m/0",
                "/m/1",
            ],
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR_V5, "open_images")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_274)
    def test_can_import_without_image_ids_file(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="a",
                    subset="train",
                    media=Image(data=np.zeros((8, 6, 3))),
                    annotations=[Label(label=0, attributes={"score": 1})],
                ),
                DatasetItem(
                    id="b",
                    subset="train",
                    media=Image(data=np.zeros((2, 8, 3))),
                    annotations=[
                        Label(label=0, attributes={"score": 0}),
                        Bbox(label=0, x=1.6, y=0.6, w=6.4, h=0.4, group=1, attributes={"score": 1}),
                        Mask(
                            label=0,
                            image=np.hstack((np.ones((2, 2)), np.zeros((2, 6)))),
                            group=1,
                            attributes={
                                "box_id": "01234567",
                                "predicted_iou": 0.5,
                            },
                        ),
                    ],
                ),
                DatasetItem(
                    id="c",
                    subset="test",
                    media=Image(data=np.ones((10, 5, 3))),
                    annotations=[
                        Label(label=1, attributes={"score": 1}),
                        Label(label=3, attributes={"score": 1}),
                        Bbox(
                            label=3,
                            x=3.5,
                            y=0,
                            w=0.5,
                            h=5,
                            group=1,
                            attributes={
                                "score": 0.7,
                                "occluded": True,
                                "truncated": False,
                                "is_group_of": True,
                                "is_depiction": False,
                                "is_inside": False,
                            },
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        "/m/0",
                        ("/m/1", "/m/0"),
                        "/m/2",
                        "/m/3",
                    ]
                ),
            },
        )

        with TestDir() as test_dir:
            dataset_path = osp.join(test_dir, "dataset")
            shutil.copytree(DUMMY_DATASET_DIR_V6, dataset_path)
            os.remove(osp.join(dataset_path, "annotations", "image_ids_and_rotation.csv"))

            dataset = Dataset.import_from(dataset_path, "open_images")

            compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_274)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR_V6)
        self.assertEqual([OpenImagesImporter.NAME], detected_formats)
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR_V5)
        self.assertEqual([OpenImagesImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_274)
    def test_save_hash_v6(self):
        imported_dataset = Dataset.import_from(DUMMY_DATASET_DIR_V6, "open_images", save_hash=True)
        for item in imported_dataset:
            self.assertTrue(bool(get_hash_key(item)))

    @mark_requirement(Requirements.DATUM_274)
    def test_save_hash_v5(self):
        imported_dataset = Dataset.import_from(DUMMY_DATASET_DIR_V5, "open_images", save_hash=True)
        for item in imported_dataset:
            self.assertTrue(bool(get_hash_key(item)))
