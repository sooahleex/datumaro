# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import io

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.visualizer import Visualizer

from ..data_loader import DatasetHelper


def main():
    data_helper_1: DatasetHelper = state["data_helper_1"]
    data_helper_2: DatasetHelper = state["data_helper_2"]
    dataset_1 = data_helper_1.dataset()
    dataset_2 = data_helper_2.dataset()
    with elements("visualize"):
        with st.container():
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Parameters")
                selected_subset = st.selectbox("Select a subset:", dataset_1.subsets())
                if selected_subset:
                    ids = [item.id for item in dataset_1.get_subset(selected_subset)]
                    selected_id = st.selectbox("Select a dataset item:", ids)

                if selected_id:
                    item = dataset_1.get(selected_id, selected_subset)
                    ann_ids = [
                        "All",
                    ] + [ann.id for ann in item.annotations]
                    selected_ann_id = st.selectbox("Select a dataset item:", ann_ids)

                selected_alpha = st.select_slider(
                    "Choose a transparency of annotations",
                    options=np.arange(0.0, 1.1, 0.1, dtype=np.float16),
                )

                visualizer = Visualizer(dataset_1, figsize=(8, 8), alpha=selected_alpha)

            with c2:
                st.subheader("Item")
                if selected_ann_id == "All":
                    fig = visualizer.vis_one_sample(selected_id, selected_subset)
                else:
                    fig = visualizer.vis_one_sample(
                        selected_id, selected_subset, ann_id=selected_ann_id
                    )
                fig.set_facecolor("none")

                # Save the Matplotlib figure to a BytesIO buffer as PNG
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png")
                plt.close(fig)
                buffer.seek(0)
                img = Image.open(buffer)

                st.image(img, use_column_width=True)

        with st.container():
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Parameters")
                selected_subset = st.selectbox("Select a subset:", dataset_2.subsets())
                if selected_subset:
                    ids = [item.id for item in dataset_2.get_subset(selected_subset)]
                    selected_id = st.selectbox("Select a dataset item:", ids)

                if selected_id:
                    item = dataset_2.get(selected_id, selected_subset)
                    ann_ids = [
                        "All",
                    ] + [ann.id for ann in item.annotations]
                    selected_ann_id = st.selectbox("Select a dataset item:", ann_ids)

                selected_alpha = st.select_slider(
                    "Choose a transparency of annotations",
                    options=np.arange(0.0, 1.1, 0.1, dtype=np.float16),
                )

                visualizer = Visualizer(dataset_2, figsize=(8, 8), alpha=selected_alpha)

            with c2:
                st.subheader("Item")
                if selected_ann_id == "All":
                    fig = visualizer.vis_one_sample(selected_id, selected_subset)
                else:
                    fig = visualizer.vis_one_sample(
                        selected_id, selected_subset, ann_id=selected_ann_id
                    )
                fig.set_facecolor("none")

                # Save the Matplotlib figure to a BytesIO buffer as PNG
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png")
                plt.close(fig)
                buffer.seek(0)
                img = Image.open(buffer)

                st.image(img, use_column_width=True)
