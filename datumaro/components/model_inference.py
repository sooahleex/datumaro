# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import gzip
import html
import os
import warnings
from collections import OrderedDict
from functools import lru_cache
from typing import List, Tuple, Union

import ftfy
import numpy as np
import regex as re
import torch
import torch.nn.functional as F
from PIL import Image
from pkg_resources import packaging
from torch import nn
from torchvision import transforms

import onnx
import onnxruntime as ort
from openvino.inference_engine import IECore

assert 'CUDAExecutionProvider' in ort.get_available_providers()

from datumaro.components.media import MultiframeImage, PointCloud, Video

if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


device = "cuda" if torch.cuda.is_available() else "cpu"

model_folder = "./tests/assets/searcher"

model = {
    "ViT-B/32": None,
    "ViT-B/16": None,
    "ViT-L/14": None,
    "ViT-L/14@336px": None,
}

models = {
    'IMG': None,
    'TXT': None,
}

add_cuda_path = False
if add_cuda_path:
    cuda_dir = 'D:/NVidia/CUDA/v11.0/bin'
    cudnn_dir = 'D:/NVidia/CUDA/v11.0/bin'
    if not (os.path.exists(cuda_dir) and os.path.exists(cudnn_dir)):
        raise ValueError("Please specify correct path for CUDA and cuDNN. Otherwise onnxruntime cannot be imported.")
    else:
        if cuda_dir == cudnn_dir:
            os.environ["PATH"] = cuda_dir + ';' + os.environ["PATH"]
        else:
            os.environ["PATH"] = cuda_dir + ';' + cudnn_dir + ';' + os.environ["PATH"]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


@lru_cache()
def default_bpe():
    return os.path.join(model_folder, "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception as e:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text


def encode_discrete(x):
    x = torch.Tensor(x)
    x = x.to(device, dtype=torch.float)
    prob = torch.sigmoid(x)
    z = torch.sign(prob - 0.5)
    return z


def load_model(model_name: str = "ViT-B/32", jit: bool = False):
    model_ = model[model_name]
    if not model_:
        model_path = os.path.join(model_folder, "ViT-B_32.pth")
        state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(
            set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks"))
        )

        model_ = CLIP(
            embed_dim,
            image_resolution,
            vision_layers,
            vision_width,
            vision_patch_size,
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers,
        )
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]

        model_.load_state_dict(state_dict, device)
        model_ = model_.to(device)
        model_.eval()

        # patch the device names
        device_holder = torch.jit.trace(
            lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
        )
        device_node = [
            n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)
        ][-1]

        def patch_device(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("prim::Constant"):
                    if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                        node.copyAttributes(device_node)

        model_.to(device)
        model_.apply(patch_device)
        patch_device(model_.encode_image)
        patch_device(model_.encode_text)
    model[model_name] = model_

    return model_

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# def _image_features(model, image, model_path):
def _image_features(model=None, image=None, model_path=None, hash_mode=None, input_blob=None, output_blob=None):
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]),
        ]
    )
    img = np.uint8(image)
    img = Image.fromarray(img)

    if np.array(img).ndim == 2 or img.mode == "RGBA":
        img = img.convert("RGB")
    img = trans(img)
    img = np.expand_dims(img, axis=0)
    img = torch.Tensor(img)
    img = img.to(device, dtype=torch.float)

    if hash_mode == 'onnx':
        ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        output = ort_session.run([output_name], {input_name: to_numpy(img)})
        features = output[0]
    elif hash_mode == 'ir' or hash_mode == 'ir_int8':
        h = model.infer(inputs={input_blob: img.cpu()})
        features = torch.from_numpy(h[output_blob])
    else:
        features = model.encode_image(img)
    return features


def _compute_hash(features):
    features = encode_discrete(features)
    features = F.normalize(features, dim=-1)

    features = features.cpu()
    hash_key = features.detach().numpy() >= 0
    hash_key = hash_key * 1
    hash_string = np.packbits(hash_key, axis=-1)
    hash_string = list(map(lambda row: "".join(["{:02x}".format(r) for r in row]), hash_string))
    return hash_string


def load_model_onnx(model_name: str = "ViT-B/32"):
    model_ = model[model_name]
    if not model_:
        model_path = os.path.join(model_folder, "clip_visual_ViT-B_32.onnx")
        model_ = onnx.load(model_path)
        model[model_name] = model_
    return model_


def hash_inference(item, hash_mode='pytorch'):
    assert not type(item) in [
        Video,
        PointCloud,
        MultiframeImage,
    ], f"Media type should be Image, Current type={type(item)}"

    if hash_mode == 'ir':
    # core = ov.Core()
        img_xml_model_path = os.path.join(model_folder, "clip_visual_ViT-B_32.xml")
        img_bin_model_path = os.path.join(model_folder, "clip_visual_ViT-B_32.bin")
        txt_xml_model_path = os.path.join(model_folder, "clip_text_ViT-B_32.xml")
        txt_bin_model_path = os.path.join(model_folder, "clip_text_ViT-B_32.bin")

        ie = IECore()
    elif hash_mode == 'ir_int8':
        img_xml_model_path = os.path.join(model_folder, "clip_visual_ViT-B_32_optimized.xml")
        img_bin_model_path = os.path.join(model_folder, "clip_visual_ViT-B_32_optimized.bin")
        txt_xml_model_path = os.path.join(model_folder, "clip_text_ViT-B_32.xml")
        txt_bin_model_path = os.path.join(model_folder, "clip_text_ViT-B_32.bin")

        ie = IECore()

    elif hash_mode == 'onnx':
        img_model_path = os.path.join(model_folder, "clip_visual_ViT-B_32.onnx")
        txt_model_path = os.path.join(model_folder, "clip_text_ViT-B_32.onnx")

    if isinstance(item, str):
        if len(item.split()) > 1:
            prompt_text = item
        else:
            prompt_text = f"a photo of a {item}"
        text = tokenize(prompt_text).to(device)
        
        if hash_mode == 'ir' or hash_mode == 'ir_int8':
            txt_model = models['TXT']
            if not txt_model:
                txt_net = ie.read_network(txt_xml_model_path, txt_bin_model_path)
                txt_model = ie.load_network(network=txt_net, device_name='CPU')
            input_blob = next(iter(txt_model.input_info))
            output_blob = next(iter(txt_model.outputs))

            h = txt_model.infer(inputs={input_blob: text.cpu()})
            features = torch.from_numpy(h[output_blob])

        elif hash_mode == 'onnx':
            ort_session = ort.InferenceSession(txt_model_path, providers=['CUDAExecutionProvider'])
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            output = ort_session.run([output_name], {input_name: to_numpy(text)})
            features = output[0]
        else:
            model = load_model()
            features = model.encode_text(text)

        
    elif isinstance(item.data, type(None)):
        return []
    else:
        if hash_mode == 'ir' or hash_mode == 'ir_int8':
            img_model = models['IMG']
            if not img_model:
                img_net = ie.read_network(img_xml_model_path, img_bin_model_path)
                img_model = ie.load_network(network=img_net, device_name='CPU')
                input_blob = next(iter(img_model.input_info))
                output_blob = next(iter(img_model.outputs))

            features = _image_features(model=img_model, image=item.data, hash_mode=hash_mode, input_blob=input_blob, output_blob=output_blob)

        elif hash_mode == 'onnx':
            features = _image_features(image=item.data, model_path=img_model_path, hash_mode=hash_mode)
        else:
            model = load_model()
            features = _image_features(model=model, image=item.data)

    hash_string = _compute_hash(features)
    return hash_string


def tokenize(
    texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False
) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    _tokenizer = SimpleTokenizer()

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, : len(tokens)] = torch.tensor(tokens)

    return result
