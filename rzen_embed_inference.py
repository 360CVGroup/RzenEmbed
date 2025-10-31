from __future__ import annotations
from torch import nn
import logging
import math
import os
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig

from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration


class RzenEmbed(nn.Module):
    def __init__(
        self,
        model_name: str = "RzenAI/RzenEmbed-v2-7B",
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_image_tokens=256,
        max_image_tokens=1280,
        min_video_tokens=160,
        max_video_tokens=180,
        max_length=2000,
        attn_implementation="flash_attention_2",
        processor: Optional[AutoProcessor] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        model_name = model_path or model_name

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config._attn_implementation = attn_implementation
        config.padding_side = "right"
        config.use_cache = False

        self.base = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, config=config,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )

        self.base.eval()
        self.normalize = True
        self.device = device
        self.base = self.base.to(self.device)
        print(f"model.device: {str(self.base.device)}")
        min_pixels = min_image_tokens * 28 * 28
        max_pixels = max_image_tokens * 28 * 28
        self.max_length = max_length
        if processor is None:
            processor = AutoProcessor.from_pretrained(
                model_name, min_pixels=min_pixels, max_pixels=max_pixels
            )
        self.processor = processor
        self.processor.tokenizer.padding_side = 'right'
        self.defualt_instruction = 'You are a helpful assistant.'
        self.sep = ' '

        min_pixels_video = min_video_tokens * 28 * 28
        max_pixels_video = max_video_tokens * 28 * 28 # debug
        self.qwen2vl_video_processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels_video, max_pixels=max_pixels_video
        )
        self.qwen2vl_video_processor.tokenizer.padding_side = 'right'

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        # pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        # video_grid_thw: Optional[torch.LongTensor] = None,
        pooling_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.base.model.embed_tokens(input_ids)
            has_image = (pixel_values is not None) and any([pv is not None for pv in pixel_values])
            if has_image:
                if type(pixel_values) is list:
                    pixel_values = torch.cat([torch.from_numpy(pv) for pv in pixel_values]).to(input_ids.device)  # shape=[BS*n_patch,C*H*W]
                    image_grid_thw = torch.cat([torch.from_numpy(thw) for thw in image_grid_thw]).to(input_ids.device)  # shape=[BS,H,W]
                pixel_values = pixel_values.type(self.base.visual.get_dtype())
                image_embeds = self.base.visual(pixel_values, grid_thw=image_grid_thw).to(inputs_embeds.device)
                image_mask = input_ids == self.base.config.image_token_id
                inputs_embeds[image_mask] = image_embeds

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # print(inputs_embeds.shape)
        outputs = self.base.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        pooling_mask = attention_mask if pooling_mask is None else pooling_mask
        left_padding = (pooling_mask[:, -1].sum() == pooling_mask.shape[0])  # TODO
        if left_padding:
            embeddings = outputs.last_hidden_state[:, -1]
        else:
            sequence_lengths = pooling_mask.sum(dim=1) - 1
            batch_size = outputs.last_hidden_state.shape[0]
            embeddings = outputs.last_hidden_state[torch.arange(
                batch_size, device=outputs.last_hidden_state.device
            ), sequence_lengths]
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()


    def _process_images(self, images):
        """Convert single image or list of images to processed format"""
        if isinstance(images, Image.Image) or isinstance(images, str):
            return [fetch_image(images)]
        return [fetch_image(i) for i in images]


    def embed(self, texts: list[str], images: list[Image.Image], **kwargs):
        # self.base.to(self.device)
        # Inputs must be batched
        if any(isinstance(item, list) for item in images):
            is_video = True
        else:
            is_video = False

        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        # Determine batch size
        
        batch_size = len(texts) if texts is not None else len(images)  # type: ignore

        input_texts, input_images = [], []
        instruction = self.defualt_instruction
        for i in range(batch_size):
            text = texts[i] if texts is not None else None
            image = images[i] if images is not None else None

            input_str = ""
            processed_image = None
            if image is not None:
                processed_image = self._process_images(image)
                input_images += processed_image
                input_str += "<|vision_start|><|image_pad|><|vision_end|>" * len(processed_image)

            if text is not None:
                input_str += text

            msg = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"

            input_texts.append(msg)

        if len(input_images) == 0:
            input_images = None

        if is_video:
            inputs = self.qwen2vl_video_processor(
                text=input_texts,
                images=input_images,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=input_texts,
                images=input_images,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # TODO
        with torch.no_grad():
            embeddings = self.forward(**inputs)
        return embeddings

    def encode(self, sentences: list[str], *, prompt_name=None, **kwargs):
        return self.get_fused_embeddings(texts=sentences, prompt_name=prompt_name, **kwargs)

    def get_image_embeddings(self, images: list[Image.Image] | DataLoader, **kwargs):
        return self.get_fused_embeddings(images=images, **kwargs)

    def get_text_embeddings(self, texts: list[str], **kwargs):
        return self.get_fused_embeddings(texts=texts, **kwargs)

    def get_fused_embeddings(self, texts: list[str] = None, images: list[Image.Image] | DataLoader = None, **kwargs):
        assert texts or images, "Either 'texts' or 'images' must be provided - both cannot be None or empty"
        instruction = kwargs.pop('instruction', None)
        if instruction is not None:
            if texts is not None:
                texts = [instruction + text for text in texts]
            else:
                texts = [instruction] * len(images)

        if isinstance(images, DataLoader):
            image_loader = images
            batch_size = image_loader.batch_size
            image_loader.dataset.transform = None
        else:
            batch_size = kwargs.pop('batch_size', 32)
            if images is None:
                image_loader = None
            else:
                image_loader = DataLoader(
                    images,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=custom_collate_fn,
                    num_workers=min(math.floor(os.cpu_count() / 2), 8),
                )

        if texts is None:
            assert image_loader is not None
            n_batch = len(image_loader)
        else:
            n_batch = len(texts) // batch_size + int(len(texts) % batch_size > 0)
            image_loader = image_loader or [None] * n_batch

        all_embeddings = list()
        none_batch = [None] * batch_size
        show_progress_bar = kwargs.pop('show_progress_bar', False)
        pbar = tqdm(total=n_batch, disable=not show_progress_bar, mininterval=1, miniters=10, desc='encode')
        for n, img_batch in zip(range(0, n_batch * batch_size, batch_size), image_loader):
            text_batch = none_batch[:len(img_batch)] if texts is None else texts[n: n+batch_size]
            img_batch = none_batch[:len(text_batch)] if img_batch is None else img_batch
            embeddings = self.embed(texts=text_batch, images=img_batch, **kwargs)
            pbar.update(1)
            all_embeddings.append(embeddings.cpu())
        pbar.close()
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings


def custom_collate_fn(batch):
    return batch


### Copied from qwen_vl_utils.vision_process.py
import base64
from io import BytesIO
import requests

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if max(h_bar, w_bar) / min(h_bar, w_bar) > MAX_RATIO:
        logging.warning(
            f"Absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(h_bar, w_bar) / min(h_bar, w_bar)}"
        )
        if h_bar > w_bar:
            h_bar = w_bar * MAX_RATIO
        else:
            w_bar = h_bar * MAX_RATIO
    return h_bar, w_bar


def fetch_image(image: str | Image.Image, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        headers = {'User-Agent': 'My User Agent 1.0'}
        image_obj = Image.open(requests.get(image, headers=headers, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")

    width, height = image.size

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    image = image.resize((resized_width, resized_height))

    return image
###


if __name__ == '__main__':
    rzen = RzenEmbed("RzenAI/RzenEmbed-v2-7B")

    queries = [
        "A curious kitten and a gentle puppy share a moment of connection on the grass.",
        "Fresh fridge full of berries yogurt milk and snacks."
    ]
    candidates = [
        "assets/example1.jpg",
        "assets/example2.jpg",
    ]

    query_instruction = "Find me an everyday image that matches the given caption: "
    candidate_instruction = "Represent the given image."

    # Generate embeddings and compute similarity
    query_embeds = rzen.get_fused_embeddings(instruction=query_instruction, texts=queries)
    candidate_embeds = rzen.get_fused_embeddings(instruction=candidate_instruction, images=candidates)

    # Calculate text-to-image similarity scores
    similarity_scores = query_embeds @ candidate_embeds.T
    print(similarity_scores)