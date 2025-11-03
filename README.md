# RzenEmbed-v2-7B
[RzenEmbed Paper](https://arxiv.org/abs/2510.27350)

RzenEmbed-v2-7B is a multimodal embedding model developed and open-sourced by 360CVGroup, fine-tuned from Qwen2-VL. It achieves state-of-the-art (SOTA) results on the MMEB-V2, MMEB-Visdoc, and MMEB-Video benchmarks (as of September 29, 2025).

### MMEB-V2

| Model                    | Model Size (B) | Overall   | Image-Overall | Video-Overall | Visdoc-Overall |
| ------------------------ | -------------- | --------- | ------------- | ------------- | -------------- |
| RzenEmbed-v2-7B          | 8.29           | **71.61** | 75.92         | **55.73**     | **77.06**      |
| seed-1.6-embedding       | unknown        | 71.27     | **77.78**     | 55.34         | 73.44          |
| Ops-MM-embedding-v1-7B   | 8.29           | 67.61     | 72.72         | 53.76         | 70.34          |
| Ops-MM-embedding-v1-2B   | 2.21           | 63.44     | 69.03         | 47.56         | 66.96          |
| interestFM-UIR-CAFe-7B   | 8.03           | 60.63     | 67.56         | 42.4          | 63.92          |
| VLM2Vec-V2.0-Qwen2VL-2B  | 2.21           | 58.02     | 64.85         | 34.85         | 65.36          |
| gme-Qwen2-VL-7B-Instruct | 8.29           | 57.83     | 55.95         | 38.43         | 75.18          |
| gme-Qwen2-VL-2B-Instruct | 2.21           | 54.08     | 51.89         | 33.64         | 72.71          |

### MMEB-Image

| Models                 | Model Size(B) | Image-Overall | I-CLS     | I-QA      | I-RET    | I-VG     |
| ---------------------- | ------------- | ------------- | --------- | --------- | -------- | -------- |
| seed-1.6-embedding     | unknown       | **77.78**     | **76.06** | **73.97** | 77.9     | 91.25    |
| RzenEmbed-v2-7B        | 8.29          | 75.92         | 70.61     | 71.67     | **78.5** | **92.1** |
| QQMM-embed-v2          | 8.29          | 75.28         | 72.97     | 71.85     | 76.01    | 87.42    |
| ReCo-7B                | 8.29          | 73.87         | 70.95     | 71.52     | 73.66    | 87.7     |
| OEmbedding-v1-7B       | 8.29          | 72.79         | 70.05     | 68.1      | 73.84    | 88.25    |
| Ops-MM-embedding-v1-7B | 8.29          | 72.72         | 69.65     | 69.58     | 73.09    | 87.15    |
| QQMM-embed             | 8.297         | 72.175        | 70.07     | 69.52     | 71.175   | 87.075   |
| B3_Qwen2_7B            | 8.29          | 72            | 70        | 66.5      | 74.1     | 84.6     |

### MMEB-Video

| Models                   | Model Size(B) | Video-Overall | V-CLS     | V-QA     | V-RET     | V-MRET    |
| ------------------------ | ------------- | ------------- | --------- | -------- | --------- | --------- |
| RzenEmbed-v2-7B          | 8.29          | **55.73**     | 58.82     | **63.5** | 50.97     | 45.54     |
| seed-1.6-embedding       | unknown       | 55.34         | 54.99     | 60.85    | **51.33** | **53.45** |
| Ops-MM-embedding-v1-7B   | 8.29          | 53.76         | **59.68** | 62.22    | 45.72     | 43.21     |
| interestFM-UIR-CAFe-7B   | 8.03          | 42.4          | 35.81     | 58.66    | 34.44     | 39.53     |
| gme-Qwen2-VL-7B-Instruct | 8.29          | 38.43         | 37.44     | 50.35    | 28.37     | 36.96     |
| interestFM-UIR-CAFe-0.5B | 0.894         | 35.87         | 33.9      | 41.72    | 29.69     | 39.69     |
| LamRA-Ret                | 8.29          | 34.96         | 39.27     | 42.6     | 24.26     | 32.84     |
| VLM2Vec-V2.0-Qwen2VL-2B  | 2.21          | 34.58         | 39.3      | 34.32    | 28.77     | 36.82     |

### MMEB-Visdoc

| Models                   | Model Size(B) | Visdoc-Overall | ViDoRe-V1 | ViDoRe-V2 | VisRAG   | VisDoc-OOD |
| ------------------------ | ------------- | -------------- | --------- | --------- | -------- | ---------- |
| RzenEmbed-v2-7B          | 8.29          | **77.06**      | **89.7**  | **60.7**  | **88.7** | 44.38      |
| gme-Qwen2-VL-7B-Instruct | 8.29          | 75.18          | 89.44     | 55.61     | 84.99    | **44.4**   |
| seed-1.6-embedding       | unknown       | 73.44          | 85.53     | 56.57     | 84.74    | 43.14      |
| gme-Qwen2-VL-2B-Instruct | 2.21          | 72.71          | 86.15     | 53.96     | 82.52    | 43.12      |
| colpali-v1.3             | 2.92          | 70.97          | 83.6      | 51.98     | 81.13    | 43.12      |
| Ops-MM-embedding-v1-7B   | 8.29          | 70.34          | 80.05     | 59.59     | 79.32    | 43.34      |
| Ops-MM-embedding-v1-2B   | 2.21          | 66.96          | 76.39     | 53.18     | 77.64    | 41.17      |
| VLM2Vec-V2.0-Qwen2VL-2B  | 2.21          | 65.36          | 75.52     | 44.86     | 79.38    | 39.43      |

## Usage

### Text-to-Image Retrieval

Retrieve images that match text captions.

```python
from rzen_embed_inference import RzenEmbed

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
```

### Image-to-Text Retrieval

Find text captions that best match given images.

```python
from rzen_embed_inference import RzenEmbed

rzen = RzenEmbed("RzenAI/RzenEmbed-v2-7B")

queries = [
    "assets/example1.jpg",
    "assets/example2.jpg",
]
candidates = [
    "A curious kitten and a gentle puppy share a moment of connection on the grass.",
    "Fresh fridge full of berries yogurt milk and snacks."
]

query_instruction = "Find an image caption describing the given everyday image."

query_embeds = rzen.get_fused_embeddings(instruction=query_instruction, images=queries)
candidate_embeds = rzen.get_fused_embeddings(texts=candidates)

# Calculate image-to-text similarity scores
similarity_scores = query_embeds @ candidate_embeds.T
print(similarity_scores)
```

### Document Retrieval

Match text queries with document images for information retrieval.

```python
from rzen_embed_inference import RzenEmbed

rzen = RzenEmbed("RzenAI/RzenEmbed-v2-7B")

queries = [
    "What is the main variable being analyzed on the x-axis of these graphs?",
    "What is the personnel costs in the 4th year?"
]
candidates = [
    "assets/example3.jpg",
    "assets/example4.jpg",
]

query_instruction = "Find a document image that matches the given query: "
candidate_instruction = "Understand the content of the provided document image."

# Generate embeddings for document retrieval
query_embeds = rzen.get_fused_embeddings(instruction=query_instruction, texts=queries)
candidate_embeds = rzen.get_fused_embeddings(instruction=candidate_instruction, images=candidates)

# Calculate text-to-document similarity
similarity_scores = query_embeds @ candidate_embeds.T
print(similarity_scores)
```

### Video Retrieval

Retrieve videos based on text captions.

```python
import cv2
import numpy as np
from rzen_embed_inference import RzenEmbed

def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            break
    cap.release()
    return frames

rzen = RzenEmbed("RzenAI/RzenEmbed-v2-7B")

queries = [
    "A traditional boat glides along a river lined with blooming cherry blossoms under an overcast sky in a modern cityscape.",
    "Tiny ginger kitten meows cutely by the water."
]

# Extract frames from videos
video_path_list = [
    "assets/example5.mp4",
    "assets/example6.mp4",
]
candidates = [extract_frames(video_path, num_frames=8) for video_path in video_path_list]

query_instruction = "Find the video snippet that corresponds to the given caption: "
candidate_instruction = "Understand the content of the provided video."

# Generate embeddings for video retrieval
query_embeds = rzen.get_fused_embeddings(instruction=query_instruction, texts=queries)
candidate_embeds = rzen.get_fused_embeddings(instruction=candidate_instruction, images=candidates)

# Calculate text-to-video similarity scores
similarity_scores = query_embeds @ candidate_embeds.T
print(similarity_scores)
```
