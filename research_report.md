# Research Report: Mapping External Repos to Slide Recognition Pipeline

This report analyzes how the `PV2DOC` and `sfguide-video-analysis` repositories relate to our existing slide recognition tool in the CDV pipeline and provides recommendations for integration or improvement.

## 1. Current CDV Pipeline Analysis ([ocr_v3.py](file:///Volumes/SSD%20Daniel/CDV/apps/worker/src/pipeline/python/ocr_v3.py), [slide_ocr_v2.py](file:///Volumes/SSD%20Daniel/CDV/apps/worker/src/pipeline/python/slide_ocr_v2.py))
Our current pipeline is a sophisticated heuristic-based system:
- **Detection**: Perceptional hashing (pHashing), scene cut detection (OpenCV), and regional text likelihood estimation.
- **OCR**: Multi-backend support (Tesseract, EasyOCR, GCV) with regional focus (lower-third, slides, full-frame).
- **Processing**: Semantic classification via regex and keywords, perceptual hash deduplication, and observation fusion.
- **Strengths**: Low compute overhead, fine-grained control over specific screen regions.
- **Weaknesses**: Vulnerable to complex layouts, requires heavy tuning of thresholds (SSIM, pHash distance), and regex-based classification is fragile.

---

## 2. PV2DOC Relationship (`jwr0218/PV2DOC`)
`PV2DOC` (PresentationVideo2Document) focuses on converting presentation videos into structured documents.

### Key Relationships & Potential Improvements for CDV:
- **Object Detection (YOLO/Mask R-CNN)**: `PV2DOC` uses YOLOv5 to detect `figures` and `formulas`. 
    - *Mapping*: We could add a YOLO pass to our [slide_ocr_v2.py](file:///Volumes/SSD%20Daniel/CDV/apps/worker/src/pipeline/python/slide_ocr_v2.py) to identify non-text elements (diagrams, charts) which currently just get "extracted" as a whole slide.
- **Text Clustering**: It uses `AgglomerativeClustering` based on OCR coordinates and word size to reconstruct document layout.
    - *Mapping*: Our current "fusion" logic in `ocr_v2_fusion.py` could be enhanced with this clustering technique to better group bullet points or disjointed text on a slide into coherent blocks.
- **Alternative Change Detection**: Uses SSIM (Structural Similarity Index) for frame difference detection.
    - *Mapping*: We currently use `absdiff` and `mean`. SSIM is generally more robust to lighting changes and noise.

---

## 3. sfguide-video-analysis Relationship (`Snowflake-Labs/...`)
This repository demonstrates a modern, multimodal approach to video analysis.

### Key Relationships & Potential Improvements for CDV:
- **Multimodal AI (Qwen2.5-VL)**: Instead of multi-step heuristics, it uses a Vision-Language Model to extract "key moments" and "semantic events" via natural language prompts.
    - *Mapping*: This is the "North Star" for our `ocr_v4`. We could replace the `classify_text` regex mess and complex scene-cut logic with a single prompt to a VL model (like GPT-4o or Qwen2.5-VL).
- **Semantics over Heuristics**: It focuses on *descriptions* of events rather than just raw OCR text.
    - *Mapping*: Our pipeline is "OCR-first." We extract text and then guess what it is. The Snowflake approach is "Context-first" (e.g., "A slide showing the main sermon point about Grace appears").

---

## 4. Mapping Summary Table

| Feature | CDV (Current) | PV2DOC | SF Guide (Multimodal) |
| :--- | :--- | :--- | :--- |
| **Change Detection** | pHashing / Heuristic | SSIM (Structural Similarity) | Model-determined segments |
| **Logic Type** | Heuristic Rules | Object Detection + Clustering | Multimodal Prompting (Natural Language) |
| **Extraction** | Raw OCR (Tesseract/GCV) | OCR + Formula/Figure Detection | Semantic Descriptions + OCR |
| **Deduplication** | pHash Distance + Text Sim | Spatial Clustering | Semantic Grouping |

---

## 5. Recommendations

1. **Short Term (PV2DOC Inspiration)**:
    - Implement **SSIM** as an alternative or supplement to our current scene-cut detection for better handling of transitions.
    - Adopt **Agglomerative Clustering** for our OCR result fusion to improve the structural quality of the extracted slide text.

2. **Long Term (Snowflake Guide Inspiration)**:
    - Prototype an **`ocr_v4.py`** that uses a multimodal model (like `gpt-4o-mini` or `gemini-1.5-flash`) for the "Slide Recognition" pass. This would likely eliminate the need for much of our current [ocr_events.py](file:///Volumes/SSD%20Daniel/CDV/apps/worker/src/pipeline/python/ocr_events.py) heuristic logic.

3. **Hybrid Approach**: 
    - Use the current fast heuristic pipeline to find "candidate" slide frames, and then use a Multimodal AI (like the SF Guide) to "caption" and "validate" only those frames, rather than running a model on every second of video.
