"""Local Qwen-VL defect detector (Phase 2).

Runs a Qwen2.5-VL / Qwen2-VL multimodal model fully in-process via Hugging Face
``transformers`` -- no external API or server required. Heavy dependencies
(``torch``, ``transformers``, ``qwen_vl_utils``, ``Pillow``) are imported lazily
and the model is loaded once and cached on first use.

For tests (and to keep the import light), both the image loader and the
inference ``runner`` are injectable.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from backend.app.schemas.inspection import DefectObservation
from backend.app.vision.config import VisionConfig
from backend.app.vision.defect_detector import DefectDetector
from backend.app.vision.vlm_common import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    observation_from_payload,
    open_pil_image,
)

# A runner takes (image, system_prompt, user_prompt) and returns raw model text.
Runner = Callable[[Any, str, str], str]


def _load_qwen(model_id: str, device: str) -> tuple[Any, Any]:
    """Load a Qwen-VL model + processor, tolerating transformers versions."""
    try:
        import transformers  # type: ignore
        from transformers import AutoProcessor  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise RuntimeError(
            "transformers is required for the Qwen detector. Install it with: "
            "pip install -r backend/requirements-vision.txt"
        ) from exc

    model = None
    last_error: Exception | None = None
    for cls_name in (
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
    ):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(
                model_id, torch_dtype="auto", device_map=device
            )
            break
        except Exception as exc:  # noqa: BLE001 - try the next class
            last_error = exc

    if model is None:
        # Generic fallback for newer transformers releases.
        try:
            from transformers import AutoModelForImageTextToText  # type: ignore

            model = AutoModelForImageTextToText.from_pretrained(
                model_id, torch_dtype="auto", device_map=device
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load Qwen-VL model '{model_id}'. "
                f"Last error: {last_error or exc}"
            ) from (last_error or exc)

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


class QwenVLDefectDetector(DefectDetector):
    """Defect detector backed by a local Qwen-VL model (transformers)."""

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        runner: Optional[Runner] = None,
        image_loader: Optional[Callable[[bytes], Any]] = None,
    ) -> None:
        self.config = config or VisionConfig()
        self._runner = runner
        self._image_loader = image_loader or open_pil_image
        self._model: Any = None
        self._processor: Any = None

    def detect(
        self,
        *,
        filename: str,
        image_bytes: Optional[bytes] = None,
        hints: Optional[dict] = None,
    ) -> DefectObservation:
        hints = hints or {}
        if not image_bytes:
            return DefectObservation(
                defect_type="unknown",
                description="No image data provided to the Qwen detector.",
                confidence=0.0,
                location=hints.get("location"),
            )

        image = self._image_loader(image_bytes)
        runner = self._runner or self._default_runner
        content = runner(image, SYSTEM_PROMPT, USER_PROMPT)
        return observation_from_payload(content, hints)

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            self._model, self._processor = _load_qwen(
                self.config.qwen_model, self.config.qwen_device
            )
        return self._model, self._processor

    def _default_runner(self, image: Any, system_prompt: str, user_prompt: str) -> str:
        import torch  # type: ignore
        from qwen_vl_utils import process_vision_info  # type: ignore

        model, processor = self._ensure_model()
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=self.config.qwen_max_new_tokens
            )
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
