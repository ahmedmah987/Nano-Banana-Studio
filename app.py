import io
import json
import textwrap
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image, ImageOps
import gradio as gr

# Google Gen AI SDK
from google import genai
from google.genai import types

# ===================== Model Names =====================
TEXT_MODEL  = "gemini-2.5-flash"                 # Text model for prompt enhancement
IMAGE_MODEL = "gemini-2.5-flash-image-preview"   # Image gen/edit model

# ===================== Helpers: Images & Response =====================
def _save_part_to_image_bytes(part) -> Optional[bytes]:
    """Return raw image bytes if the part includes inline_data."""
    if hasattr(part, "inline_data") and part.inline_data is not None:
        return part.inline_data.data
    return None

def _pil_to_inline_image(pil_img: Image.Image, mime="image/png"):
    """Convert PIL.Image to inline Part for the API."""
    buf = io.BytesIO()
    if mime == "image/jpeg":
        pil_img.convert("RGB").save(buf, format="JPEG", quality=95)
    else:
        pil_img.save(buf, format="PNG")
    data = buf.getvalue()
    return types.Part.from_bytes(data=data, mime_type=mime)

def _to_pil(img, force_gray: bool = False) -> Optional[Image.Image]:
    """
    Accepts PIL.Image | numpy.ndarray | bytes | None and returns PIL.Image.
    If force_gray=True => convert to L.
    """
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("L") if force_gray else img
    if isinstance(img, np.ndarray):
        if force_gray:
            if img.ndim == 2:
                pil = Image.fromarray(img.astype(np.uint8), mode="L")
            else:
                pil = Image.fromarray(img.astype(np.uint8)).convert("L")
            return pil
        if img.ndim == 2:
            return Image.fromarray(img.astype(np.uint8), mode="L").convert("RGBA")
        return Image.fromarray(img.astype(np.uint8)).convert("RGBA")
    if isinstance(img, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(img)).convert("RGBA")
        except Exception:
            return None
    try:
        return Image.open(io.BytesIO(img)).convert("RGBA")
    except Exception:
        return None

def _extract_bg_and_mask(editor_data):
    """
    Extract (background_image, mask_image) from Gradio ImageEditor output.
    Supports:
      1) {'image': <np|PIL>, 'mask': <np|PIL>}
      2) {'background': <np|PIL>, 'layers': [dict|np|PIL, ...]}  -> first layer as mask
      3) (image, mask) or [image, mask]
    Returns (PIL.Image, PIL.Image[L]) or (None, None) on failure.
    """
    bg = mask = None

    # Case 1
    if isinstance(editor_data, dict) and "image" in editor_data and "mask" in editor_data:
        bg = _to_pil(editor_data["image"])
        mask = _to_pil(editor_data["mask"], force_gray=True)
        return bg, mask

    # Case 2
    if isinstance(editor_data, dict) and "background" in editor_data:
        bg = _to_pil(editor_data.get("background"))
        layers = editor_data.get("layers", [])
        if isinstance(layers, (list, tuple)) and len(layers) > 0:
            layer0 = layers[0]
            if isinstance(layer0, dict) and "mask" in layer0:
                mask = _to_pil(layer0["mask"], force_gray=True)
            else:
                mask = _to_pil(layer0, force_gray=True)
        return bg, mask

    # Case 3
    if isinstance(editor_data, (list, tuple)) and len(editor_data) >= 2:
        bg = _to_pil(editor_data[0])
        mask = _to_pil(editor_data[1], force_gray=True)
        return bg, mask

    return None, None

def _debug_dump_response(resp) -> str:
    """Human-friendly diagnostics when no image is returned."""
    try:
        if getattr(resp, "text", None):
            return "Model text:\n" + resp.text.strip()
    except Exception:
        pass
    try:
        cand = resp.candidates[0]
        parts = getattr(cand, "content", None)
        parts = parts.parts if parts else []
        lines = []
        for p in parts:
            if getattr(p, "text", None):
                lines.append(p.text)
        if lines:
            return "Model text:\n" + "\n".join(lines)
    except Exception:
        pass
    try:
        return "Raw (truncated):\n" + textwrap.shorten(json.dumps(resp.to_dict(), ensure_ascii=False), width=2000)
    except Exception:
        return "Could not print response."

def _extract_img_from_resp(resp) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Return (PIL.Image, debug_message_or_None)."""
    img_bytes = None
    try:
        cand = resp.candidates[0]
        parts = cand.content.parts if hasattr(cand, "content") else []
        for part in parts:
            img_bytes = _save_part_to_image_bytes(part)
            if img_bytes:
                break
    except Exception:
        pass

    if not img_bytes:
        try:
            finish = getattr(resp.candidates[0], "finish_reason", None)
            finish_name = getattr(finish, "name", str(finish)) if finish else None
            safety = getattr(resp.candidates[0], "safety_ratings", None)
            safety_msg = ""
            if safety:
                lines = []
                for r in safety:
                    cat = getattr(r, "category", None)
                    cat_n = getattr(cat, "name", None) if cat else None
                    prob = getattr(r, "probability", None)
                    prob_n = getattr(prob, "name", None) if prob else None
                    if cat_n or prob_n:
                        lines.append(f"- {cat_n or 'Unknown'}: {prob_n or 'Unknown'}")
                if lines:
                    safety_msg = "Safety:\n" + "\n".join(lines)
            debug_text = _debug_dump_response(resp)
            msg = "Could not extract an image from the response.\n"
            if finish_name:
                msg += f"finish_reason: {finish_name}\n"
            if safety_msg:
                msg += safety_msg + "\n"
            if debug_text:
                msg += "\n" + debug_text
            return None, msg
        except Exception:
            return None, "Could not extract an image; response may be text-only or was blocked by safety."

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        return img, None
    except Exception as e:
        return None, f"Failed to open image: {e}"

# ===================== Aspect/Size Controls =====================
def _parse_aspect_choice(choice: str, orientation: str) -> Optional[float]:
    """
    Convert 'W:H' string to float ratio (W/H). Returns None for 'Original'.
    Applies orientation flip if needed.
    """
    if not choice or choice.lower() == "original":
        return None
    try:
        w, h = choice.split(":")
        w, h = float(w), float(h)
        ratio = w / h if h != 0 else None
        if ratio:
            if orientation.lower() == "portrait" and ratio > 1:
                ratio = 1.0 / ratio
            if orientation.lower() == "landscape" and ratio < 1:
                ratio = 1.0 / ratio
        return ratio
    except Exception:
        return None

def _compute_target_size(aspect_ratio: Optional[float], target_w: Optional[int], target_h: Optional[int], orientation: str) -> Tuple[int, int]:
    """
    Resolve target size (W,H).
    Priority:
      - Use manual W/H if both provided.
      - If only one and AR provided, compute the other.
      - Else, default long side 1024 using AR.
      - If no AR & no manual -> (0,0) meaning 'no resize'.
    """
    if target_w is not None and target_w <= 0: target_w = None
    if target_h is not None and target_h <= 0: target_h = None

    if target_w and target_h:
        return int(target_w), int(target_h)
    if target_w and aspect_ratio:
        h = int(round(target_w / aspect_ratio))
        return int(target_w), max(1, h)
    if target_h and aspect_ratio:
        w = int(round(target_h * aspect_ratio))
        return max(1, w), int(target_h)

    if aspect_ratio:
        if orientation.lower() == "portrait":
            H = 1024
            W = int(round(H * aspect_ratio))
            return max(1, W), H
        else:
            W = 1024
            H = int(round(W / aspect_ratio))
            return W, max(1, H)

    return 0, 0

def _resize_image(img: Image.Image, target_w: int, target_h: int, mode: str) -> Image.Image:
    """Resize with FIT (pad) or FILL (crop)."""
    if target_w <= 0 or target_h <= 0:
        return img

    if mode.lower().startswith("fit"):
        resized = ImageOps.contain(img, (target_w, target_h))
        canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        x = (target_w - resized.width) // 2
        y = (target_h - resized.height) // 2
        canvas.paste(resized, (x, y))
        return canvas

    # Fill (cover + center crop)
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    x1 = (new_w - target_w) // 2
    y1 = (new_h - target_h) // 2
    x2 = x1 + target_w
    y2 = y1 + target_h
    return resized.crop((x1, y1, x2, y2))

def _maybe_apply_size(img: Optional[Image.Image],
                      apply: bool,
                      aspect_choice: str,
                      orientation: str,
                      manual_w: Optional[float],
                      manual_h: Optional[float],
                      resize_mode: str) -> Optional[Image.Image]:
    """Apply resize if requested."""
    if img is None or not apply:
        return img
    mw = int(manual_w) if manual_w and manual_w > 0 else None
    mh = int(manual_h) if manual_h and manual_h > 0 else None
    ar = _parse_aspect_choice(aspect_choice, orientation)
    tw, th = _compute_target_size(ar, mw, mh, orientation)
    if tw <= 0 or th <= 0:
        return img
    return _resize_image(img, tw, th, resize_mode)

# ===================== Client Factory =====================
def get_client_or_error(api_key: str):
    """Create a Gemini client per-call; nothing persisted."""
    api_key = (api_key or "").strip()
    if not api_key:
        return None, "Please enter your API Key first."
    try:
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"Failed to create client: {e}"

# ===================== Core: Prompt Enhance =====================
def enhance_prompt(api_key: str, base_prompt: str, style: str, add_camera: bool, add_lighting: bool) -> str:
    client, err = get_client_or_error(api_key)
    if err:
        return err
    if not base_prompt.strip():
        return "Please write a prompt first…"

    guidance = [
        "You are a prompt engineer for an image generation model.",
        "Rewrite and enhance the user's prompt for Gemini 2.5 Flash Image.",
        "Be specific about composition, subject, scene details, materials, environment, and color palette.",
        "Keep it concise but information-dense; avoid redundancies.",
    ]
    extras = []
    if style:
        extras.append(f"Style: {style}")
    if add_camera:
        extras.append("Include camera directives (focal length, angle, framing).")
    if add_lighting:
        extras.append("Include lighting directives (time of day, key/fill/rim, mood).")

    final_instruction = "\n".join(guidance + extras + [f"User prompt: {base_prompt}"])

    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=final_instruction
    )
    try:
        return resp.text.strip()
    except Exception:
        return base_prompt

# ===================== Core: Generate / Edit / Inpaint (Multi-Output) =====================
def _multi_generate(client, prompt: str, n: int) -> List[Tuple[Optional[Image.Image], Optional[str]]]:
    results = []
    for _ in range(n):
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[types.Part.from_text(text=prompt)]
        )
        img, dbg = _extract_img_from_resp(resp)
        results.append((img, dbg))
    return results

def _multi_edit_text(client, image: Image.Image, instruction: str, n: int) -> List[Tuple[Optional[Image.Image], Optional[str]]]:
    results = []
    img_part = _pil_to_inline_image(image, mime="image/png")
    for _ in range(n):
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[types.Part.from_text(text=instruction), img_part]
        )
        img, dbg = _extract_img_from_resp(resp)
        results.append((img, dbg))
    return results

def _multi_inpaint_mask(client, image: Image.Image, mask: Image.Image, instruction: str, n: int) -> List[Tuple[Optional[Image.Image], Optional[str]]]:
    """
    Note: mask-guided inpaint via instruction (no official mask param).
    """
    results = []
    mask = mask.convert("L")
    img_part  = _pil_to_inline_image(image, mime="image/png")
    mask_part = _pil_to_inline_image(mask,  mime="image/png")
    masked_instruction = (
        "Apply the following edit ONLY within the white regions of the provided mask. "
        "All other areas must remain unchanged. "
        f"Instruction: {instruction}"
    )
    for _ in range(n):
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[types.Part.from_text(text=masked_instruction), img_part, mask_part]
        )
        img, dbg = _extract_img_from_resp(resp)
        results.append((img, dbg))
    return results

def _postprocess_results(results: List[Tuple[Optional[Image.Image], Optional[str]]],
                         apply_resize: bool,
                         aspect_choice: str,
                         orientation: str,
                         manual_w: Optional[float],
                         manual_h: Optional[float],
                         resize_mode: str):
    """
    Converts (img, dbg) pairs into a clean list of PIL images for Gallery.
    - Filters out None (Gallery can't handle None entries).
    - Applies optional resize to valid images.
    - Returns (images, status_message).
    """
    images: List[Image.Image] = []
    msgs: List[str] = []

    for img, dbg in results:
        if img is not None:
            img = _maybe_apply_size(img, apply_resize, aspect_choice, orientation, manual_w, manual_h, resize_mode)
            images.append(img)
        elif dbg:
            msgs.append(dbg)

    ok = len(images)
    total = len(results)
    status = f"Done: {ok} / {total}"
    if msgs:
        status += "\n\nDiagnostics:\n- " + "\n- ".join(msgs[:3])

    # Important: return a list with no None entries (even if empty)
    return images, status

def generate_images(api_key: str,
                    prompt: str,
                    apply_resize: bool,
                    aspect_choice: str,
                    orientation: str,
                    manual_w: Optional[float],
                    manual_h: Optional[float],
                    resize_mode: str,
                    num_outputs: int):
    client, err = get_client_or_error(api_key)
    if err:
        return [], err
    if not prompt.strip():
        return [], "Please write a prompt first."
    try:
        res = _multi_generate(client, prompt, int(num_outputs))
        return _postprocess_results(
            res,
            apply_resize,
            aspect_choice,
            orientation,
            manual_w,
            manual_h,
            resize_mode
        )
    except Exception as e:
        return [], f"API error: {e}"

def edit_images_by_text(api_key: str,
                        image: Image.Image,
                        instruction: str,
                        apply_resize: bool,
                        aspect_choice: str,
                        orientation: str,
                        manual_w: Optional[float],
                        manual_h: Optional[float],
                        resize_mode: str,
                        num_outputs: int):
    client, err = get_client_or_error(api_key)
    if err:
        return [], err
    if image is None:
        return [], "Please upload an image first."
    if not instruction.strip():
        return [], "Please write edit instructions."
    try:
        res = _multi_edit_text(client, image, instruction, int(num_outputs))
        return _postprocess_results(
            res,
            apply_resize,
            aspect_choice,
            orientation,
            manual_w,
            manual_h,
            resize_mode
        )
    except Exception as e:
        return [], f"API error: {e}"

def inpaint_images_with_mask(api_key: str,
                             editor_data,
                             instruction: str,
                             apply_resize: bool,
                             aspect_choice: str,
                             orientation: str,
                             manual_w: Optional[float],
                             manual_h: Optional[float],
                             resize_mode: str,
                             num_outputs: int):
    client, err = get_client_or_error(api_key)
    if err:
        return [], err
    if editor_data is None:
        return [], "Please upload an image and paint a mask (white = edit region)."
    bg, mask_img = _extract_bg_and_mask(editor_data)
    if bg is None or mask_img is None:
        return [], "Could not read background or mask from the editor."
    if not instruction.strip():
        return [], "Please write inpaint instructions."
    try:
        res = _multi_inpaint_mask(client, bg, mask_img, instruction, int(num_outputs))
        return _postprocess_results(
            res,
            apply_resize,
            aspect_choice,
            orientation,
            manual_w,
            manual_h,
            resize_mode
        )
    except Exception as e:
        return [], f"API error: {e}"

# ===================== Gradio UI =====================
with gr.Blocks(title="Nano Banana Studio 🍌🎨") as demo:
    gr.Markdown("## Nano Banana Studio 🍌🎨 – Prompt, Edit & Inpaint UI")
    gr.Markdown("**Enter your API Key for the current session (not stored).**")

    # API Key (session-scoped)
    api_key_box = gr.Textbox(label="Google API Key", placeholder="Paste your key here…", type="password")
    session_key  = gr.State("")
    set_key_btn = gr.Button("Activate Key ✅")
    set_key_btn.click(lambda k: k or "", inputs=[api_key_box], outputs=[session_key])

    with gr.Row():
        with gr.Column(scale=1):
            # Prompt enhancer
            base_prompt = gr.Textbox(label="Base Prompt", lines=6, placeholder="Describe the image you want…")
            style = gr.Textbox(label="Style (optional)", placeholder="cinematic, photorealistic, isometric…")
            add_camera = gr.Checkbox(label="Include Camera Settings", value=True)
            add_lighting = gr.Checkbox(label="Include Lighting Settings", value=True)
            btn_enhance = gr.Button("Enhance Prompt ✨")
            enhanced_prompt = gr.Textbox(label="Enhanced Prompt", lines=8)

            # Sizing controls
            gr.Markdown("### Output Size / Aspect Controls (applied after generation/edit)")
            apply_resize = gr.Checkbox(label="Apply Resize", value=True)
            aspect_choice = gr.Dropdown(
                choices=["Original", "1:1", "3:2", "2:3", "4:3", "16:9", "9:16"],
                value="9:16",
                label="Aspect Ratio"
            )
            orientation = gr.Dropdown(
                choices=["Auto", "Landscape", "Portrait"],
                value="Auto",
                label="Orientation"
            )
            manual_w = gr.Number(label="Manual Width (optional)", value=None, precision=0)
            manual_h = gr.Number(label="Manual Height (optional)", value=None, precision=0)
            resize_mode = gr.Radio(choices=["Fit (Pad)", "Fill (Crop)"], value="Fill (Crop)", label="Resize Mode")

            # Multi-output
            num_outputs = gr.Slider(1, 6, value=1, step=1, label="Number of Outputs")

            btn_generate = gr.Button("Generate Images 🚀")
            gen_status = gr.Markdown("")

        with gr.Column(scale=1):
            gen_gallery = gr.Gallery(label="Results", show_label=True, columns=2)

    # Edit by text tab
    with gr.Tab("Edit by Text"):
        src_img = gr.Image(label="Upload Image", type="pil")
        edit_instruction = gr.Textbox(label="Edit Instruction", lines=4,
                                      placeholder="e.g., make the sky dramatic and add warm sunset light…")
        btn_edit = gr.Button("Apply Edit")
        edit_status = gr.Markdown("")
        edit_gallery = gr.Gallery(label="Edited Results", show_label=True, columns=2)

    # Inpaint tab
    with gr.Tab("Inpaint (Mask-guided)"):
        gr.Markdown("**Paint WHITE where you want changes.** Other areas should remain unchanged.")
        editor = gr.ImageEditor(label="Upload image then paint the mask in white", brush=20, eraser=20)
        mask_instruction = gr.Textbox(label="Inpaint Instruction", lines=4,
                                      placeholder="e.g., replace masked area with a smooth glass curtain wall…")
        btn_inpaint = gr.Button("Run Inpaint")
        inpaint_status = gr.Markdown("")
        inpaint_gallery = gr.Gallery(label="Inpaint Results", show_label=True, columns=2)

    # Wire events
    btn_enhance.click(
        enhance_prompt,
        inputs=[session_key, base_prompt, style, add_camera, add_lighting],
        outputs=enhanced_prompt
    )

    btn_generate.click(
        generate_images,
        inputs=[session_key, enhanced_prompt, apply_resize, aspect_choice, orientation, manual_w, manual_h, resize_mode, num_outputs],
        outputs=[gen_gallery, gen_status]
    )

    btn_edit.click(
        edit_images_by_text,
        inputs=[session_key, src_img, edit_instruction, apply_resize, aspect_choice, orientation, manual_w, manual_h, resize_mode, num_outputs],
        outputs=[edit_gallery, edit_status]
    )

    btn_inpaint.click(
        inpaint_images_with_mask,
        inputs=[session_key, editor, mask_instruction, apply_resize, aspect_choice, orientation, manual_w, manual_h, resize_mode, num_outputs],
        outputs=[inpaint_gallery, inpaint_status]
    )

if __name__ == "__main__":
    # Do NOT enable share=True for public tests with your real key.
    demo.queue().launch()

