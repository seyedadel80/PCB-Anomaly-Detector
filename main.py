# ==========================================
# Smart PCB Quality Inspection System
# ==========================================

import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import gradio as gr
from typing import Tuple, List, Optional, Any

# ==========================================
# 1. Setup processing device & Constants
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تعریف قالب‌های HTML به صورت ثابت برای تمیزی منطق اصلی کد
HTML_NORMAL_TEMPLATE = """
<div style='
color:#00ffcc;
text-shadow:0 0 15px rgba(0,255,204,0.6);
font-size:28px;
text-align:center;
font-weight:bold;
padding:20px;
border:1px solid #00ffcc;
border-radius:10px;
background:rgba(0,255,204,0.05);'>
✅ NORMAL PCB
<br><br>
<span style='font-size:20px;'>
Anomaly Score : {:.2f}%
</span>
</div>
"""

HTML_ANOMALY_TEMPLATE = """
<div style='
color:#ff3366;
text-shadow:0 0 15px rgba(255,51,102,0.6);
font-size:28px;
text-align:center;
font-weight:bold;
padding:20px;
border:1px solid #ff3366;
border-radius:10px;
background:rgba(255,51,102,0.05);'>
❌ ANOMALY DETECTED
<br><br>
<span style='font-size:20px;'>
Anomaly Score : {:.2f}%
</span>
</div>
"""

HTML_EMPTY_TEMPLATE = "<div style='text-align:center;font-size:20px;'>Please upload an image.</div>"

# ==========================================
# 2. Build the AI Model
# ==========================================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)

# مدیریت خطای ایمن برای بارگذاری وزن‌های مدل
try:
    model.load_state_dict(torch.load("best_pcb_model.pth", map_location=device, weights_only=True))
    print("✅ Model weights loaded successfully.")
except FileNotFoundError:
    print("⚠️ Warning: 'best_pcb_model.pth' not found. Please ensure the weights file is in the root directory.")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")

model.eval()
model.to(device)

# ==========================================
# 3. Prepare image transformations
# ==========================================
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2(),
])


# ==========================================
# 4. Core inspection logic
# ==========================================
# اضافه کردن Type Hinting برای وضوح ورودی و خروجی‌ها
def inspect_pcb(image: Optional[np.ndarray], threshold: float, history: List[dict]) -> Tuple[Any, ...]:
    # Handle empty input
    if image is None:
        return (
            None,
            HTML_EMPTY_TEMPLATE,
            history,
            *([None] * 5)
        )

    original_h, original_w = image.shape[:2]

    # Process image
    augmented = transform(image=image)
    tensor = augmented["image"].unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)
        anomaly_score = float(pred.max().cpu().numpy()) * 100
        pred = pred.squeeze().cpu().numpy()

    binary_mask = (pred > threshold).astype(np.uint8)
    result_image = image.copy()

    # Case 1: NORMAL PCB
    if binary_mask.max() == 0:
        status_html = HTML_NORMAL_TEMPLATE.format(anomaly_score)

    # Case 2: ANOMALY PCB
    else:
        mask_resized = cv2.resize(
            binary_mask,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        contours, _ = cv2.findContours(
            mask_resized,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw empty bounding boxes for defects
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(
                result_image,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                4
            )

        status_html = HTML_ANOMALY_TEMPLATE.format(anomaly_score)

    # Update history with the latest inspection
    history.insert(
        0,
        {
            "input": image,
            "result": result_image,
            "status": status_html
        }
    )

    # Keep only the last 5 items
    history = history[:5]

    # Simple list comprehension instead of loop
    history_imgs = [item["result"] for item in history]

    # Fill the rest with None if history is less than 5
    while len(history_imgs) < 5:
        history_imgs.append(None)

    return (
        result_image,
        status_html,
        history,
        *history_imgs
    )


# ==========================================
# Build UI
# ==========================================
custom_theme = gr.themes.Monochrome(
    neutral_hue="slate",
    spacing_size="lg",
    radius_size="lg"
)

with gr.Blocks(theme=custom_theme, title="AI PCB Inspector") as app:
    history_state = gr.State([])

    # Header with emojis
    gr.HTML("""
    <h1 style="text-align:center; color:#ffffff; margin-bottom:5px;">
    PCB Inspection System ⚙️
    </h1>
    <p style="text-align:center; color:#8b949e; margin-top:0;">
    AI-Powered Anomaly Detection using Deep Learning
    </p>
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Source", height=400)
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="Anomaly Threshold"
            )
            inspect_button = gr.Button("🔍 Run Inspection", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Vision Result", interactive=False, height=400)
            output_status = gr.HTML(label="System Status")

    # History Section
    gr.Markdown("## Recent Inspections")

    with gr.Row():
        hist_img1 = gr.Image(height=120, interactive=False)
        hist_img2 = gr.Image(height=120, interactive=False)
        hist_img3 = gr.Image(height=120, interactive=False)
        hist_img4 = gr.Image(height=120, interactive=False)
        hist_img5 = gr.Image(height=120, interactive=False)

    # Button click event
    inspect_button.click(
        fn=inspect_pcb,
        inputs=[input_image, threshold_slider, history_state],
        outputs=[
            output_image, output_status, history_state,
            hist_img1, hist_img2, hist_img3, hist_img4, hist_img5
        ]
    )


    # ==========================================
    # History Click Events
    # ==========================================
    def load_from_history(index: int, history: List[dict]) -> Tuple[Any, Any, Any]:
        if not history or index >= len(history):
            return None, None, None

        item = history[index]
        return item["input"], item["result"], item["status"]


    # Helper function to avoid late-binding issues in loops for Gradio events
    def create_history_handler(index: int):
        return lambda history: load_from_history(index, history)


    # Refactored click events using the helper function
    hist_img1.select(fn=create_history_handler(0), inputs=history_state,
                     outputs=[input_image, output_image, output_status])
    hist_img2.select(fn=create_history_handler(1), inputs=history_state,
                     outputs=[input_image, output_image, output_status])
    hist_img3.select(fn=create_history_handler(2), inputs=history_state,
                     outputs=[input_image, output_image, output_status])
    hist_img4.select(fn=create_history_handler(3), inputs=history_state,
                     outputs=[input_image, output_image, output_status])
    hist_img5.select(fn=create_history_handler(4), inputs=history_state,
                     outputs=[input_image, output_image, output_status])

# ==========================================
# Launch
# ==========================================
if __name__ == "__main__":
    app.launch(inbrowser=True)