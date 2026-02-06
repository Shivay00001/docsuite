
from PIL import Image, ImageDraw, ImageFont

def create_docsuite_icon(path):
    size = (256, 256)
    color = (0, 120, 215)  # DocSuite Blue
    text_color = (255, 255, 255)

    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)

    # Draw a simple document shape
    doc_rect = [64, 32, 192, 224]
    draw.rectangle(doc_rect, outline=text_color, width=8)
    
    # Draw text "DS"
    try:
        # Try to load a font, fallback to default
        font = ImageFont.truetype("arial.ttf", 100)
    except IOError:
        font = ImageFont.load_default()

    # Center text (approximate if default font)
    text = "DS"
    draw.text((80, 80), text, fill=text_color, font=font)

    img.save(path)
    print(f"Created placeholder icon at {path}")

if __name__ == "__main__":
    create_docsuite_icon("c:\\docsuite\\docsuite_app_icon.png")
