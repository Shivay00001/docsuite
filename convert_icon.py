
import subprocess
import sys
import os

def convert_png_to_ico(png_path, ico_path):
    try:
        from PIL import Image
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image

    if not os.path.exists(png_path):
        print(f"Error: {png_path} not found.")
        sys.exit(1)

    img = Image.open(png_path)
    img.save(ico_path, format='ICO', sizes=[(256, 256)])
    print(f"Successfully converted {png_path} to {ico_path}")

if __name__ == "__main__":
    convert_png_to_ico("docsuite_app_icon.png", "icon.ico")
