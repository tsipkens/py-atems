import os
import numpy as np
import dm3_lib as dm3
from PIL import Image, ImageDraw, ImageFont

def process_dm3():
    # Get current folder name and create a subfolder with that name
    current_folder_name = os.path.basename(os.getcwd())
    out_dir = os.path.join(os.getcwd(), f"{current_folder_name}_png")
    os.makedirs(out_dir, exist_ok=True)
    
    files = [f for f in os.listdir('.') if f.lower().endswith('.dm3')]
    
    if not files:
        return print("No DM3 files found.")

    for fname in files:
        try:
            f = dm3.DM3(fname)
            px_size = f.pxsize[0] * (1000 if f.pxsize[1] == 'micron' else 1)
            
            # Normalize image to 0-255 uint8
            data = f.imagedata.astype(float)
            data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
            img = Image.fromarray(data).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # ----- SCALE BAR -----
            # Calculate scale bar
            w, h = img.size
            bar_nm_raw = px_size * (w * 0.3) 
            
            first_digit = int(str(int(bar_nm_raw))[0])
            mag = 10**(len(str(int(bar_nm_raw))) - 1)
            
            # Round to 1, 2, or 5
            b_val = 1 if first_digit < 2 else 2 if first_digit < 5 else 5
            bar_nm = b_val * mag
            bar_px = int(bar_nm / px_size)
            
            # Drawing parameters
            margin = int(w * 0.05)
            bar_h = max(4, h // 120) # Slightly thicker bar for visibility
            
            # Load a TrueType font to allow scaling. Falls back to default if not found.
            # Adjust 'arial.ttf' to a path on your system if necessary (e.g., /Library/Fonts/ on Mac)
            font_size = int(h * 0.03) # Font size is 3% of image height
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            x1, y1 = w - margin - bar_px, h - margin
            
            # Draw Bar
            draw.rectangle([x1, y1 - bar_h, x1 + bar_px, y1], fill="black")
            
            # Draw Text
            label = f"{int(bar_nm/1000)} um" if bar_nm >= 1000 else f"{int(bar_nm)} nm"
            # Offset text slightly higher to account for larger font size
            draw.text((x1, y1 - bar_h - font_size - 5), label, fill="black", font=font)
            # ----------------------
            
            img.save(os.path.join(out_dir, os.path.splitext(fname)[0] + ".png"))
            print(f"Converted: {fname}")
        except Exception as e:
            print(f"Error {fname}: {e}")

if __name__ == "__main__":
    process_dm3()
    input("\nTask complete. Press Enter to exit...")