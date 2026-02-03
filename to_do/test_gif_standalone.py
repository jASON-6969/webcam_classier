"""
Standalone test: open a window and play one GIF.
Run: python to_do/test_gif_standalone.py
If this plays, Tkinter GIF works; if not, the issue is in display/timing.
"""
import os
import sys
import tkinter as tk
from PIL import Image, ImageTk

def main():
    base = os.path.join(os.path.dirname(__file__), 'image', 'class1')
    gif_path = os.path.join(base, 'cool-fun.gif')
    if not os.path.exists(gif_path):
        print(f"GIF not found: {gif_path}")
        return

    root = tk.Tk()
    root.title("GIF test")
    root.geometry("320x320")

    pil = Image.open(gif_path)
    frames_pil = []
    delays = []
    default_delay = pil.info.get('duration', 100)
    n = 0
    while True:
        try:
            pil.seek(n)
            f = pil.copy().convert('RGB')
            frames_pil.append(f)
            d = pil.info.get('duration', default_delay)
            delays.append(max(50, min(2000, int(d))))
            n += 1
        except EOFError:
            break

    if not frames_pil:
        print("No frames")
        root.destroy()
        return

    photo_frames = [ImageTk.PhotoImage(img) for img in frames_pil]
    label = tk.Label(root, image=photo_frames[0])
    label.pack(expand=True, fill=tk.BOTH)
    label.image = photo_frames[0]

    idx = [0]
    def tick():
        i = idx[0] % len(photo_frames)
        label.configure(image=photo_frames[i])
        label.image = photo_frames[i]
        idx[0] += 1
        delay = delays[i] if i < len(delays) else 100
        root.after(delay, tick)

    root.after(100, tick)
    root.mainloop()

if __name__ == "__main__":
    main()
