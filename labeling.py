import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import csv
import os
import pandas as pd

class ImageLabeler(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Labeler")
        self.geometry("800x600")

        self.label_var = tk.StringVar(value="Label: None")
        self.image_label = tk.Label(self, text="No Image Loaded")
        self.image_label.pack()

        self.label_display = tk.Label(self, textvariable=self.label_var)
        self.label_display.pack()

        self.progress_var = tk.StringVar(value="Progress: 0/0")
        self.progress_display = tk.Label(self, textvariable=self.progress_var)
        self.progress_display.pack()

        self.unlabeled_var = tk.StringVar(value="Unlabeled Images: 0")
        self.unlabeled_display = tk.Label(self, textvariable=self.unlabeled_var)
        self.unlabeled_display.pack()

        self.load_button = tk.Button(self, text="Load Folder", command=self.load_folder)
        self.load_button.pack()

        self.label_0_button = tk.Button(self, text="Label 0", command=lambda: self.label_image(0))
        self.label_0_button.pack(side=tk.LEFT, padx=10)

        self.label_1_button = tk.Button(self, text="Label 1", command=lambda: self.label_image(1))
        self.label_1_button.pack(side=tk.LEFT, padx=10)

        self.skip_button = tk.Button(self, text="Skip", command=self.next_image)
        self.skip_button.pack(side=tk.LEFT, padx=10)

        self.remove_button = tk.Button(self, text="Remove Image", command=self.remove_image)
        self.remove_button.pack(side=tk.BOTTOM, pady=10)

        self.back_button = tk.Button(self, text="Back", command=self.previous_image)
        self.back_button.pack(side=tk.BOTTOM, pady=10)

        self.filter_var = tk.StringVar(value="Unlabeled")
        self.filter_menu = tk.OptionMenu(self, self.filter_var, "Unlabeled", "None", "0", "1", command=self.apply_filter)
        self.filter_menu.pack(side=tk.BOTTOM, pady=10)

        self.image_files = []
        self.filtered_files = []
        self.current_image_index = -1
        self.labels_file = "./labels.csv"
        self.labels = []

        self.init_labels_file()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_labels_file(self):
        if not os.path.exists(self.labels_file):
            with open(self.labels_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["ImagePath", "Label"])
        else:
            self.load_existing_labels()

    def load_existing_labels(self):
        self.labels_df = pd.read_csv(self.labels_file)
        self.labels = self.labels_df.values.tolist()

    def save_labels(self):
        labels_df = pd.DataFrame(self.labels, columns=["ImagePath", "Label"])
        labels_df.to_csv(self.labels_file, index=False)

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
            existing_paths = set(self.labels_df['ImagePath'])
            new_images = [img for img in self.image_files if img not in existing_paths]
            self.labels.extend([[img, None] for img in new_images])
            self.unlabeled_var.set(f"Unlabeled Images: {len(new_images)}")
            self.apply_filter(self.filter_var.get())
            self.current_image_index = -1
            self.update_progress()
            self.next_image()

    def apply_filter(self, filter_label):
        if filter_label == "Unlabeled":
            self.filtered_files = [f for f in self.labels if f[1] is None]
        elif filter_label == "None":
            self.filtered_files = [f for f in self.labels if f[1] is None]
        else:
            self.filtered_files = [f for f in self.labels if f[1] == filter_label]
        self.current_image_index = -1
        self.update_progress()
        self.next_image()

    def next_image(self):
        self.current_image_index += 1
        if self.current_image_index < len(self.filtered_files):
            self.load_image(self.filtered_files[self.current_image_index][0])
        else:
            messagebox.showinfo("Info", "All images have been labeled or corrected")
            self.image_label.config(image="", text="No Image Loaded")
            self.label_var.set("Label: None")

    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(self.filtered_files[self.current_image_index][0])
        self.update_progress()

    def load_image(self, file_path):
        self.current_image_path = file_path
        image = Image.open(file_path)
        image = image.resize((400, 400), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.image, text="")
        label = self.get_label(file_path)
        self.label_var.set(f"Label: {label if label is not None else 'None'}")
        self.update_progress()

    def get_label(self, image_path):
        for img_path, label in self.labels:
            if img_path == image_path:
                return label
        return None

    def label_image(self, label):
        if self.current_image_path:
            if label not in [0, 1]:
                messagebox.showerror("Error", "Invalid label. Only 0 and 1 are allowed.")
                return
            self.label_var.set(f"Label: {label}")
            self.save_label(self.current_image_path, label)
            self.apply_filter(self.filter_var.get())  # Apply the filter again after labeling

    def save_label(self, image_path, label):
        label = str(label)  # Ensure label is a string
        for img_label in self.labels:
            if img_label[0] == image_path:
                img_label[1] = label
                break
        else:
            self.labels.append([image_path, label])
        self.save_labels()
        self.update_progress()

    def remove_image(self):
        if self.current_image_path:
            os.remove(self.current_image_path)
            self.labels = [label for label in self.labels if label[0] != self.current_image_path]
            self.image_files = [file for file in self.image_files if file != self.current_image_path]
            self.apply_filter(self.filter_var.get())  # Apply the filter again after removal

    def update_progress(self):
        total_images = len(self.filtered_files)
        labeled_images = len([label for _, label in self.filtered_files if label in ['0', '1']])
        self.progress_var.set(f"Progress: {labeled_images}/{total_images}")

    def on_closing(self):
        self.save_labels()
        self.destroy()

if __name__ == "__main__":
    app = ImageLabeler()
    app.mainloop()
