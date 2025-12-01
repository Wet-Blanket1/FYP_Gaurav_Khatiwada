import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os

from model import load_model, preprocess_image, run_model

model = load_model("best_model.path")

class SharpObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sharp Object Detection System")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        self.class_names = ['knife', 'scissor', 'pliers', 'gun', 'wrench']
        self.sharp_objects = ['knife', 'scissor', 'pliers']
        self.non_sharp_objects = ['gun', 'wrench']
        
        self.setup_ui()
    
    def setup_ui(self):
        title = tk.Label(self.root, text="Sharp Object Detection System", 
                        font=("Arial", 20, "bold"), bg="#f0f0f0")
        title.pack(pady=15)
        
        self.img_frame = tk.Frame(self.root, width=700, height=450, bg="#ddd")
        self.img_frame.pack(pady=15)
        self.img_frame.pack_propagate(False)
        
        self.img_label = tk.Label(self.img_frame, bg="#ddd", text="No image selected",
                                 font=("Arial", 12))
        self.img_label.pack(expand=True, fill="both")
        
        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=10)

        #==== Buttons ====
        choose_btn = tk.Button(btn_frame, text="Choose Image", 
                              command=self.choose_image,
                              bg="#4CAF50", fg="white", width=20, height=2)
        choose_btn.pack(side="left", padx=10)
        
        clear_btn = tk.Button(btn_frame, text="Clear", 
                             command=self.clear_image,
                             bg="#f44336", fg="white", width=20, height=2)
        clear_btn.pack(side="left", padx=10)
        
        result_label = tk.Label(self.root, text="Detection Results:", 
                               font=("Arial", 14, "bold"), bg="#f0f0f0")
        result_label.pack(pady=(20, 5))
        
        self.result_box = tk.Text(self.root, height=10, width=85,
                                 state="disabled", bg="white", fg="black",
                                 font=("Arial", 11), relief="solid", bd=1)
        self.result_box.pack(pady=10)
        
        scrollbar = tk.Scrollbar(self.result_box)
        scrollbar.pack(side="right", fill="y")
        self.result_box.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.result_box.yview)
    
    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if not path:
            return
        
        try:
            img_tensor, original = preprocess_image(path)
            boxes, scores, labels = run_model(model, img_tensor)
            
            detections = []
            sharp_found = False
            non_sharp_found = False
            
            for box, score, label in zip(boxes, scores, labels):
                class_name = self.class_names[label]
                confidence = float(score)
                
                if confidence > 0.5:
                    x1, y1, x2, y2 = box
                    sharp_type = 'sharp' if class_name in self.sharp_objects else 'nonsharp'
                    
                    if sharp_type == 'sharp':
                        sharp_found = True
                    else:
                        non_sharp_found = True
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_name': class_name,
                        'confidence': confidence,
                        'sharp_type': sharp_type
                    })
            
            if detections:
                annotated_image = self.draw_detections(original, detections)
                self.display_image(annotated_image)
                self.update_results(detections)
            
            if sharp_found:
                messagebox.showwarning("Warning", "Sharp object detected! Handle carefully.")
            elif non_sharp_found:
                messagebox.showinfo("Info", "No sharp objects detected.")
            else:
                messagebox.showinfo("Info", "No objects detected.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def draw_detections(self, image, detections):
        draw = ImageDraw.Draw(image)
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            sharp_type = detection['sharp_type']
            
            color = "red" if sharp_type == "sharp" else "blue"
            
            draw.rectangle(bbox, outline=color, width=4)
            
            label_text = f"{class_name} {confidence:.2f}"
            text_x = bbox[0] + 10
            text_y = bbox[1] - 35
            
            text_bbox = draw.textbbox((text_x, text_y), label_text)
            draw.rectangle(
                [text_bbox[0]-8, text_bbox[1]-8, text_bbox[2]+8, text_bbox[3]+8], 
                fill=color
            )
            
            draw.text((text_x, text_y), label_text, fill="white")
        
        return image
    
    def display_image(self, image):
        display_width = 650
        display_height = 400
        
        img_width, img_height = image.size
        ratio = min(display_width/img_width, display_height/img_height)
        new_size = (int(img_width * ratio), int(img_height * ratio))
        
        image_resized = image.resize(new_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image_resized)
        
        self.img_label.config(image=photo, text="")
        self.img_label.image = photo
    
    def update_results(self, detections):
        self.result_box.config(state="normal")
        self.result_box.delete(1.0, tk.END)
        
        if not detections:
            self.result_box.insert(tk.END, "No objects detected\n")
        else:
            sharp_count = 0
            non_sharp_count = 0
            
            self.result_box.insert(tk.END, "DETECTION RESULTS:\n")
            self.result_box.insert(tk.END, "=" * 50 + "\n\n")
            
            for detection in detections:
                class_name = detection['class_name']
                confidence = detection['confidence']
                sharp_type = detection['sharp_type']
                
                if sharp_type == "sharp":
                    sharp_count += 1
                    tag = "sharp"
                else:
                    non_sharp_count += 1
                    tag = "nonsharp"
                
                result_text = f"• {class_name.upper()}({sharp_type}) - Confidence: {confidence:.3f}\n"
                self.result_box.insert(tk.END, result_text, tag)
            
            self.result_box.insert(tk.END, "\n" + "=" * 50 + "\n")
            summary = f"SUMMARY:\n"
            summary += f"Sharp objects: {sharp_count}\n"
            summary += f"Non-sharp objects: {non_sharp_count}\n"
            summary += f"Total detected: {len(detections)}"
            
            self.result_box.insert(tk.END, summary)
        
        self.result_box.tag_configure("sharp", foreground="red", font=("Arial", 11, "bold"))
        self.result_box.tag_configure("nonsharp", foreground="blue", font=("Arial", 11))
        self.result_box.config(state="disabled")
    
    def clear_image(self):
        self.img_label.config(image="")
        self.img_label.image = None
        self.result_box.config(state="normal")
        self.result_box.delete(1.0, tk.END)
        self.result_box.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = SharpObjectDetectionApp(root)
    root.mainloop()