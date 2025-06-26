import tkinter as tk
from tkinter import ttk, messagebox, StringVar, IntVar
import cv2
from pygrabber.dshow_graph import FilterGraph
from PIL import Image, ImageTk
import threading
import numpy as np
import time
import supervision as sv
from ultralytics import YOLO
from utils import parse_model_name
from utils import check_ppe_violation


window = tk.Tk()
window.geometry("1000x600")
window.minsize(width=1000, height=600)
window.title("Real-Time PPE Detection with YOLO")

# Variabel global
grid_mode = IntVar(value=1)
comboboxes = []
frames = []
captures = {}
camera_threads = []
camera_caps = []
camera_frames = {}
running = False
alert_labels = {}  # key: index, value: tk.Label
streaming_running = False  # global status streaming

def detect_available_cameras():
    devices = FilterGraph().get_input_devices()

    available = []

    for idx, name in enumerate(devices):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append((idx, name))
            cap.release()

    return available

# def refresh_sources():
#     available_cameras = detect_available_cameras()
#     clear_comboboxes()
#     build_frames
#     build_frames()

# Event saat grid diubah
def change_grid():
    global streaming_running
    clear_comboboxes()
    build_comboboxes()
    build_frames()
    if streaming_running:
        stop_streaming()
        start_camera_button.config(text="Start", state="normal")
        streaming_running = False

def clear_comboboxes():
    for cb in comboboxes:
        cb.destroy()
    comboboxes.clear()


def build_comboboxes():
    camera_names = [name for _, name in available_cameras]

    for i in range(grid_mode.get()):
        cb = ttk.Combobox(sources_list_frame, values=camera_names, state="readonly")
        cb.set(f"Choose Camera #{i+1}")
        cb.pack(side="top", fill="x", anchor="e", pady=(0,2))
        comboboxes.append(cb)


def build_frames():
    global alert_labels
    for label in alert_labels.values():
        label.destroy()
    alert_labels.clear()

    # print(alert_labels)

    for f in frames:
        f.destroy()
    frames.clear()

    for i in range(4):  # misalnya kamu bisa pakai max 10 baris/kolom
        camera_frame.rowconfigure(i, weight=0)
        camera_frame.columnconfigure(i, weight=0)
    rows = cols = 1 if grid_mode.get() == 1 else 2

    # Konfigurasi grid camera_frame
    for r in range(rows):
        camera_frame.rowconfigure(r, weight=1)
    for c in range(cols):
        camera_frame.columnconfigure(c, weight=1)

    for i in range(grid_mode.get()):
        cam = tk.Label(camera_frame, text=f"No Source #{i+1}", bg="black", fg="white", relief=tk.GROOVE)
        cam.config(width=1, height=1)
        cam.grid(row=i // cols, column=i % cols, sticky="nsew", padx=4, pady=4)
        frames.append(cam)

        alert_label = tk.Label(alert_frame, text="Status: Kamera Mati", fg="blue", font=("Arial", 11), borderwidth=1, relief="groove")
        alert_label.pack(fill="x", pady=4)
        alert_labels[i] = alert_label

def camera_worker(index, cam_index, selected_model, conf_value, iou_value):
    global running
    cap = cv2.VideoCapture(cam_index)
    camera_caps.append(cap)

    # Load model lokal langsung (tidak perlu WebSocket)
    model_version, model_weight = parse_model_name(selected_model)
    model_path = f"./model/{model_version}/onnx/{model_weight}-detect-ppe-img640/1/best.pt"
    current_model = YOLO(model_path, task="detect")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(
        text_scale=0.3,
        text_position=sv.geometry.core.Position("CENTER")
    )

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Prediksi langsung
        results = current_model.predict(frame, imgsz=640, conf=conf_value, iou=iou_value, device=0)
        result = results[0]
        detections = sv.Detections.from_ultralytics(result)

        # Annotasi box
        annotated_image = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        # Tambah label
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(detections['class_name'], detections.confidence)
        ]
        annotated_image = label_annotator.annotate(
            scene=annotated_image.copy(),
            detections=detections,
            labels=labels
        )

        # Opacity (optional, bisa dihapus)
        opacity = 0.6
        overlay = annotated_image.copy()
        output = frame.copy()
        output = cv2.addWeighted(overlay, opacity, output, 1 - opacity, 0)

        # Cek alert
        cls = result.boxes.cls.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()
        names = result.names
        alert_messages = check_ppe_violation(cls, xyxy, names)

        # Update frame untuk ditampilkan
        camera_frames[index] = output

        # Update UI label alert
        if alert_messages:
            alert_text = "\n".join(alert_messages)
            alert_labels[index].config(text=alert_text, fg="red")
        else:
            alert_labels[index].config(text="✅ Aman", fg="green")

        # Tunggu sebentar agar tidak terlalu cepat
        time.sleep(0.03)

def update_frames():
    for i, label in enumerate(frames):
        frame = camera_frames.get(i)
        if frame is not None:
            h_frame, w_frame = frame.shape[:2]
            aspect_ratio = w_frame / h_frame

            parent_width = label.winfo_width()
            parent_height = label.winfo_height()
            # print(parent_height, parent_width)
            # Perhitungan target size
            target_width = parent_width
            target_height = int(parent_width / aspect_ratio)

            if target_height > parent_height:
                target_height = parent_height
                target_width = int(parent_height * aspect_ratio)

            resized_frame = cv2.resize(frame, (target_width, target_height))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(resized_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            label.imgtk = imgtk
            label.config(image=imgtk)
    if running:
        window.after(30, update_frames)

def toggle_streaming():
    global streaming_running
    if streaming_running:
        stop_streaming()
        start_camera_button.config(text="Start", state="normal")
        streaming_running = False
    else:
        # Cek model dan kamera di start_streaming, tapi kita harus modifikasi sedikit
        if not start_streaming():
            # start_streaming mengembalikan False kalau gagal (misal model belum dipilih)
            return
        start_camera_button.config(text="Stop", state="normal")
        streaming_running = True

def start_streaming():
    selected_model = combobox_model_input.get()
    conf_value = float(conf_var.get())
    iou_value = float(iou_var.get())

    if selected_model == "Choose the model":
        messagebox.showwarning("Model Not Selected", "Please select the object detection model.")
        combobox_model_input.focus_set()
        return False # Hentikan start_streaming jika model belum dipilih


    global running, camera_threads, camera_caps, camera_frames
    stop_streaming()  # stop existing first if any
    camera_threads = []
    camera_caps = []
    camera_frames = {}
    running = True
    valid_cameras = []

    for i, cb in enumerate(comboboxes):
        selected_name = cb.get()
        if selected_name != "Pilih Kamera":
            # cari index dari available_cameras berdasarkan nama kamera
            match = next((idx for idx, name in available_cameras if name == selected_name), None)
            if match is not None:
                valid_cameras.append((i, match))

    # print(valid_cameras)
    # Minimal harus ada 1 kamera valid, kalau tidak tampilkan pesan dan batalkan start
    if not valid_cameras:
        messagebox.showwarning("Warning", "At least one camera must be selected.")
        return False

    # Jalankan thread hanya untuk kamera yang valid
    for i, cam_index in valid_cameras:
        t = threading.Thread(target=camera_worker, args=(i, cam_index, selected_model, conf_value, iou_value))
        t.daemon = True
        t.start()
        camera_threads.append(t)

    update_frames()
    return True

def stop_streaming():
    # start_camera_button.config(state="normal")
    global running
    running = False
    for cap in camera_caps:
        if cap.isOpened():
            cap.release()
    camera_caps.clear()
    camera_frames.clear()
    build_frames()

# main layout widgets
menu_frame = tk.Frame(window, borderwidth=4)
main_frame = ttk.Frame(window, borderwidth=4)

menu_frame.pack(side="left", fill="y")
menu_frame.pack_propagate(False)
menu_frame.config(width=300)

main_frame.pack(side="left", fill="both", expand=True) 


# sources frame
sources_frame = ttk.Frame(menu_frame, borderwidth=10, relief=tk.GROOVE)
sources_frame.place(x=0, rely=0, relwidth=1, relheight=0.33)

sources_title = ttk.Label(sources_frame, text="Sources", font=('Arial', 12, 'bold'), anchor="center")
sources_title.pack(fill='x', side="top", pady=(0, 4))

sources_button_frame = ttk.Frame(sources_frame)
sources_button_frame.pack(fill='x', side='top', pady=(0, 4))

# refresh_sources_button = ttk.Button(sources_button_frame, text="Refresh Sources")
# refresh_sources_button.pack(side="right")

grid_option_frame = ttk.Frame(sources_frame)
grid_option_frame.pack(fill="x", side="top", pady=(0, 4))

display_camera_label = ttk.Label(grid_option_frame, text="Choose Display Grid:", font=('Arial', 10), anchor="center")
display_camera_label.pack(fill='x', side="left")

display_camera_radio1 = ttk.Radiobutton(grid_option_frame, variable=grid_mode, text="1", value=1, command=change_grid)
display_camera_radio1.pack(side="left")

display_camera_radio4 = ttk.Radiobutton(grid_option_frame, variable=grid_mode, text="4", value=4, command=change_grid)
display_camera_radio4.pack(side="left")

sources_list_frame = ttk.Frame(sources_frame)
sources_list_frame.pack(fill="both", expand=True)

# content frame
detect_frame = ttk.Frame(main_frame, borderwidth=10, relief=tk.GROOVE)
detect_frame.pack(fill="both", expand=True)

detect_title = ttk.Label(detect_frame, text="Detection", font=('Arial', 12, 'bold'), anchor='center')
detect_title.pack(fill='x', pady=(0, 4))

combobox_model_frame = ttk.Frame(detect_frame)
combobox_model_frame.pack(fill="x", side='top', pady=(0,4))

combobox_model_label = ttk.Label(combobox_model_frame, text="Model:", font=('Arial', 10), anchor="center")
combobox_model_label.pack(fill='x', side="left")

model_options_var = StringVar()
# Daftar pilihan model
model_choices = [
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"
]
combobox_model_input = ttk.Combobox(combobox_model_frame, textvariable=model_options_var, values=model_choices, state="readonly")
combobox_model_input.set("Choose the model")
combobox_model_input.pack(fill='x', side='left')

camera_frame = ttk.Frame(detect_frame, relief=tk.GROOVE)
camera_frame.pack(fill="both", expand=True, pady=(0, 4))

camera_controller_frame = ttk.Frame(detect_frame)
camera_controller_frame.pack(fill="x")

start_camera_button = ttk.Button(camera_controller_frame, text="Start", command=toggle_streaming)
start_camera_button.pack(fill="x", expand=True)

# configuration frame
yolo_settings_frame = ttk.Frame(menu_frame, borderwidth=10, relief=tk.GROOVE)
yolo_settings_frame.place(x=0, rely=0.33, relwidth=1, relheight=0.33)

yolo_settings_title = ttk.Label(yolo_settings_frame, text="Parameter", font=("Arial", 12, "bold"), anchor="center")
yolo_settings_title.pack(fill="x", pady=(0, 4))


# Variabel untuk menampung nilai slider
conf_var = tk.DoubleVar(value=0.5)
iou_var = tk.DoubleVar(value=0.7)

# confidence threshold
conf_scale_frame = ttk.Frame(yolo_settings_frame)
conf_scale_frame.pack(fill="x", pady=(0, 4))

conf_label = ttk.Label(conf_scale_frame, text="Conf:")
conf_label.pack(side="left")

conf_value_label = ttk.Label(conf_scale_frame, text=f"{conf_var.get():.2f}")
conf_value_label.pack(side="right")

def update_conf(val):
    conf_value_label.config(text=f"{float(val):.2f}")

conf_scale = ttk.Scale(yolo_settings_frame, from_=0.0, to=1.0, variable=conf_var, command=update_conf)
conf_scale.pack(fill="x", pady=(0, 4))

# iou threshold
iou_scale_frame = ttk.Frame(yolo_settings_frame)
iou_scale_frame.pack(fill="x", pady=(0, 4))

iou_label = ttk.Label(iou_scale_frame, text="IoU:")
iou_label.pack(side="left")

iou_value_label = ttk.Label(iou_scale_frame, text=f"{iou_var.get():.2f}")
iou_value_label.pack(side="right")

def update_iou(val):
    iou_value_label.config(text=f"{float(val):.2f}")

iou_scale = ttk.Scale(yolo_settings_frame, from_=0.0, to=1.0, variable=iou_var, command=update_iou)
iou_scale.pack(fill="x", pady=(0, 4))

# status frame
status_frame = ttk.Frame(menu_frame, borderwidth=10, relief=tk.GROOVE)
status_frame.place(x=0, rely=0.66, relwidth=1, relheight=0.34)

status_title = ttk.Label(status_frame, text="Status", font=("Arial", 12, "bold"), anchor="center")
status_title.pack(fill="x", pady=(0, 4))

alert_frame = tk.Frame(status_frame)
alert_frame.pack(fill="x")


available_cameras = detect_available_cameras()
change_grid()
build_frames

window.mainloop()