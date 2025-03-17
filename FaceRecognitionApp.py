#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install git+https://github.com/rcmalli/keras-vggface


# In[2]:


#pip install keras_applications


# In[3]:


#pip install mtcnn


# In[16]:


#pip install rawpy pillow


# In[17]:


import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import time
# for loading/processing the images
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model
import keras
import keras_vggface
from keras_vggface.vggface import VGGFace
import mtcnn
from keras.utils.data_utils import get_file
import keras_vggface.utils
import tensorflow as tf
import PIL
import rawpy
# from google.colab.patches import cv2_imshow
import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
import math

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
from scipy.spatial.distance import euclidean

# for everything else
import os
import os.path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
from sklearn.metrics.pairwise import euclidean_distances

# def convert_arw_to_jpg(path):
    
#     arw_files = [file for file in os.listdir(path) if file.lower().endswith('.arw')]
#     # List all ARW files in the input directory
    
#     jpg_files_name = [file.split('.')[0] for file in os.listdir(path) if file.lower().endswith('.jpg')]
    
#     # Convert ARW files to JPG
#     for arw_file in arw_files:
#         if arw_file.split('.')[0] not in jpg_files_name:
#             arw_path = os.path.join(path, arw_file)
#             output_path = os.path.join(path, os.path.splitext(arw_file)[0] + '.jpg')
            
#             # Read the ARW file using rawpy
#             with rawpy.imread(arw_path) as raw:
#                 # Process the raw data
#                 rgb = raw.postprocess()
                
#                 # Save the processed image as JPG using PIL
#                 img = Image.fromarray(rgb)
#                 img.save(output_path)
#             print(f'Converted {arw_file} to {output_path}')
#         else:
#             print('present')
#     print('Conversion complete.')
    

def get_images(path):
    os.chdir(path)
    face_img = []

    [file for file in os.listdir(path) if file.lower().endswith('.arw')]

    # creates a ScandirIterator aliased as files
    with os.scandir(path) as files:
        # loops through each file in the directory
        for file in files:
            if file.name.lower().endswith('.jpg'):
                face_img.append(file.name)
    return face_img

def get_videos(path):
    os.chdir(path)
    face_vid = []
    with os.scandir(path) as files:
        for file in files:
            if file.name.lower().endswith('.mp4'):
                face_vid.append(file.name)
    return face_vid
    
def get_mtcnn(img):
  photo = plt.imread(img)
  face_detector = mtcnn.MTCNN()
  face_roi = face_detector.detect_faces(photo)
  x1, y1, width, height = face_roi[0]['box']
  # width, height = width+100 , height+150
  x2, y2 = x1 + width, y1+height
  face = photo[y1:y2, x1:x2]
  # print(face)
  return face

def extract_features(img, model):
  # print(img.shape)
  img = Image.fromarray(img)
  img = img.resize((224, 224),Image.Resampling.LANCZOS)
  resized_image_array = np.array(img)
  # print("resized image: ",resized_image_array.shape)
  reshaped_img = resized_image_array.reshape(1,224, 224,3)
  # print("model reshape image : ", reshaped_img.shape)
  imgx = preprocess_input(reshaped_img)
  # print(imgx.shape)
  features = model.predict(imgx, use_multiprocessing=True)
  # print("features: ",features)

  return features

def get_top_frames_faces(path,video):
  cap = cv2.VideoCapture(path+'/'+video)
  time.sleep(0.05)

  if not cap.isOpened():
    time.sleep(0.05)
    print("Error: Could not open video file.")
    exit()

  frames = []
  frames_with_faces = []
  frame_rate = int((int(cap.get(cv2.CAP_PROP_FPS)))/2)
  elapsed_time = 0
  faces_confidence = []
  face_detector = mtcnn.MTCNN()

  while True:
    ret, frame = cap.read()
    time.sleep(0.05)

    if not ret:
        time.sleep(0.05)
        break
    elapsed_time += 1

    if elapsed_time >= frame_rate:
      face_roi = face_detector.detect_faces(frame)
      time.sleep(0.05)

      if face_roi!=[]:
        time.sleep(0.05)
        print(frame.shape)
        x1, y1, width, height = face_roi[0]['box']
        x2, y2 = x1 + width, y1+height
        face = frame[y1:y2, x1:x2]
        print(face.shape)
        frames_with_faces.append(face)
        frames.append(frame)
        faces_confidence.append(face_roi[0]['confidence'])
        time.sleep(0.05)
      elapsed_time = 0
      print(len(frames))
      time.sleep(0.05)

    # frames.append(frame)
    if len(frames_with_faces) ==2:
      time.sleep(0.05)
      break
  cap.release()
  # top_ids = np.argsort(faces_confidence)[::-1][:2]
  # top_frames = [frames[i] for i in top_ids]
  # top_faces = [frames_with_faces[i] for i in top_ids]
  top_frame = []
  top_face = []
  time.sleep(0.05)
  if len(faces_confidence)>0:
        top_face_id = np.argmax(faces_confidence)
        print(top_face_id)
        top_frame = frames[top_face_id]
        top_face = frames_with_faces[top_face_id]
  return top_frame, top_face




# Function to handle image processing (simulated with a sleep)
def process_images():
    image_path = entry_image_path.get()
    num_people = entry_num_people.get()
    final_path = entry_final_path.get()
    num_people = int(num_people)

    if not final_path:
        messagebox.showerror("Error", "Please Select the destination folder")
        return

    if not image_path:
        messagebox.showerror("Error", "Please Select the Image folder")
        return

    if not num_people:
        messagebox.showerror("Error", "Please enter the Total Number of People.")
        return
    progress_var.set(1)
    progress_label.config(text=f"Loading Data from the path: 1%")
    root.update_idletasks()
    time.sleep(0.1)
    
    
    arw_files = [file for file in os.listdir(image_path) if file.lower().endswith('.arw')]
    # List all ARW files in the input directory
    
    jpg_files_name = [file.split('.')[0] for file in os.listdir(image_path) if file.lower().endswith('.jpg')]
    
    aa = 9/len(arw_files)
    lp = 1
    # Convert ARW files to JPG
    for arw_file in arw_files:
        lp=lp+aa
        lp = round(lp, 2)
        progress_var.set(lp)
        progress_label.config(text=f"Loading Data from the path: {lp}%")
        root.update_idletasks()
        time.sleep(0.2)
        if arw_file.split('.')[0] not in jpg_files_name:
            arw_path = os.path.join(image_path, arw_file)
            output_path = os.path.join(image_path, os.path.splitext(arw_file)[0] + '.jpg')
            
            # Read the ARW file using rawpy
            with rawpy.imread(arw_path) as raw:
                # Process the raw data
                rgb = raw.postprocess()
                
                # Save the processed image as JPG using PIL
                img = Image.fromarray(rgb)
                img.save(output_path)
            print(f'Converted {arw_file} to {output_path}')
        else:
            print('present')
    print('Conversion complete.')
#     convert_arw_to_jpg(image_path)
    
    
    face_img = get_images(image_path)
    
    
    progress_var.set(10)
    progress_label.config(text=f"Detecting Faces: 10%")
    root.update_idletasks()
    time.sleep(0.1)
    
    data = {}
    aa = 40/len(face_img)
    lp = 10
    for face in face_img:
        # print(face)
        try:
            lp=lp+aa
            lp = round(lp, 2)
            progress_var.set(lp)
            progress_label.config(text=f"Detecting Face:{face} ({lp}%)")
            root.update_idletasks()
            time.sleep(0.1)
            feat = get_mtcnn(face)
            data[face] = feat
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    
    progress_var.set(50)
    progress_label.config(text=f"Loading Feature Extraction Models: 50%")
    root.update_idletasks()
    time.sleep(0.05)
    filenames = np.array(list(data.keys()))
    vggface_resnet = VGGFace(model='resnet50')
    vggface_resnet = Model(inputs=vggface_resnet.inputs, outputs=vggface_resnet.layers[-2].output)
    
    progress_var.set(55)
    progress_label.config(text=f"Extracting Features from faces: 55%")
    root.update_idletasks()
    time.sleep(0.05)
    resnet_feat_data = {}
    # loop through each image in the dataset
    aa = 15/len(data)
    lp = 55
    for face in data:
        try:
            lp=lp+aa
            lp = round(lp, 2)
            print(lp)
            progress_var.set(lp)
            progress_label.config(text=f"Extracting Features from face:{face} ({lp}%)")
            root.update_idletasks()
            time.sleep(0.05)
            feat = extract_features(data[face],vggface_resnet)
            resnet_feat_data[face] = feat
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    filenames = np.array(list(resnet_feat_data.keys()))
    
    progress_var.set(70)
    progress_label.config(text=f"preparing features for clustering: 70%")
    root.update_idletasks()
    time.sleep(0.05)
    feat = np.array(list(resnet_feat_data.values()))
    feat = feat.reshape(-1,feat.shape[2])
    progress_var.set(72)
    progress_label.config(text=f"clustering faces based on features: 72%")
    root.update_idletasks()
    time.sleep(0.05)
    x = feat
    kmeans = KMeans(n_clusters=num_people, random_state=22)
    kmeans.fit(x)
    progress_var.set(75)
    progress_label.config(text=f"clustering faces based on features: 75%")
    root.update_idletasks()
    time.sleep(0.05)
    groups = {}
    for file, cluster in zip(filenames,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    progress_var.set(80)
    progress_label.config(text=f"clustering faces based on features: 80%")
    root.update_idletasks()
    time.sleep(0.05)
            
    cluster_centroids = kmeans.cluster_centers_
    distance_threshold = 80
#     merged_clusters = []
    test = []
    remaining_clusters = list(range(len(cluster_centroids)))
    
    progress_var.set(85)
    progress_label.config(text=f"clustering faces based on features: 85%")
    root.update_idletasks()
    time.sleep(0.05)
    while remaining_clusters:
        current_cluster = remaining_clusters[0]
        clusters_to_merge = [current_cluster]
        
        for i in range(1, len(remaining_clusters)):
            candidate_cluster = remaining_clusters[i]
            
            distance = euclidean(cluster_centroids[current_cluster], cluster_centroids[candidate_cluster])
            
            if distance < distance_threshold:
                clusters_to_merge.append(candidate_cluster)
        
        if len(clusters_to_merge)>0:
            merged_cluster = [k for c in clusters_to_merge for k in groups[c]]
        else:
            merged_cluster = groups[current_cluster]
        
        remaining_clusters = [c for c in remaining_clusters if c not in clusters_to_merge]
        test.append(merged_cluster)
        
    progress_var.set(90)
    progress_label.config(text=f"Creating group folders: 90%")
    root.update_idletasks()
    time.sleep(0.05)
    source_folder = image_path
    output_path = final_path
    dest_folders = [output_path+'/'+str(i) for i in range(0,len(test))]
    for grp_folder in dest_folders:
        if not os.path.exists(grp_folder):
            os.makedirs(grp_folder)
    progress_var.set(95)
    progress_label.config(text=f"Processing group folders: 95%")
    root.update_idletasks()
    time.sleep(0.05)
    for i, grp_img in enumerate(test):
        for img in grp_img:
            source_path = os.path.join(source_folder, img)
            destination_path = os.path.join(dest_folders[i], img)
            shutil.copy(source_path, destination_path)
    progress_var.set(100)
    progress_label.config(text=f"Processing Completed: 100%")
    root.update_idletasks()
    time.sleep(0.05)

#     for i in range(101):
#         progress_var.set(i)
#         progress_label.config(text=f"Processing: {i}%")
#         root.update_idletasks()
#         time.sleep(0.05)

    messagebox.showinfo("Image Processing", f"Processing images in folder: {image_path} for {num_people} people completed")
    # Reset input fields
    entry_image_path.delete(0, tk.END)
    entry_num_people.delete(0, tk.END)
    entry_final_path.delete(0,tk.END)
    entry_video_path.delete(0,tk.END)

    # Reset the progress bar
    progress_var.set(0)
    # Reset the progress label
    progress_label.config(text="")


# Function to handle video processing (simulated with a sleep)
def process_video():
    video_path = entry_video_path.get()
    num_people = entry_num_people.get()
    final_path = entry_final_path.get()
    num_people = int(num_people)

    if not final_path:
        messagebox.showerror("Error", "Please Select the destination folder")
        return
    if not video_path:
        messagebox.showerror("Error", "Please Select the Video folder")
        return
    if not num_people:
        messagebox.showerror("Error", "Please enter the Total Number of People.")
        return
    
    progress_var.set(1)
    progress_label.config(text=f"Loading Data from the path: 1%")
    root.update_idletasks()
    time.sleep(0.1)
    face_vid = get_videos(video_path)
    
    frame_data = {}
    face_data = {}
    progress_var.set(5)
    progress_label.config(text=f"Detecting Faces: 5%")
    root.update_idletasks()
    time.sleep(0.2)
    aa = 45/len(face_vid)
    lp = 0
    for face in face_vid:
        # print(face)
        try:
            lp=lp+aa
            lp = round(lp, 2)
            progress_var.set(lp)
            progress_label.config(text=f"Detecting Face:{face} ({lp}%)")
            root.update_idletasks()
            time.sleep(0.1)
            frame, feat = get_top_frames_faces(video_path,face)
            frame_data[face] = frame
            face_data[face] = feat
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    filenames = np.array(list(frame_data.keys()))
    
    progress_var.set(50)
    progress_label.config(text=f"Loading Feature Extraction Models: 50%")
    root.update_idletasks()
    time.sleep(0.05)
    vggface_resnet = VGGFace(model='resnet50')
    vggface_resnet = Model(inputs=vggface_resnet.inputs, outputs=vggface_resnet.layers[-2].output)
    
    progress_var.set(55)
    progress_label.config(text=f"Extracting Features from faces: 55%")
    root.update_idletasks()
    time.sleep(0.05)
    resnet_feat_data = {}
    # loop through each image in the dataset
    aa = 15/len(face_data)
    lp = 55
    for face in face_data:
        try:
            lp=lp+aa
            lp = round(lp, 2)
            progress_var.set(lp)
            progress_label.config(text=f"Extracting Features from face:{face} ({lp}%)")
            root.update_idletasks()
            time.sleep(0.05)
#             print(face_data[face])
            if len(face_data[face])>0:
                feat = extract_features(face_data[face],vggface_resnet)
                resnet_feat_data[face] = feat
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    filenames = np.array(list(resnet_feat_data.keys()))
    
    progress_var.set(70)
    progress_label.config(text=f"preparing features for clustering: 70%")
    root.update_idletasks()
    time.sleep(0.05)
    feat = np.array(list(resnet_feat_data.values()))
    feat = feat.reshape(-1,feat.shape[2])
    progress_var.set(72)
    progress_label.config(text=f"clustering faces based on features: 72%")
    root.update_idletasks()
    time.sleep(0.05)
    x = feat
    kmeans = KMeans(n_clusters=num_people, random_state=22)
    kmeans.fit(x)
    progress_var.set(75)
    progress_label.config(text=f"clustering faces based on features: 75%")
    root.update_idletasks()
    time.sleep(0.05)
    groups = {}
    for file, cluster in zip(filenames,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    progress_var.set(80)
    progress_label.config(text=f"clustering faces based on features: 80%")
    root.update_idletasks()
    time.sleep(0.05)
            
    cluster_centroids = kmeans.cluster_centers_
    distance_threshold = 10
#     merged_clusters = []
    test = []
    remaining_clusters = list(range(len(cluster_centroids)))
    
    progress_var.set(85)
    progress_label.config(text=f"clustering faces based on features: 85%")
    root.update_idletasks()
    time.sleep(0.05)
    while remaining_clusters:
        current_cluster = remaining_clusters[0]
        clusters_to_merge = [current_cluster]
        
        for i in range(1, len(remaining_clusters)):
            candidate_cluster = remaining_clusters[i]
            
            distance = euclidean(cluster_centroids[current_cluster], cluster_centroids[candidate_cluster])
            
            if distance < distance_threshold:
                clusters_to_merge.append(candidate_cluster)
        
        if len(clusters_to_merge)>0:
            merged_cluster = [k for c in clusters_to_merge for k in groups[c]]
        else:
            merged_cluster = groups[current_cluster]
        
        remaining_clusters = [c for c in remaining_clusters if c not in clusters_to_merge]
        test.append(merged_cluster)
        
    progress_var.set(90)
    progress_label.config(text=f"Creating group folders: 90%")
    root.update_idletasks()
    time.sleep(0.05)
    source_folder = video_path
    output_path = final_path
    dest_folders = [output_path+'/'+str(i) for i in range(0,len(test))]
    for grp_folder in dest_folders:
        if not os.path.exists(grp_folder):
            os.makedirs(grp_folder)
    progress_var.set(95)
    progress_label.config(text=f"Processing group folders: 95%")
    root.update_idletasks()
    time.sleep(0.05)
    for i, grp_img in enumerate(test):
        for img in grp_img:
            source_path = os.path.join(source_folder, img)
            destination_path = os.path.join(dest_folders[i], img)
            shutil.copy(source_path, destination_path)
#     grppp = [i for grp in test for i in grp]
#     for noface in face_vid:
#         for grp_img in grppp:
#             if noface not in grp_img:
#                 source_path = os.path.join(source_folder, noface)
#                 destination_path = os.path.join(output_path+'/'+str(0), noface)
#                 shutil.copy(source_path, destination_path)
    progress_var.set(100)
    progress_label.config(text=f"Processing Completed: 100%")
    root.update_idletasks()
    time.sleep(0.05)

#     for i in range(101):
#         progress_var.set(i)
#         progress_label.config(text=f"Processing: {i}%")
#         root.update_idletasks()
#         time.sleep(0.05)

    messagebox.showinfo("Video Processing", f"Processing videos in folder: {video_path} for {num_people} people completed")
    # Reset input fields
    entry_image_path.delete(0, tk.END)
    entry_num_people.delete(0, tk.END)
    entry_final_path.delete(0, tk.END)
    entry_video_path.delete(0, tk.END)

    # Reset the progress bar
    progress_var.set(0)
    # Reset the progress label
    progress_label.config(text="")

# Function to open a folder dialog and update the entry field
def browse_for_folder(entry_field):
    folder_path = filedialog.askdirectory()
    entry_field.delete(0, tk.END)
    entry_field.insert(0, folder_path)

# Create the main application window
root = tk.Tk()
root.title("Face Recognition App")

window_width = 600
window_height = 240
root.geometry(f"{window_width}x{window_height}")
root.resizable(False, False)


# Define dark mode colors
dark_color = "#1E5128"
bg_input_color = "#171C17"
dark_btn_color = "#4E9F3D"
dark_background_color = "#191A19"
dark_text_color = "white"

# Set the background color for the dark mode theme
root.configure(bg=dark_background_color)

# Create labels, input fields, and buttons with dark mode theme
label_image_path = tk.Label(root, text="Select Image Folder:", bg=dark_background_color, fg=dark_text_color)
entry_image_path = tk.Entry(root, width=40, bg=bg_input_color, fg=dark_text_color)
button_browse_image = tk.Button(root, text="Browse", command=lambda: browse_for_folder(entry_image_path) , bg=dark_color,fg=dark_text_color)
button_image = tk.Button(root, text="Process Images", command=process_images, bg=dark_btn_color,fg=dark_text_color)

label_video_path = tk.Label(root, text="Select Video Folder:", bg=dark_background_color, fg=dark_text_color)
entry_video_path = tk.Entry(root, width=40, bg=bg_input_color,fg=dark_text_color)
button_browse_video = tk.Button(root, text="Browse", command=lambda: browse_for_folder(entry_video_path), bg=dark_color,fg=dark_text_color)
button_video = tk.Button(root, text="Process Videos", command=process_video, bg=dark_btn_color,fg=dark_text_color)

label_num_people = tk.Label(root, text="Total Number of People:", bg=dark_background_color, fg=dark_text_color)
entry_num_people = tk.Entry(root, width=10, bg=bg_input_color,fg=dark_text_color)


label_final_path = tk.Label(root, text="Select Output Folder:", bg=dark_background_color, fg=dark_text_color)
entry_final_path = tk.Entry(root, width=40, bg=bg_input_color, fg=dark_text_color)
button_final_image = tk.Button(root, text="Browse", command=lambda: browse_for_folder(entry_final_path) , bg=dark_color,fg=dark_text_color)


# Create a custom style for the progress bar with a background color
style = ttk.Style()
style.configure("Custom.Horizontal.TProgressbar", troughcolor=bg_input_color)

# Create a progress bar with the custom style
progress_var = tk.DoubleVar()
progress = ttk.Progressbar(root, variable=progress_var, maximum=100, style="Custom.Horizontal.TProgressbar")

# Create a label to display progress text
progress_label = tk.Label(root, text="", bg=dark_background_color, fg=dark_text_color)


# Arrange widgets using the grid layout manager
label_image_path.grid(row=0, column=0, padx=10, pady=15, sticky="e")
entry_image_path.grid(row=0, column=1, padx=10, pady=5)
button_browse_image.grid(row=0, column=2, padx=10, pady=5)
button_image.grid(row=0, column=3, padx=10, pady=5)

label_video_path.grid(row=1, column=0, padx=10, pady=5, sticky="e")
entry_video_path.grid(row=1, column=1, padx=10, pady=5)
button_browse_video.grid(row=1, column=2, padx=10, pady=5)
button_video.grid(row=1, column=3, padx=10, pady=5)

label_num_people.grid(row=2, column=0, padx=10, pady=5, sticky="e")
entry_num_people.grid(row=2, column=1, padx=10, pady=5)

label_final_path.grid(row=3, column=0, padx=10, pady=15, sticky="e")
entry_final_path.grid(row=3, column=1, padx=10, pady=5)
button_final_image.grid(row=3, column=2, padx=10, pady=5)

progress.grid(row=5, column=0, columnspan=4, padx=10, pady=5, sticky="we")
progress_label.grid(row=4, column=0, columnspan=4, padx=10, pady=5, sticky="we")

# Start the main event loop
root.mainloop()

