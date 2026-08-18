from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import config
import time
import torch

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


charclassnames = ['0','9','b','d','ein','ein','g','gh','h','n','s','1','malul','n','s','sad','t','ta','v','y','2'
                  ,'3','4','5','6','7','8']


def _display_detected_frames(conf, model_object, model_char, st_count, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Predict the objects in the image using YOLOv8 model
    res_object = model_object.predict(image, conf=conf)
    

    char_result = "No plate detected"
    
    for i in res_object:
        bbox = i.boxes
        for box in bbox:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_names = int(box.cls[0])

            #check plate to recognize characters with yolov8n model
            if cls_names == 1:
                char_display = []
                #crop plate from frame
                plate_img = image[y1:y2, x1:x2]
                #detect characters of plate with yolov8n model
                plate_output = model_char(plate_img, conf=0.4)
                
                #extract bounding box and class names
                bbox = plate_output[0].boxes.xyxy
                cls = plate_output[0].boxes.cls
                #make a dict and sort it from left to right
                keys = cls.numpy().astype(int)
                values = bbox[:, 0].numpy().astype(int)
                dictionary = list(zip(keys, values))
                sorted_list = sorted(dictionary, key=lambda x: x[1])
                #convert all characters to a string
                for i in sorted_list:
                    char_class = i[0]
                    char_display.append(charclassnames[char_class])
                
                if len(char_display) == 8:
                    char_result = 'Plate: ' + (''.join(char_display))
                else:
                    char_result = "Plate: incomplete detection"
    
    res_plotted = res_object[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_container_width=True
                   )
    
    text_placeholder = st.empty()
    text_placeholder.text(char_result)


@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model_object, model_char):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_container_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res_object = model_object.predict(uploaded_image,
                                    conf=conf)
                boxes = res_object[0].boxes
                    #extract bounding box and class names
                res_plotted = res_object[0].plot()[:, :, ::-1]    
                
                
                char_result = "No plate detected"
                
                for i in res_object:
                    bbox = i.boxes
                    for box in bbox:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cls_names = int(box.cls[0])

                        #check plate to recognize characters with yolov8n model
                        if cls_names == 1:
                            char_display = []
                            #crop plate from frame
                            plate_img = uploaded_image.crop((x1, y1, x2, y2))
                            #detect characters of plate with yolov8n model
                            plate_output = model_char(plate_img, conf=0.4)
                            
                            #extract bounding box and class names
                            bbox = plate_output[0].boxes.xyxy
                            cls = plate_output[0].boxes.cls
                            #make a dict and sort it from left to right
                            keys = cls.numpy().astype(int)
                            values = bbox[:, 0].numpy().astype(int)
                            dictionary = list(zip(keys, values))
                            sorted_list = sorted(dictionary, key=lambda x: x[1])
                            #convert all characters to a string
                            for i in sorted_list:
                                char_class = i[0]
                                char_display.append(charclassnames[char_class])
                            
                            if len(char_display) == 8:
                                char_result = 'Plate: ' + (''.join(char_display))
                            else:
                                char_result = "Plate: incomplete detection"

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_container_width=True)
                    st.write(char_result)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model_object, model_char):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.markdown(
            f'<style>video {{ width: {640}px !important; height: auto !important; }}</style>',
            unsafe_allow_html=True
        )
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(tfile.name)
                    
                   
                    total_cars = 0
                    total_plates = 0
                    detected_plates = {}  
                    unique_plates = [] 
                    frame_count = 0
                    
                    st_frame = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                  
                    plates_list_placeholder = st.empty()
                    detected_plates_display = []  
                    
                  
                    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            frame_count += 1
                            
                           
                            progress = frame_count / total_frames if total_frames > 0 else 0
                            progress_bar.progress(min(progress, 1.0))
                            status_text.text(f"Processing frame {frame_count}/{total_frames}")
                            
                            
                            res_object = model_object.predict(image, conf=conf)
                            
                           
                            for i in res_object:
                                bbox = i.boxes
                                for box in bbox:
                                    cls_names = int(box.cls[0])
                                    if cls_names == 0:  # car
                                        total_cars += 1
                                    elif cls_names == 1:  # plate
                                        total_plates += 1
                                        
                                       
                                        x1, y1, x2, y2 = box.xyxy[0]
                                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                        plate_img = image[y1:y2, x1:x2]
                                        plate_output = model_char(plate_img, conf=0.4)
                                        
                                        bbox_char = plate_output[0].boxes.xyxy
                                        cls_char = plate_output[0].boxes.cls
                                        
                                        if len(cls_char) > 0:
                                            keys = cls_char.numpy().astype(int)
                                            values = bbox_char[:, 0].numpy().astype(int)
                                            dictionary = list(zip(keys, values))
                                            sorted_list = sorted(dictionary, key=lambda x: x[1])
                                            
                                            char_display = []
                                            for item in sorted_list:
                                                char_class = item[0]
                                                char_display.append(charclassnames[char_class])
                                            
                                            if len(char_display) == 8:
                                                plate_text = ''.join(char_display)
                                                
                                                
                                                if plate_text not in detected_plates:
                                                    detected_plates[plate_text] = {
                                                        'first_seen': frame_count,
                                                        'timestamp': time.time()
                                                    }
                                                    unique_plates.append(plate_text)
                                                    detected_plates_display.append(plate_text)
                                                    
                                                  
                                                    if detected_plates_display:
                                                        plates_html = "### 🚗 Detected Plates:\n\n"
                                                        for idx, plate in enumerate(detected_plates_display, 1):
                                                            plates_html += f"{idx}. `{plate}`\n\n"
                                                        plates_list_placeholder.markdown(plates_html)
                           
                            res_plotted = res_object[0].plot()
                            st_frame.image(res_plotted,
                                           caption=f'Processing Frame {frame_count}',
                                           channels="BGR",
                                           use_container_width=True)
                            
                        else:
                            vid_cap.release()
                            break
                    
                  
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("✅ Video processing completed!")
                    
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model_object, model_char):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_count = st.empty()
        st_frame = st.empty()
        
       
        total_cars = 0
        total_plates = 0
        detected_plates = set()
        detected_plates_display = []
        plates_list_placeholder = st.empty()
        
        while not flag:
            success, image = vid_cap.read()
            if success:
                res_object = model_object.predict(image, conf=conf)
                
               
                for i in res_object:
                    bbox = i.boxes
                    for box in bbox:
                        cls_names = int(box.cls[0])
                        if cls_names == 0:
                            total_cars += 1
                        elif cls_names == 1:
                            total_plates += 1
                            
                            
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            plate_img = image[y1:y2, x1:x2]
                            plate_output = model_char(plate_img, conf=0.4)
                            
                            bbox_char = plate_output[0].boxes.xyxy
                            cls_char = plate_output[0].boxes.cls
                            
                            if len(cls_char) > 0:
                                keys = cls_char.numpy().astype(int)
                                values = bbox_char[:, 0].numpy().astype(int)
                                dictionary = list(zip(keys, values))
                                sorted_list = sorted(dictionary, key=lambda x: x[1])
                                
                                char_display = []
                                for item in sorted_list:
                                    char_class = item[0]
                                    char_display.append(charclassnames[char_class])
                                
                                if len(char_display) == 8:
                                    plate_text = ''.join(char_display)
                                    if plate_text not in detected_plates:
                                        detected_plates.add(plate_text)
                                        detected_plates_display.append(plate_text)
                                        
                                        # به‌روزرسانی نمایش لیست پلاک‌ها
                                        if detected_plates_display:
                                            plates_html = "###  Detected Plates:\n\n"
                                            for idx, plate in enumerate(detected_plates_display, 1):
                                                plates_html += f"{idx}. `{plate}`\n\n"
                                            plates_list_placeholder.markdown(plates_html)
                
                res_plotted = res_object[0].plot()
                st_frame.image(res_plotted,
                               caption='Webcam Feed',
                               channels="BGR",
                               use_container_width=True)
                
              
                st_count.write(f" Cars: {total_cars} |  Plates: {total_plates} |  Unique: {len(detected_plates)}")
                
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
