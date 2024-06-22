import firebase_admin
from firebase_admin import credentials , messaging


# Initialize Firebase Admin SDK with credentials
cred = credentials.Certificate('safety-gear-detection-system-firebase-adminsdk-l9t7b-9798738456.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'safety-gear-detection-system.appspot.com'
})







import pyrebase

# Configure Firebase project settings
firebaseConfig = {
    
  'apiKey': "AIzaSyDvmbyb_1nzs3lEeJzOclQclwT1Jo-ywaQ",
  'authDomain': "safety-gear-detection-system.firebaseapp.com",
  'projectId': "safety-gear-detection-system",
  'storageBucket': "safety-gear-detection-system.appspot.com",
  'messagingSenderId': "738051321837",
  'appId': "1:738051321837:web:9b9e71b5ff9d8aeb91c1e4",
  'measurementId': "G-167BZWLW9P",
  'databaseURL' : 'https://safety-gear-detection-system-default-rtdb.firebaseio.com/'
  
  }

# Initialize Pyrebase
firebase = pyrebase.initialize_app(firebaseConfig)

auth = firebase.auth()






from firebase_admin import firestore

# # Initialize Firestore client
db = firestore.client()







import tkinter as tk
from PIL import ImageTk, Image
from firebase_admin import firestore
from tkinter import messagebox
import datetime





# Initialize Firestore client
db = firestore.client()

# Declare uid and rtsp_link as global variables
uid = None
rtsp_link = None
is_detection_enabled = None
device_token = None






def login():
    global uid  # Access the global uid variable
    email = entry_email.get()
    password = entry_password.get()

    try:
        user = auth.sign_in_with_email_and_password(email=email, password=password)
        id_token = user['idToken']
        uid = user['localId']  # Assign the value to the global uid variable
        messagebox.showinfo("Login Successful", "Welcome!")
        return id_token, uid, email
    except:
        messagebox.showerror("Login Failed", "Invalid email or password")
        return None, None, None






def retrieve_user_data(uid):
    # Function to retrieve user data from Firestore
    # You can perform actions based on the UID here
    print("Retrieving user data for UID:", uid)
    doc_ref = db.collection("Users").document(uid)
    doc = doc_ref.get()
    if doc.exists:
        user_data = doc.to_dict()
        rtsp_link = user_data.get('RTSP', '')
        is_detection_enabled = user_data.get('isDetectionEnabled', False)
        device_token = user_data.get('deviceToken', '')  # Fetch deviceToken value
        return rtsp_link, is_detection_enabled, device_token
    else:
        print("User data not found for UID:", uid)
        return None, False, None








def create_folder_and_add_data(uid, email, login_time, rtsp_link):
    # Define the collection name
    collection_name = "Users"

    # Define the document data
    data = {
        "email": email,
        "uid": uid,
        "login_time": login_time.strftime("%Y-%m-%d %H:%M:%S"),
        "rtsp_link": rtsp_link  # Add the RTSP link field
    }

    try:
        # Add data to the Firestore collection
        doc_ref = db.collection(collection_name).document(uid)
        doc_ref.set(data)
        print("Data added to Firestore successfully!")
    except Exception as e:
        print("Error:", e)








def on_login():
    global rtsp_link, is_detection_enabled, id_token, uid, email, device_token  # Access the global variables
    id_token, uid, email = login()
    if id_token:
        # Call function to retrieve user data
        rtsp_link, is_detection_enabled, device_token = retrieve_user_data(uid)
        if rtsp_link is not None:
            print("RTSP link:", rtsp_link)
        else:
            print("No RTSP link found in user data")
        
        # Do something with the ID token, UID, email, RTSP link, isDetectionEnabled, and deviceToken

        print("ID Token:", id_token)
        print("UID:", uid)
        print("Email:", email)
        print("isDetectionEnabled:", is_detection_enabled)
        print("Device Token:", device_token)

        # Close the window after successful login
        messagebox.showinfo("Login Successful", "Welcome!")
        root.destroy()

        return id_token, uid, email, rtsp_link, is_detection_enabled, device_token






def load_background_image():
    # Load background image and resize if necessary
    image = Image.open("Simple Lined Black Login Page Wireframe Website UI Prototype/1.jpg")
    image = image.resize((600, 400), Image.LANCZOS)  # Use LANCZOS resampling
    return ImageTk.PhotoImage(image)  # Return PhotoImage object

root = tk.Tk()
root.geometry("600x400")  #  geometry for a larger window
root.title("Authentication")

# Load the background image
bg_image = load_background_image()

# Create a label with the background image
background_label = tk.Label(root, image=bg_image)
background_label.image = bg_image  # Keep a reference to avoid garbage collection
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Email entry
entry_email = tk.Entry(root, width=15, font=("Arial", 14), bd=2)  
entry_email.place(x=350, y=150)  # Adjust position

# Password entry
entry_password = tk.Entry(root, show="*", width=15, font=("Arial", 14), bd=2)  
entry_password.place(x=350, y=200)  # Adjust position

# Login button
button_login = tk.Button(root, text="Login", command=on_login)
button_login.place(x=415, y=255)  # Adjust position 

root.mainloop()





print(rtsp_link)

# rtsp_link = "rtsp://admin:admin@192.168.100.5:1935"
# rtsp_link = 'people.mp4'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from openvino.inference_engine import IECore
from firebase_admin import firestore, storage
from datetime import datetime, timedelta
import time

# Initialize Firestore client
bucket = storage.bucket()

# Define global variables
device_token = None
cap = None  # Initialize cap object

# Function to load OpenVINO model
def load_openvino_model(xml_path, bin_path):
    ie = IECore()
    net = ie.read_network(model=xml_path, weights=bin_path)
    exec_net = ie.load_network(network=net, device_name="CPU")
    return exec_net

# Function to calculate Euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Function to perform safety gear detection
def perform_safety_gear_detection(frame, exec_net_safety_gear):
    input_blob = next(iter(exec_net_safety_gear.input_info))
    input_shape = exec_net_safety_gear.input_info[input_blob].input_data.shape
    frame_resized = cv2.resize(frame, (input_shape[3], input_shape[2]))
    person_blob = cv2.dnn.blobFromImage(frame_resized, size=(input_shape[3], input_shape[2]), ddepth=cv2.CV_8U)
    outputs = exec_net_safety_gear.infer(inputs={input_blob: person_blob})

    class_labels = {
        1: 'person',
        2: 'vest',
        3: 'no-helmet',
        4: 'helmet',
        5: 'no-vest'
    }

    safety_gear_detected = []

    for output in outputs.values():
        for detection in output[0][0]:
            confidence = detection[2]

            if confidence > 0.2:
                class_id = int(detection[1])
                class_label = class_labels.get(class_id, "Unknown")

                if class_label in ['helmet', 'vest']:
                    safety_gear_detected.append(class_label)

    if 'helmet' in safety_gear_detected and 'vest' in safety_gear_detected:
        safety_gear_detection_result = "Helmet and Vest"
    elif 'helmet' in safety_gear_detected:
        safety_gear_detection_result = "vest missing"
    elif 'vest' in safety_gear_detected:
        safety_gear_detection_result = "helmet missing"
    else:
        safety_gear_detection_result = "helmet and vest missing"

    return safety_gear_detection_result

# Function to upload image to Firebase Storage and save its link to Firestore
import os

def resize_frame(frame):
    target_width = 544
    target_height = 320
    return cv2.resize(frame, (target_width, target_height))

def upload_image_and_save_to_firestore(image, uid, alert_type):
    # Get current date and time
    current_time = datetime.now()

    # Encode image to JPEG format
    _, img_encoded = cv2.imencode('.jpg', image)

    # Generate file path
    file_name = f"{current_time.strftime('%Y-%m-%d_%H-%M-%S')}_Capture.jpg"
    directory = f"Alerts/{uid}"
    file_path = os.path.join(directory, file_name)

    # Save image locally
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'wb') as f:
        f.write(img_encoded)

    # Upload image to Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(file_path)
    blob.upload_from_filename(file_path)

    # Get the public URL of the uploaded image
    link = blob.generate_signed_url(expiration=timedelta(days=1), version="v4")

    # Save image details to Firestore under 'Alerts' folder
    doc_ref = firestore.client().collection('Alerts').document(uid).collection('Captures').document(file_name)
    doc_ref.set({
        'Name': "Capture",
        'DateTime': current_time,
        'Link': link,
        'alertType': alert_type
    })

    # Return the link of the saved image
    return link

def send_notification(title, body, token):
    if token:
        # Create a message to send
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body
            ),
            token=token
        )

        # Send the message
        response = messaging.send(message)
        print('Successfully sent message:', response)
    else:
        print("Device token is empty. Cannot send notification.")


# Function to display video 
def display_video(frame, persons):
    try:
        frame = resize_frame(frame)  # Resize frame to 320x544

        for bbox, person_info in persons.items():
            (startX, startY, endX, endY) = bbox
            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            text = f"{person_info['class_label']}"
            cv2.putText(frame, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Video", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):  # Check if 'q' key is pressed to exit
            cv2.destroyAllWindows()  # Close all OpenCV windows
            return False  # Return False when 'q' key is pressed
        return True  # Return True otherwise
    except Exception as e:
        print("An error occurred while displaying the video:", e)
        return True  # Return True in case of error



def perform_inference_with_person_detection_video(rtsp_link, person_xml_path, person_bin_path, safety_gear_xml_path, safety_gear_bin_path, uid):
    global cap, is_detection_enabled, device_token
    exec_net_person = load_openvino_model(person_xml_path, person_bin_path)
    exec_net_safety_gear = load_openvino_model(safety_gear_xml_path, safety_gear_bin_path)

    person_class_labels = {1: 'person'}
    detected_persons = {}  # Dictionary to store detected persons along with their IDs
    next_person_id = 0  # Counter to assign IDs to detected persons

    def on_snapshot(doc_snapshot, changes, read_time):
        global rtsp_link, is_detection_enabled, cap, device_token  # Use nonlocal to access and modify outer variables
        for change in changes:
            if change.type.name == 'MODIFIED':
                data = change.document.to_dict()
                detection_enabled = data.get('isDetectionEnabled')
                rtsp = data.get('RTSP')
                device_token_update = data.get('deviceToken')
                if detection_enabled is not None:
                    print("Detection enabled status in document", change.document.id, ":", detection_enabled)
                    is_detection_enabled = detection_enabled

                if device_token_update is not None and device_token_update != device_token:
                    print("Device token updated to:", device_token_update)
                    device_token = device_token_update

                if rtsp is not None:
                    print("RTSP link updated to:", rtsp)
                    rtsp_link = rtsp
                    if cap is not None:  # If cap object already exists, release it
                        cap.release()
                    cap = cv2.VideoCapture(rtsp_link)  # Open new video capture object with updated RTSP link
                    if not cap.isOpened():
                        print("Failed to open RTSP link:", rtsp_link)




    user_doc_ref = db.collection('Users').document(uid)
    user_doc_watch = user_doc_ref.on_snapshot(on_snapshot)

    # Initialize last_rtsp_check_time before the loop
    last_rtsp_check_time = time.time()

    while True:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(rtsp_link)  # Try to initialize it
            if not cap.isOpened():
                print("Failed to open RTSP link. Waiting for update...")
                time.sleep(5)  # Wait for a few seconds before trying again
                continue  # Continue to the next iteration of the loop without checking RTSP updates yet

        # Check for updates to the RTSP link and device token after every 3 seconds
        if time.time() - last_rtsp_check_time >= 3:
            user_doc_ref = db.collection('Users').document(uid)
            doc = user_doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                rtsp_update = data.get('RTSP')
                device_token_update = data.get('deviceToken')
                if rtsp_update is not None and rtsp_update != rtsp_link:
                    print("RTSP link updated to:", rtsp_update)
                    rtsp_link = rtsp_update
                    if cap is not None:  # If cap object already exists, release it
                        cap.release()
                    cap = cv2.VideoCapture(rtsp_link)  # Open new video capture object with updated RTSP link
                    if not cap.isOpened():
                        print("Failed to open RTSP link:", rtsp_link)
                if device_token_update is not None and device_token_update != device_token:
                    print("Device token updated to:", device_token_update)
                    device_token = device_token_update
            last_rtsp_check_time = time.time()  # Update the last check time

        ret, frame = cap.read()
        if not ret:
            break

        if is_detection_enabled:
            # Perform inference
            person_input_blob = next(iter(exec_net_person.input_info))
            person_input_shape = exec_net_person.input_info[person_input_blob].input_data.shape

            if frame.shape[:2] != (person_input_shape[2], person_input_shape[3]):
                frame_resized = cv2.resize(frame, (person_input_shape[3], person_input_shape[2]))
            else:
                frame_resized = frame.copy()

            person_blob = cv2.dnn.blobFromImage(frame_resized, size=(person_input_shape[3], person_input_shape[2]), ddepth=cv2.CV_8U)
            person_outputs = exec_net_person.infer(inputs={person_input_blob: person_blob})

            current_frame_persons = {}  # Dictionary to store detected persons in the current frame

            for person_output in person_outputs.values():
                for person_detection in person_output[0][0]:
                    person_confidence = person_detection[2]

                    if person_confidence > 0.5:
                        person_id = None
                        person_class_id = int(person_detection[1])
                        person_class_label = person_class_labels.get(person_class_id, "Unknown")
                        person_box = person_detection[3:7] * np.array([frame_resized.shape[1], frame_resized.shape[0], frame_resized.shape[1], frame_resized.shape[0]])
                        (person_startX, person_startY, person_endX, person_endY) = person_box.astype("int")

                        current_frame_persons[(person_startX, person_startY, person_endX, person_endY)] = {'class_label': person_class_label}

            for bbox, person_info in current_frame_persons.items():
                min_distance = float('inf')
                closest_person_id = None

                for person_id, person_bbox in detected_persons.items():
                    prev_bbox = person_bbox['bbox']
                    distance = euclidean_distance(bbox[0], bbox[1], prev_bbox[0], prev_bbox[1])

                    if distance < min_distance:
                        min_distance = distance
                        closest_person_id = person_id

                if min_distance < 70:  # Threshold distance for considering the same person
                    detected_persons[closest_person_id]['bbox'] = bbox
                    if 'safety_gear_detected' not in detected_persons[closest_person_id]:
                        safety_gear_result = perform_safety_gear_detection(frame_resized, exec_net_safety_gear)
                        detected_persons[closest_person_id]['safety_gear_detected'] = True
                        if safety_gear_result != "Helmet and Vest":
                            # Capture snapshot and upload image
                            image_url = upload_image_and_save_to_firestore(frame_resized, uid, safety_gear_result)

                            message_title = "Safety Gear Alert"
                            message_body = f"Safety gear detection result: {safety_gear_result}"


                            send_notification(message_title, message_body, device_token)

                            print(f"Person ID: {closest_person_id}, Class: {person_info['class_label']}, Safety Gear Detection Result: {safety_gear_result}")
                else:
                    detected_persons[next_person_id] = {'bbox': bbox}
                    safety_gear_result = perform_safety_gear_detection(frame_resized, exec_net_safety_gear)
                    detected_persons[next_person_id]['safety_gear_detected'] = True
                    if safety_gear_result != "Helmet and Vest":




                        # to upload the alerts and send notification#


                        image_url = upload_image_and_save_to_firestore(frame_resized, uid, safety_gear_result)
                        message_title = "Safety Gear Alert"
                        message_body = f"Safety gear detection result: {safety_gear_result}"
                        
                        send_notification(message_title, message_body, device_token)

                        print(f"Person ID: {next_person_id}, Class: {person_info['class_label']}, Safety Gear Detection Result: {safety_gear_result}")
                    next_person_id += 1

            display_video(frame, current_frame_persons)

            
        else:
            # Display raw video
            cap = cv2.VideoCapture(rtsp_link)
            check_interval = 5  # Check the flag every 5 seconds
            last_check_time = time.time()
            frame_delay = 30  # Delay between frames in milliseconds

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to retrieve frame")
                    break

                cv2.imshow("Video", frame)
                cv2.waitKey(frame_delay)  # Introduce a delay between frames

                # Check if it's time to check the flag
                current_time = time.time()
                if current_time - last_check_time >= check_interval:
                    user_doc_ref = db.collection('Users').document(uid)
                    doc = user_doc_ref.get()
                    if doc.exists:
                        data = doc.to_dict()
                        is_detection_enabled = data.get('isDetectionEnabled', False)
                        device_token_update = data.get('deviceToken')
                        if is_detection_enabled:
                            # Switch to performing inference
                            cap.release()
                            cv2.destroyAllWindows()
                            perform_inference_with_person_detection_video(rtsp_link, person_xml_path, person_bin_path, safety_gear_xml_path, safety_gear_bin_path, uid)
                            break  # Exit the current loop as inference has started
                    last_check_time = current_time

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    person_xml_path = "person_detection_0013_model/person-detection-retail-0013.xml"
    person_bin_path = "person_detection_0013_model/person-detection-retail-0013.bin"
    safety_gear_xml_path = "worker-safety-mobilenet/worker_safety_mobilenet_ir.xml"
    safety_gear_bin_path = "worker-safety-mobilenet/worker_safety_mobilenet_ir.bin"

    perform_inference_with_person_detection_video(rtsp_link, person_xml_path, person_bin_path, safety_gear_xml_path, safety_gear_bin_path, uid)

if cap is not None:
    cap.release()
cv2.destroyAllWindows()

