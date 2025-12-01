import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import requests
import os
import zipfile
from deepface import DeepFace
import tempfile # Thư viện mới để tạo file tạm duy nhất
import time 

# --- 1. Thiết lập trang Streamlit ---
st.set_page_config(
    page_title="Hệ thống Điểm danh Khuôn mặt DeepFace (GDrive)",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Hệ thống Điểm danh Khuôn mặt DeepFace")
st.caption("Dataset được tải từ Google Drive công khai qua file ZIP.")

# --- 2. Cấu hình & Hằng số ---
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'

# Vui lòng thay thế chuỗi này bằng File ID của file ZIP dataset công khai của bạn.
# VD: GDRIVE_FILE_ID = "1a2b3c4d5e6f7g8h9i0j"
GDRIVE_FILE_ID = "YOUR_GDRIVE_FILE_ID_HERE" 
ZIP_FILENAME = "dataset_archive.zip" 
DATASET_FOLDER = "dataset" 
# Sử dụng detector_backend="opencv" để tránh lỗi TypeError/Keras/TensorFlow
DETECTOR_BACKEND = "opencv"


@st.cache_resource
def load_face_cascade(url, filename):
    """ Tải Haar Cascade cho OpenCV. """
    try:
        r = requests.get(url)
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(r.content)
            classifier = cv2.CascadeClassifier(filename)
            if not classifier.empty():
                return classifier
    except Exception as e:
        st.error(f"Lỗi khi tải hoặc khởi tạo Haar Cascade: {e}")
        return None

face_cascade = load_face_cascade(HAAR_CASCADE_URL, CASCADE_FILENAME)


@st.cache_resource(show_spinner="Đang tải và giải nén Dataset từ Google Drive (Chỉ chạy lần đầu)...")
def download_and_extract_dataset(file_id, zip_name, target_folder):
    """
    Tải file ZIP công khai từ Google Drive và giải nén.
    """
    if not file_id or file_id == "YOUR_GDRIVE_FILE_ID_HERE":
        st.error("❌ Vui lòng thay thế 'YOUR_GDRIVE_FILE_ID_HERE' bằng File ID thực tế của file ZIP dataset.")
        return False
        
    # Kiểm tra nhanh: Nếu thư mục dataset tồn tại và đã có cache DeepFace (đã được xử lý trước đó)
    deepface_cache = os.path.join(target_folder, 'representations_arcface.pkl')
    if os.path.exists(deepface_cache) and os.path.isdir(target_folder) and len(os.listdir(target_folder)) > 1:
         st.success(f"Dataset đã sẵn sàng tại '{target_folder}'. Bỏ qua tải xuống.")
         return True
    
    st.info(f"Đang tải dataset từ Google Drive File ID: {file_id}...")
    DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        # Tải xuống file ZIP
        response = requests.get(DOWNLOAD_URL, stream=True)
        response.raise_for_status() 
        
        # Xử lý trường hợp Google Drive cảnh báo file lớn
        if "confirm" in response.headers.get("Content-Disposition", ""):
            st.warning("Google Drive đang yêu cầu xác nhận tải file lớn. Đang thử tải lại.")
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    params = {'id': file_id, 'confirm': value}
                    response = requests.get(DOWNLOAD_URL, params=params, stream=True)
                    response.raise_for_status()
                    break

        # Lưu file zip
        with open(zip_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Giải nén
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            # DeepFace yêu cầu dataset folder phải nằm ngay trong thư mục gốc
            zip_ref.extractall(".")
            
        st.success(f"Giải nén thành công vào thư mục '{target_folder}'.")
        
        # Xóa file zip tạm
        if os.path.exists(zip_name):
            os.remove(zip_name)
        
        return True

    except Exception as e:
        st.error(f"❌ Lỗi khi tải xuống hoặc giải nén dataset từ Drive: {e}. Vui lòng kiểm tra File ID và quyền chia sẻ công khai.")
        if os.path.exists(zip_name):
            os.remove(zip_name)
        return False


# --- 3. Hàm Phát hiện Khuôn mặt (Dùng cho hiển thị khung) ---
def detect_and_draw_face(image_bytes, cascade):
    """
    Dùng Haar Cascade để phát hiện và vẽ khung khuôn mặt trên ảnh.
    """
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = []
    if cascade is not None:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Vẽ khung vuông lên ảnh
    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    processed_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return processed_image_rgb, len(faces) > 0, len(faces), image_bgr


# --- 4. Hàm DeepFace Recognition (Sử dụng detector_backend="opencv") ---
def verify_face_against_dataset(target_image_path, dataset_folder):
    """
    Sử dụng DeepFace để so sánh ảnh đầu vào với dataset.
    """
    try:
        # THAY ĐỔI QUAN TRỌNG: Sử dụng detector_backend="opencv"
        df_list = DeepFace.find(
            img_path=target_image_path, 
            db_path=dataset_folder, 
            model_name="ArcFace",
            distance_metric="cosine",
            enforce_detection=True, 
            detector_backend=DETECTOR_BACKEND 
        )
        
        if isinstance(df_list, list) and len(df_list) > 0 and not df_list[0].empty:
            best_match = df_list[0].iloc[0]
            identity_path = best_match['identity']
            # Lấy tên người từ tên file (loại bỏ phần mở rộng)
            person_name = os.path.splitext(os.path.basename(identity_path))[0] 
            distance = best_match['ArcFace_cosine'] 
            return person_name, distance
        
        return None, None
    
    except ValueError as e:
        if "Face could not be detected" in str(e):
             # DeepFace.find() sẽ ném ValueError nếu không tìm thấy khuôn mặt
             st.error(f"❌ Lỗi DeepFace: Không phát hiện khuôn mặt để so khớp. Vui lòng thử lại ảnh rõ ràng hơn.")
        else:
            st.error(f"❌ Lỗi DeepFace: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Lỗi trong quá trình so khớp DeepFace: {e}")
        return None, None


# --- 5. Giao diện và Luồng Ứng dụng ---

# 5.1 KHỞI TẠO VÀ TẢI DATASET (Chạy đầu tiên)
dataset_ready = download_and_extract_dataset(GDRIVE_FILE_ID, ZIP_FILENAME, DATASET_FOLDER)

st.markdown("---")

if not dataset_ready:
     st.warning("⚠️ Vui lòng cấu hình đúng File ID ZIP công khai và thử lại.")
     st.stop() # Dừng ứng dụng nếu dataset chưa sẵn sàng

st.info(f"Dataset đã tải xong. DeepFace sẽ sử dụng detector: **{DETECTOR_BACKEND.upper()}**.")


# 5.2 CHỤP ẢNH VÀ XỬ LÝ
captured_file = st.camera_input("Chụp ảnh điểm danh:")

if captured_file is not None:
    if face_cascade is None:
        st.error("Không thể tiếp tục do lỗi tải bộ phân loại khuôn mặt.")
    else:
        image_bytes = captured_file.getvalue()
        
        # Mở spinner trong lúc xử lý
        with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
            
            # 1. Phát hiện khuôn mặt và vẽ khung (Dùng cho hiển thị)
            processed_image_np, face_detected, num_faces, image_bgr = detect_and_draw_face(image_bytes, face_cascade)
            processed_image = Image.fromarray(processed_image_np)
            
            # 2. LƯU ẢNH TẠM THỜI DUY NHẤT (QUAN TRỌNG: Dùng tempfile)
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            TEMP_IMAGE_PATH = temp_file.name
            temp_file.close() # Đóng file handle để cv2.imwrite có thể ghi vào
            
            cv2.imwrite(TEMP_IMAGE_PATH, image_bgr)
            
            # 3. Thực hiện so khớp DeepFace
            match_name, distance = verify_face_against_dataset(TEMP_IMAGE_PATH, DATASET_FOLDER)

        # Xóa file tạm sau khi đã xử lý xong
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)
            
        st.subheader("🖼️ Ảnh đã chụp và Nhận diện")
        st.image(processed_image, caption="Khuôn mặt đã phát hiện được đánh dấu bằng khung màu xanh dương (OpenCV).", use_column_width=True)

        st.markdown("---")
        st.subheader("💡 Kết quả Điểm danh")
        
        if match_name:
            st.balloons()
            st.success(f"✅ **ĐIỂM DANH THÀNH CÔNG!**")
            st.markdown(f"""
            * **Người trùng khớp:** **{match_name}**
            * **Độ tương đồng (Khoảng cách Cosine):** `{distance:.4f}`
            """)
            
        elif face_detected and num_faces > 0:
            st.warning(f"⚠️ **Phát hiện {num_faces} khuôn mặt, nhưng không khớp với dataset.**")
            st.markdown("""
            * **Gợi ý:** Khuôn mặt được phát hiện, nhưng không đủ độ tương đồng với bất kỳ người nào trong dataset.
            """)
            
        else:
            st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
            st.markdown("Vui lòng thử lại. Đảm bảo khuôn mặt của bạn nằm gọn và rõ ràng trong khung hình.")
