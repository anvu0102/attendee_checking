import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import requests
import os
import zipfile # Thêm thư viện để giải nén
from deepface import DeepFace

# --- 1. Thiết lập trang Streamlit ---
st.set_page_config(
    page_title="Hệ thống Điểm danh Khuôn mặt DeepFace (GDrive)",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Hệ thống Điểm danh Khuôn mặt DeepFace")
st.caption("Dataset được tải từ Google Drive công khai.")

# --- 2. Tải và Thiết lập Haar Cascade (Dùng cho phát hiện khung) ---
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'
face_cascade = None
TEMP_IMAGE_PATH = "captured_face.jpg" # Đường dẫn tạm để lưu ảnh chụp

# --- Cấu hình Google Drive Dataset ---
# Vui lòng thay thế chuỗi này bằng File ID của file ZIP dataset công khai của bạn.
GDRIVE_FILE_ID = "1qX4I983WrBYMWdQals3g_ijbeepf8BtG" 
ZIP_FILENAME = "dataset.zip" 
DATASET_FOLDER = "dataset" 

@st.cache_resource
def load_face_cascade(url, filename):
    """ Tải Haar Cascade (giống code cũ). """
    try:
        r = requests.get(url)
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(r.content)
            
            classifier = cv2.CascadeClassifier(filename)
            if not classifier.empty():
                st.success("Tải Haar Cascade thành công.")
                return classifier
            
    except Exception as e:
        st.error(f"Lỗi khi tải hoặc khởi tạo Haar Cascade: {e}")
        return None

# Khởi tạo bộ phân loại
face_cascade = load_face_cascade(HAAR_CASCADE_URL, CASCADE_FILENAME)


@st.cache_resource(show_spinner="Đang tải và giải nén Dataset từ Google Drive (Chỉ chạy lần đầu)...")
def download_and_extract_dataset(file_id, zip_name, target_folder):
    """
    Tải file ZIP công khai từ Google Drive và giải nén vào thư mục DeepFace dataset.
    Sử dụng @st.cache_resource để chỉ chạy một lần.
    """
    if not file_id or file_id == "YOUR_GDRIVE_FILE_ID_HERE":
        st.error("❌ Vui lòng thay thế 'YOUR_GDRIVE_FILE_ID_HERE' bằng File ID thực tế.")
        return False
        
    # Kiểm tra nếu dataset đã được giải nén thành công (để tránh tải lại)
    if os.path.exists(target_folder) and os.path.isdir(target_folder) and len(os.listdir(target_folder)) > 0:
        # Kiểm tra nhanh: Nếu file `representations_arcface.pkl` của DeepFace đã tồn tại
        # thì dataset đã sẵn sàng.
        deepface_cache = os.path.join(target_folder, 'representations_arcface.pkl')
        if os.path.exists(deepface_cache):
             st.success(f"Dataset đã sẵn sàng tại '{target_folder}'. Bỏ qua tải xuống.")
             return True
        st.info("Dataset folder tồn tại nhưng thiếu cache DeepFace, đang thử tải lại...")


    st.info(f"Đang tải dataset từ Google Drive File ID: {file_id}...")
    
    # URL tải file từ Google Drive
    DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        response = requests.get(DOWNLOAD_URL, stream=True)
        response.raise_for_status() 
        
        # Xử lý trường hợp Google Drive cảnh báo về dung lượng lớn (cookies)
        if "confirm" in response.headers.get("Content-Disposition", ""):
            st.warning("Google Drive đang yêu cầu xác nhận tải file lớn. Đang thử tải lại.")
            
            # Lấy confirm token
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

        st.success(f"Tải xuống {zip_name} thành công.")
        
        # Giải nén
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            zip_ref.extractall(target_folder)
            
        st.success(f"Giải nén thành công vào thư mục '{target_folder}'.")
        
        # Xóa file zip tạm
        os.remove(zip_name)
        
        return True

    except Exception as e:
        st.error(f"❌ Lỗi khi tải xuống hoặc giải nén dataset từ Drive: {e}")
        if os.path.exists(zip_name):
            os.remove(zip_name)
        return False


# --- 3. Hàm Phát hiện Khuôn mặt (Giữ nguyên) ---
def detect_and_draw_face(image_bytes, cascade):
    """
    Nhận diện khuôn mặt trên ảnh đầu vào, vẽ khung, và trả về ảnh đã xử lý 
    cùng với cờ (flag) cho biết có khuôn mặt hay không.
    """
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = []
    if cascade is not None:
        faces = cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    processed_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return processed_image_rgb, len(faces) > 0, len(faces), image_bgr # Thêm len(faces) và image_bgr


# --- 4. Hàm DeepFace Recognition ---
def verify_face_against_dataset(target_image_path, dataset_folder):
    try:
        df_list = DeepFace.find(
            img_path=target_image_path, 
            db_path=dataset_folder, 
            model_name="ArcFace",
            distance_metric="cosine",
            enforce_detection=True, 
            # THÊM THAM SỐ NÀY để tránh RetinaFace gây lỗi
            detector_backend="opencv" 
        )
        
        if isinstance(df_list, list) and len(df_list) > 0 and not df_list[0].empty:
            best_match = df_list[0].iloc[0]
            identity_path = best_match['identity']
            person_name = os.path.splitext(os.path.basename(identity_path))[0]
            distance = best_match['ArcFace_cosine'] 
            return person_name, distance
        
        return None, None
    
    except ValueError as e:
        if "Face could not be detected" in str(e):
             st.error("❌ Không phát hiện khuôn mặt trong ảnh chụp. Vui lòng thử lại.")
        else:
            st.error(f"❌ Lỗi DeepFace: {e}")
        return None, None
    except Exception as e:
        st.error(f"❌ Lỗi trong quá trình so khớp DeepFace: {e}")
        return None, None


# --- 5. Giao diện và Luồng Ứng dụng ---
st.info(f"Đảm bảo đã thay thế **'YOUR_GDRIVE_FILE_ID_HERE'** bằng File ID của file ZIP dataset công khai trên Google Drive.")

# 5.1 KHỞI TẠO VÀ TẢI DATASET
dataset_ready = download_and_extract_dataset(GDRIVE_FILE_ID, ZIP_FILENAME, DATASET_FOLDER)

st.markdown("---")

# 5.2 CHỤP ẢNH VÀ XỬ LÝ
captured_file = st.camera_input("Chụp ảnh điểm danh:")

if captured_file is not None:
    if not dataset_ready: # Kiểm tra dataset đã sẵn sàng chưa
        st.error("Không thể xử lý do lỗi tải dataset từ Google Drive.")
    elif face_cascade is None:
        st.error("Không thể tiếp tục do lỗi tải bộ phân loại khuôn mặt.")
    else:
        image_bytes = captured_file.getvalue()
        
        with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
            # 1. Phát hiện khuôn mặt và vẽ khung
            processed_image_np, face_detected, num_faces, image_bgr = detect_and_draw_face(image_bytes, face_cascade)
            
            processed_image = Image.fromarray(processed_image_np)
            
            # 2. Lưu ảnh tạm thời để DeepFace sử dụng
            cv2.imwrite(TEMP_IMAGE_PATH, image_bgr)
            
            # 3. Thực hiện so khớp DeepFace
            match_name, distance = verify_face_against_dataset(TEMP_IMAGE_PATH, DATASET_FOLDER)

        # Xóa file tạm sau khi đã xử lý
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)
            
        st.subheader("🖼️ Ảnh đã chụp và Nhận diện")
        st.image(processed_image, caption="Khuôn mặt đã phát hiện được đánh dấu bằng khung màu xanh dương.", use_column_width=True)

        st.markdown("---")
        st.subheader("💡 Kết quả Điểm danh")
        
        if match_name:
            st.success(f"✅ **ĐIỂM DANH THÀNH CÔNG!**")
            st.markdown(f"""
            * **Người trùng khớp:** **{match_name}**
            * **Khoảng cách Cosine (ArcFace):** {distance:.4f}
            """)
            
        elif face_detected:
            st.warning(f"⚠️ **Phát hiện {num_faces} khuôn mặt, nhưng không khớp với dataset.**")
            st.markdown("""
            * Vui lòng kiểm tra lại ánh sáng hoặc độ rõ của khuôn mặt.
            * Đảm bảo tên file ảnh trong dataset khớp với tên người đăng ký.
            """)
            
        else:
            st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
            st.markdown("Vui lòng thử lại. Đảm bảo khuôn mặt của bạn nằm gọn và rõ ràng trong khung hình.")
