import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import requests
import os
import zipfile
from deepface import DeepFace
import tempfile
import time 
import pandas as pd

# --- 1. Thiết lập trang Streamlit ---
st.set_page_config(
    page_title="Hệ thống Điểm danh Khuôn mặt DeepFace (GDrive)",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Hệ thống Điểm danh Khuôn mặt DeepFace")
st.caption("Sử dụng ID Drive và OAuth Credentials từ st.secrets.")

# --- 2. Cấu hình & Hằng số (TẢI TỪ ST.SECRETS) ---
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'

# TẢI CÁC THÔNG TIN TỪ ST.SECRETS
# Đảm bảo các khóa này đã được định nghĩa trong file secrets.toml
try:
    GDRIVE_CLIENT_ID = st.secrets["GDRIVE_CLIENT_ID"]
    GDRIVE_CLIENT_SECRET = st.secrets["GDRIVE_CLIENT_SECRET"]
    GDRIVE_DATASET_FOLDER_ID = st.secrets["GDRIVE_DATASET_ID"] 
    GDRIVE_CHECKLIST_ID = st.secrets["GDRIVE_CHECKLIST_ID"]
    GDRIVE_NEW_DATA_FOLDER_ID = st.secrets["GDRIVE_NEW_DATA_ID"]
except KeyError as e:
    st.error(f"❌ Lỗi: Không tìm thấy khóa {e} trong st.secrets.")
    st.info("Vui lòng đảm bảo bạn đã định nghĩa tất cả các khóa (CLIENT_ID, CLIENT_SECRET, DATASET_ID, CHECKLIST_ID, NEW_DATA_ID) trong file .streamlit/secrets.toml hoặc trong giao diện Secrets của Streamlit Cloud.")
    st.stop()

# Các hằng số khác
DATASET_FOLDER = "dataset" 
CHECKLIST_FILENAME = "checklist.xlsx" 
CHECKLIST_SESSION_KEY = "attendance_df" 
DETECTOR_BACKEND = "opencv"


# --- Hàm Giả Lập Xác Thực Token ---
def get_valid_access_token_mock(client_id, client_secret):
    """ 
    [MOCK] Giả lập quy trình OAuth 2.0 để lấy Access Token.
    Trong thực tế, hàm này sẽ sử dụng Client ID/Secret để yêu cầu và làm mới token.
    """
    if client_id.startswith("YOUR_OAUTH"):
        st.error("❌ Lỗi cấu hình: Client ID vẫn là placeholder. Không thể giả lập token.")
        return None
    
    st.success("✅ Giả lập: Đã sử dụng Client ID/Secret để tạo Access Token (Token thực tế cần luồng OAuth).")
    # Giả lập trả về một Token
    return "MOCK_ACCESS_TOKEN_" + client_id[:5] 


# --- Hàm tải file đơn lẻ từ Google Drive (Dùng cho Checklist XLSX) ---
def download_file_from_gdrive(file_id, output_filename, access_token=None):
    """ Tải file từ Google Drive. Cần Access Token cho các file không công khai. """
    DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    headers = {}
    if access_token:
        # Nếu dùng token, phải thêm vào Header
        headers = {'Authorization': f'Bearer {access_token}'} 
    
    try:
        response = requests.get(DOWNLOAD_URL, stream=True, headers=headers)
        response.raise_for_status() 
        
        if "confirm" in response.headers.get("Content-Disposition", ""):
            params = {'id': file_id, 'confirm': 't'}
            response = requests.get(DOWNLOAD_URL, params=params, stream=True, headers=headers)
            response.raise_for_status()

        with open(output_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"❌ Lỗi khi tải file {output_filename} từ Drive: {e}")
        st.warning("Gợi ý: Kiểm tra ID file, quyền chia sẻ và Access Token.")
        return False


# --- MOCK: Tải Folder Dataset (Mô phỏng) ---
@st.cache_resource(show_spinner="Đang mô phỏng tải Dataset FOLDER từ Google Drive...")
def download_dataset_folder_mock(folder_id, target_folder, access_token):
    """ MOCK: Mô phỏng tải toàn bộ nội dung folder Drive vào thư mục local, sử dụng token. """
    st.warning("⚠️ CHÚ Ý: Hàm này chỉ MOCK (giả lập). Cần Google Drive API thực tế và Access Token hợp lệ.")
    st.info(f"Giả lập: Sử dụng Access Token để truy cập Folder ID: {folder_id}.")

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        try:
            # Tạo cấu trúc file giả định để DeepFace có thể chạy
            temp_img1 = np.zeros((100, 100, 3), dtype=np.uint8)
            temp_img2 = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(target_folder, "1.jpg"), temp_img1) 
            cv2.imwrite(os.path.join(target_folder, "2.jpg"), temp_img2) 
            st.success(f"Mô phỏng: Đã tạo thư mục '{target_folder}' với các file mẫu. Sẵn sàng cho DeepFace.")
            return True
        except Exception as e:
            st.error(f"Lỗi khi tạo file mock: {e}")
            return False

    deepface_cache = os.path.join(target_folder, 'representations_arcface.pkl')
    if os.path.isdir(target_folder) and (len(os.listdir(target_folder)) > 2 or os.path.exists(deepface_cache)):
         st.success(f"Dataset folder đã sẵn sàng tại '{target_folder}'. Bỏ qua tải xuống.")
         return True
    
    return False


@st.cache_data(show_spinner="Đang tải và xử lý Checklist (XLSX) từ Google Drive...")
def load_checklist(file_id, filename, access_token):
    """ Tải checklist XLSX và đọc thành DataFrame. """
    
    if not os.path.exists(filename):
        # Truyền token vào hàm download
        download_file_from_gdrive(file_id, filename, access_token)
        
    if os.path.exists(filename):
        try:
            # ĐỌC FILE XLSX
            df = pd.read_excel(filename) 
            return df
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file checklist: {e}. Đảm bảo file có định dạng XLSX.")
            return None
    return None

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

# --- 3. Hàm Phát hiện Khuôn mặt (Giữ nguyên) ---
def detect_and_draw_face(image_bytes, cascade):
    """ Dùng Haar Cascade để phát hiện và vẽ khung khuôn mặt trên ảnh. """
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = []
    if cascade is not None:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    processed_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return processed_image_rgb, len(faces) > 0, len(faces), image_bgr

# --- 4. Hàm DeepFace Recognition (Giữ nguyên) ---
def verify_face_against_dataset(target_image_path, dataset_folder):
    """ Sử dụng DeepFace để so sánh ảnh đầu vào với dataset. """
    try:
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
            stt_match = os.path.splitext(os.path.basename(identity_path))[0] 
            distance = best_match['ArcFace_cosine'] 
            return stt_match, distance
        return None, None
    except Exception as e:
        if "Face could not be detected" in str(e):
             st.error(f"❌ Lỗi DeepFace: Không phát hiện khuôn mặt để so khớp.")
        else:
            st.error(f"❌ Lỗi DeepFace: {e}")
        return None, None

# --- 5. Hàm MOCK UPLOAD lên Google Drive ---
def upload_to_gdrive_mock(file_path, drive_folder_id, drive_filename, access_token):
    """
    [MOCK/PLACEHOLDER] Hàm giả định việc tải file lên Google Drive, sử dụng token.
    """
    if access_token is None:
        st.error("❌ Lỗi Auth: Không thể upload vì không có Access Token hợp lệ.")
        return False
    
    st.success(f"✅ **Mô phỏng Upload:** Tải file '{drive_filename}' thành công.")
    st.info(f"Đã giả lập lưu vào Drive Folder ID: **{drive_folder_id}** bằng Access Token.")
    
    return True

# --- 6. Logic Ghi Dữ Liệu và Lưu Ảnh Mới ---

def update_checklist_and_save_new_data(stt_match, session_name, image_bytes, access_token):
    """
    Cập nhật DataFrame checklist và lưu ảnh mới lên Drive.
    """
    if CHECKLIST_SESSION_KEY not in st.session_state:
        st.error("Lỗi: Không tìm thấy DataFrame checklist trong Session State.")
        return

    df = st.session_state[CHECKLIST_SESSION_KEY]
    
    # 1. Cập nhật Checklist (Đánh 'X')
    if stt_match is not None:
        try:
            stt_col = df.columns[0] 
            stt_match_str = str(stt_match).split('_')[0] 
            
            row_index = df[df[stt_col].astype(str).str.contains(stt_match_str, regex=False)].index
            
            if not row_index.empty:
                df.loc[row_index[0], session_name] = 'X'
                st.session_state[CHECKLIST_SESSION_KEY] = df 
                
                st.success(f"✅ **Đã cập nhật điểm danh** cho STT **{df.loc[row_index[0], stt_col]}** vào cột **{session_name}**.")
                st.info(f"⚠️ **Mô phỏng:** DataFrame này cần được ghi trở lại file XLSX Drive ID: **{GDRIVE_CHECKLIST_ID}** bằng Access Token.")
                
            else:
                st.warning(f"⚠️ Không tìm thấy STT **{stt_match_str}** trong checklist để cập nhật.")
        except Exception as e:
            st.error(f"Lỗi khi cập nhật checklist: {e}")
            
    # 2. Lưu ảnh mới lên Drive (Nếu không khớp)
    else: 
        if 'new_data_counter' not in st.session_state:
            st.session_state['new_data_counter'] = 0
            
        st.session_state['new_data_counter'] += 1
        new_counter = st.session_state['new_data_counter']
        
        session_num = session_name.replace("Buổi ", "")
        drive_filename = f"B{session_num}_{new_counter}.jpg" 
        
        # --- TẠO FILE TẠM ĐỂ UPLOAD ---
        temp_file_for_upload = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        TEMP_UPLOAD_PATH = temp_file_for_upload.name
        temp_file_for_upload.close()
        
        try:
            image_to_save = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_to_save.save(TEMP_UPLOAD_PATH, format='JPEG')
            
            # 2. Gọi hàm Upload Drive (MOCK) và truyền token
            upload_to_gdrive_mock(TEMP_UPLOAD_PATH, GDRIVE_NEW_DATA_FOLDER_ID, drive_filename, access_token)

        except Exception as e:
             st.error(f"❌ Lỗi khi tạo file tạm hoặc gọi hàm upload: {e}")
        finally:
            if os.path.exists(TEMP_UPLOAD_PATH):
                os.remove(TEMP_UPLOAD_PATH)


# --- 7. Giao diện và Luồng Ứng dụng ---
# LẤY TOKEN ĐẦU TIÊN
ACCESS_TOKEN = get_valid_access_token_mock(GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET)

if not ACCESS_TOKEN:
    st.error("❌ Không thể tiếp tục do không lấy được Access Token hợp lệ từ quy trình OAuth giả lập.")
    st.stop()


# 7.1 KHỞI TẠO VÀ TẢI DATASET & CHECKLIST
# Tải Folder Dataset (MOCK)
dataset_ready = download_dataset_folder_mock(GDRIVE_DATASET_FOLDER_ID, DATASET_FOLDER, ACCESS_TOKEN) 
# Tải Checklist (XLSX)
checklist_df = load_checklist(GDRIVE_CHECKLIST_ID, CHECKLIST_FILENAME, ACCESS_TOKEN)

if checklist_df is not None:
    st.session_state[CHECKLIST_SESSION_KEY] = checklist_df
    
st.markdown("---")

if not dataset_ready:
     st.warning("⚠️ Lỗi mô phỏng tải Dataset Folder. Vui lòng kiểm tra ID Drive Folder và quyền truy cập.")
     st.stop()
     
if checklist_df is None:
     st.warning("⚠️ Lỗi tải hoặc đọc file Checklist. Vui lòng kiểm tra File ID và quyền truy cập bằng token.")
     st.stop()


st.info(f"Dataset đã sẵn sàng. Checklist có {len(checklist_df)} người.")


# 7.2 CHỌN BUỔI HỌC (Dropdown)
attendance_cols = [col for col in st.session_state[CHECKLIST_SESSION_KEY].columns if "Buổi" in col]

if not attendance_cols:
     st.error("Không tìm thấy cột 'Buổi' trong checklist. Vui lòng kiểm tra lại cấu trúc file XLSX.")
     st.stop()

selected_session = st.selectbox(
    "1️⃣ **Chọn Buổi Điểm Danh**", 
    attendance_cols, 
    index=0,
    help="Chọn buổi tương ứng để cập nhật cột điểm danh trong checklist."
)
st.success(f"Đang điểm danh cho: **{selected_session}**")

st.markdown("---")

# 7.3 CHỤP ẢNH VÀ XỬ LÝ
captured_file = st.camera_input("2️⃣ Chụp ảnh điểm danh:")

if captured_file is not None:
    
    image_bytes = captured_file.getvalue()
    
    with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
        
        processed_image_np, face_detected, num_faces, image_bgr = detect_and_draw_face(image_bytes, face_cascade)
        processed_image = Image.fromarray(processed_image_np)
        
        # LƯU ẢNH TẠM THỜI cho DeepFace so khớp
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        TEMP_IMAGE_PATH = temp_file.name
        temp_file.close() 
        
        cv2.imwrite(TEMP_IMAGE_PATH, image_bgr)
        
        # Thực hiện so khớp DeepFace
        stt_match, distance = verify_face_against_dataset(TEMP_IMAGE_PATH, DATASET_FOLDER)

    # Xóa file tạm của DeepFace
    if os.path.exists(TEMP_IMAGE_PATH):
        os.remove(TEMP_IMAGE_PATH)
        
    st.subheader("🖼️ Ảnh đã chụp và Nhận diện")
    st.image(processed_image, caption="Khuôn mặt đã phát hiện được đánh dấu.", use_column_width=True)

    st.markdown("---")
    st.subheader("💡 Kết quả Điểm danh")
    
    if stt_match:
        st.balloons()
        st.success(f"✅ **ĐIỂM DANH THÀNH CÔNG!**")
        st.markdown(f"""
        * **STT trùng khớp:** **{stt_match}**
        * **Độ tương đồng (Khoảng cách Cosine):** `{distance:.4f}`
        """)
        # Cập nhật checklist (truyền token)
        update_checklist_and_save_new_data(stt_match, selected_session, None, ACCESS_TOKEN)
        
    elif face_detected and num_faces == 1:
        st.warning(f"⚠️ **Phát hiện 1 khuôn mặt, nhưng không khớp với dataset.**")
        # Lưu ảnh mới (truyền image_bytes và token)
        update_checklist_and_save_new_data(None, selected_session, image_bytes, ACCESS_TOKEN) 
        
    elif face_detected and num_faces > 1:
        st.error(f"❌ **Phát hiện nhiều khuôn mặt ({num_faces}). Vui lòng chỉ có 1 người trong khung hình.**")

    else:
        st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
        st.markdown("Vui lòng thử lại. Đảm bảo khuôn mặt của bạn nằm gọn và rõ ràng trong khung hình.")

st.markdown("---")
st.subheader("📋 Trạng thái Checklist Hiện tại (Trong Session)")
if CHECKLIST_SESSION_KEY in st.session_state:
    st.dataframe(st.session_state[CHECKLIST_SESSION_KEY])
