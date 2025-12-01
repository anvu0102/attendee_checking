import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import requests

# --- 1. Thiết lập trang Streamlit ---
st.set_page_config(
    page_title="Hệ thống Điểm danh Khuôn mặt",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Hệ thống Điểm danh Khuôn mặt")
st.caption("Sử dụng camera để chụp ảnh và thực hiện nhận diện khuôn mặt.")

# --- 2. Tải và Thiết lập Haar Cascade (Quan trọng cho Cloud Deploy) ---
# Tải bộ phân loại khuôn mặt Haar Cascade một cách đáng tin cậy
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'
face_cascade = None

@st.cache_resource
def load_face_cascade(url, filename):
    """
    Tải Haar Cascade từ URL và lưu trữ trong bộ nhớ đệm của Streamlit.
    """
    try:
        # 1. Thử tải tệp cục bộ
        classifier = cv2.CascadeClassifier(filename)
        if not classifier.empty():
            st.success("Đã tải Haar Cascade cục bộ.")
            return classifier

        # 2. Nếu không có, tải từ GitHub
        st.warning("Không tìm thấy tệp cục bộ. Đang tải từ GitHub...")
        r = requests.get(url)
        if r.status_code == 200:
            # Lưu dữ liệu vào tệp tạm thời hoặc sử dụng trực tiếp bytes
            # Cách phổ biến nhất là tải xuống và đọc
            with open(filename, 'wb') as f:
                f.write(r.content)
            
            classifier = cv2.CascadeClassifier(filename)
            if not classifier.empty():
                st.success("Tải Haar Cascade từ GitHub thành công.")
                return classifier
            
    except Exception as e:
        st.error(f"Lỗi khi tải hoặc khởi tạo Haar Cascade: {e}")
        return None

# Khởi tạo bộ phân loại
face_cascade = load_face_cascade(HAAR_CASCADE_URL, CASCADE_FILENAME)


# --- 3. Hàm Nhận diện Khuôn mặt (Sử dụng OpenCV) ---
def detect_and_draw_face(image_bytes, cascade):
    """
    Nhận diện khuôn mặt trên ảnh đầu vào, vẽ khung, và trả về ảnh đã xử lý 
    cùng với cờ (flag) cho biết có khuôn mặt hay không.
    """
    # Chuyển đổi bytes thành mảng NumPy (OpenCV/BGR format)
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Chuyển sang ảnh xám để nhận diện
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = []
    if cascade is not None:
        # Thực hiện nhận diện khuôn mặt
        faces = cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

    # Vẽ khung vuông lên ảnh
    for (x, y, w, h) in faces:
        # Vẽ khung màu xanh dương (BGR: 255, 0, 0)
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Chuyển lại từ BGR sang RGB để Streamlit/PIL hiển thị đúng
    processed_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return processed_image_rgb, len(faces) > 0, faces


# --- 4. Giao diện và Luồng Ứng dụng ---
st.info("Nhấn 'Chụp ảnh' để Streamlit truy cập camera và bắt đầu quá trình điểm danh.")

# Sử dụng widget camera_input của Streamlit
captured_file = st.camera_input("Chụp ảnh điểm danh:")

if captured_file is not None:
    if face_cascade is None:
        st.error("Không thể tiếp tục do lỗi tải bộ phân loại khuôn mặt. Vui lòng kiểm tra nhật ký.")
    else:
        # Đọc bytes của ảnh
        image_bytes = captured_file.getvalue()
        
        with st.spinner('Đang xử lý ảnh và nhận diện khuôn mặt...'):
            # Gọi hàm xử lý ảnh
            processed_image_np, face_detected, face_locations = detect_and_draw_face(image_bytes, face_cascade)
            
            # Chuyển mảng NumPy về đối tượng Image để hiển thị
            processed_image = Image.fromarray(processed_image_np)
            
        st.subheader("🖼️ Ảnh đã chụp và Nhận diện")
        st.image(processed_image, caption="Khuôn mặt đã phát hiện được đánh dấu bằng khung màu xanh dương.", use_column_width=True)

        # Kiểm tra kết quả
        st.markdown("---")
        st.subheader("💡 Kết quả Điểm danh")
        
        if face_detected:
            st.success(f"✅ **Đã phát hiện {len(face_locations)} khuôn mặt.**")
            st.markdown(f"""
            > **BƯỚC TIẾP THEO (Mô phỏng):** Khuôn mặt đã được chụp và sẵn sàng để so sánh với dữ liệu dataset.
            > 
            > *Giả định:* Nếu khuôn mặt khớp với database:
            > **✅ ĐIỂM DANH THÀNH CÔNG!**
            """)
            
        else:
            st.warning("⚠️ **Không phát hiện thấy khuôn mặt.**")
            st.markdown("Vui lòng thử lại. Đảm bảo khuôn mặt của bạn nằm gọn và rõ ràng trong khung hình, với đủ ánh sáng.")
