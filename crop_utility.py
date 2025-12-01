import os
import cv2
import numpy as np
import requests
from PIL import Image

# -------------------------------------------------------------------
# --- MOCKUP CÁC HẰNG SỐ CẦN THIẾT (Giả định lấy từ config.py) ---
# Thư mục dataset hiện tại chứa ảnh gốc
DATASET_FOLDER = "dataset" 
# Tên file Haar Cascade đã tải
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
# URL để tải Haar Cascade (nếu chưa có)
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
# Tên thư mục đầu ra chứa ảnh đã cắt
CROPPED_DATASET_FOLDER = "cropped_dataset" 
# -------------------------------------------------------------------

# --- HÀM TẢI VÀ KHỞI TẠO HAAR CASCADE ---
def load_face_cascade(url, filename):
    """ Tải Haar Cascade cho OpenCV (Dùng lại logic từ file gốc). """
    try:
        if not os.path.exists(filename):
            print(f"Đang tải {filename}...")
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(r.content)
            else:
                print(f"Lỗi tải file Haar Cascade: HTTP status {r.status_code}")
                return None

        classifier = cv2.CascadeClassifier(filename)
        if not classifier.empty():
            print("✅ Haar Cascade đã sẵn sàng.")
            return classifier
        else:
            print("Lỗi: Khởi tạo Haar Cascade thất bại.")
            return None
    except Exception as e:
        print(f"Lỗi khi tải hoặc khởi tạo Haar Cascade: {e}")
        return None

# Load cascade (chỉ chạy 1 lần)
face_cascade = load_face_cascade(HAAR_CASCADE_URL, CASCADE_FILENAME)
# -------------------------------------------------------------------


def crop_dataset_images(input_folder, output_folder, cascade, padding_percent=0.2):
    """
    Cắt khuôn mặt từ tất cả ảnh trong thư mục đầu vào và lưu vào thư mục đầu ra.
    Chỉ xử lý các ảnh có EXACTLY 1 khuôn mặt được phát hiện.
    
    Args:
        input_folder (str): Đường dẫn đến thư mục chứa ảnh dataset gốc.
        output_folder (str): Đường dẫn đến thư mục lưu ảnh đã cắt.
        cascade (cv2.CascadeClassifier): Bộ phân loại Haar Cascade đã tải.
        padding_percent (float): Phần trăm padding thêm vào khung khuôn mặt (ví dụ: 0.2 là 20%).
    """
    if cascade is None:
        print("Lỗi: Haar Cascade chưa được tải. Không thể tiến hành cắt ảnh.")
        return

    # 1. Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Đã tạo thư mục đầu ra: {output_folder}")
    else:
        print(f"Thư mục đầu ra đã tồn tại: {output_folder}")

    total_files = 0
    cropped_count = 0
    skipped_count = 0

    # 2. Lặp qua tất cả các file trong thư mục đầu vào
    for filename in os.listdir(input_folder):
        # Chỉ xử lý các file ảnh phổ biến
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            total_files += 1
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # Đọc ảnh (dùng cv2.IMREAD_COLOR để đảm bảo đọc đủ 3 kênh)
                image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    print(f"❌ Bỏ qua {filename}: Không thể đọc file ảnh.")
                    skipped_count += 1
                    continue
                
                # Chuyển sang ảnh xám để phát hiện
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                
                # Phát hiện khuôn mặt
                faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) == 1:
                    # Phát hiện 1 khuôn mặt: Tiến hành cắt
                    (x, y, w, h) = faces[0]
                    
                    # Thêm Padding (20% theo logic trong main_app)
                    padding = int(padding_percent * w)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image_bgr.shape[1], x + w + padding)
                    y2 = min(image_bgr.shape[0], y + h + padding)

                    # Cắt ảnh
                    cropped_face_bgr = image_bgr[y1:y2, x1:x2]
                    
                    # Lưu ảnh đã cắt
                    cv2.imwrite(output_path, cropped_face_bgr)
                    cropped_count += 1
                    print(f"✅ Đã cắt và lưu: {filename}")
                    
                else:
                    # Bỏ qua nếu không phải 1 khuôn mặt
                    if len(faces) == 0:
                        print(f"⚠️ Bỏ qua {filename}: Không phát hiện khuôn mặt.")
                    else:
                        print(f"⚠️ Bỏ qua {filename}: Phát hiện {len(faces)} khuôn mặt.")
                    skipped_count += 1

            except Exception as e:
                print(f"❌ Lỗi xử lý file {filename}: {e}")
                skipped_count += 1
                
    print("\n--- TÓM TẮT XỬ LÝ DATASET ---")
    print(f"Thư mục nguồn: {input_folder}")
    print(f"Thư mục đích: {output_folder}")
    print(f"Tổng số file ảnh được xử lý: {total_files}")
    print(f"Số file đã cắt và lưu: {cropped_count}")
    print(f"Số file đã bỏ qua (0 hoặc >1 khuôn mặt/Lỗi): {skipped_count}")

# -------------------------------------------------------------------
# --- CÁCH SỬ DỤNG ---
if __name__ == "__main__":
    # Đặt tên thư mục nguồn và đích
    INPUT_DIR = DATASET_FOLDER      # Mặc định là 'dataset'
    OUTPUT_DIR = CROPPED_DATASET_FOLDER # Mặc định là 'cropped_dataset'
    
    # CHÚ Ý: Cần đảm bảo thư mục INPUT_DIR ('dataset') đã tồn tại 
    # và chứa ảnh trước khi chạy hàm này.
    
    if face_cascade and os.path.exists(INPUT_DIR):
        print("Bắt đầu quá trình cắt ảnh dataset...")
        crop_dataset_images(INPUT_DIR, OUTPUT_DIR, face_cascade)
        print("Hoàn thành quá trình cắt ảnh.")
    elif not os.path.exists(INPUT_DIR):
        print(f"❌ Lỗi: Thư mục dataset nguồn '{INPUT_DIR}' không tồn tại. Vui lòng tạo thư mục và đặt ảnh vào.")
    else:
        print("❌ Lỗi: Không thể chạy hàm cắt ảnh do lỗi Haar Cascade.")
