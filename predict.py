import os
os.environ["KERAS_BACKEND"] = "torch"
import cv2
import numpy as np
import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cấu hình Class Names và Model cho từng loại cây
PLANT_CONFIG = {
    "cam": {
        "model_path": "F:/TTCS/APP/models/model_cay_cam_finetuned.keras",
        "class_names": [
            "benh_da_dang", 
            "benh_loet", 
            "la_khoe", 
            "thieu_dinh_duong"
        ]
    },
    "chanh": {
        "model_path": "F:/TTCS/APP/models/model_cay_chanh_finetuned.keras",
        "class_names": [
            "La_khoe", 
            "La_vang_thoi_re", 
            "Lo_loet", 
            "dom_den"
        ]
    },
    "dua_hau": {
        "model_path": "F:/TTCS/APP/models/model_cay_dua_hau_finetuned.keras",
        "class_names": [
            "Benh_suong_mai", 
            "Benh_than_thu", 
            "La_khoe_manh", 
            "virus_kham"
        ]
    },
    "le": {
        "model_path": "F:/TTCS/APP/models/model_cay_le_finetuned.keras",
        "class_names": [
            "Chay_la", 
            "Dom_la", 
            "La_khoe", 
            "Sen_tran"
        ]
    },
    "oi": {
        "model_path": "F:/TTCS/APP/models/model_cay_oi_finetuned.keras",
        "class_names": [
            "La_khoe", 
            "Loet", 
            "Than_thu", 
            "ri_sat"
        ]
    },
    "xoai": {
        "model_path": "F:/TTCS/APP/models/model_cay_xoai_finetuned.keras",
        "class_names": [
            "Benh_loet_do_vi_khuan", 
            "Benh_phan_trang", 
            "Benh_than_thu", 
            "Bo_canh_cung_cat", 
            "Chet_dan", 
            "La_khoe", 
            "Nam_bo_hong", 
            "Ruoi_gay_u"
        ]
    },
    "cay_buoi": {
        "model_path": os.path.join(BASE_DIR, "Keras_Models_Final", "Citrus", "citrus_model.keras"),
        "class_names": [
            "Bệnh đốm đen", 
            "Bệnh loét vi khuẩn", 
            "Bệnh vàng lá gân xanh", 
            "Cây khỏe mạnh"
        ]
    },
    "thanh_long": {
        "model_path": os.path.join(BASE_DIR, "Keras_Models_Final", "Dragonfruit", "dragonfruit_model.keras"),
        "class_names": [
            "Bệnh đốm đen", 
            "Bệnh đốm nâu", 
            "Bệnh thối rễ", 
            "Cây khỏe mạnh"
        ]
    },
    "vai": {
        "model_path": os.path.join(BASE_DIR, "Keras_Models_Final", "Lychee", "lychee_model.keras"),
        "class_names": [
            "Bệnh phấn trắng", 
            "Bệnh sương mai", 
            "Bệnh thán thư", 
            "Cây khỏe mạnh"
        ]
    }
}

# Biến toàn cục để lưu cache model
loaded_models = {}

def get_model(plant_type):
    """Load và cache model theo loại cây"""
    if plant_type not in PLANT_CONFIG:
        raise ValueError(f"Không hỗ trợ loại cây: {plant_type}")
        
    if plant_type not in loaded_models:
        path = PLANT_CONFIG[plant_type]["model_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy model tại {path}")
            
        print(f"Loading model cho {plant_type}...")
        loaded_models[plant_type] = keras.models.load_model(path, compile=False)        
    return loaded_models[plant_type]

def preprocess_image(image_path, target_size=(224, 224)):
    """Sử dụng OpenCV để đọc và tiền xử lý ảnh"""
    # Đọc ảnh bằng OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Không thể đọc ảnh. Kiểm tra lại đường dẫn.")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_disease(image_path, plant_type):
    """Hàm chính để dự đoán bệnh"""
    try:
        # 1. Đọc và tiền xử lý ảnh
        processed_img = preprocess_image(image_path)
        
        # 2. Load model
        model = get_model(plant_type)
        class_names = PLANT_CONFIG[plant_type]["class_names"]
        
        # 3. Dự đoán
        predictions = model.predict(processed_img)
        
        # 4. Trích xuất kết quả
        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_index])
        disease_name = class_names[predicted_class_index]
        
        return {
            "status": "success",
            "plant_type": plant_type,
            "disease": disease_name,
            "confidence": confidence
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }