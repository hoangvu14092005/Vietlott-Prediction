# src/data_loader.py
import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class LotteryDataLoader:
    def __init__(self, db_path="data/lottery.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Khởi tạo cấu trúc Database nếu chưa có."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Tạo bảng chứa ID, Ngày, và 6 số
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kqxsmb_645 (
                draw_id INTEGER PRIMARY KEY,
                draw_date TEXT,
                num1 INTEGER, num2 INTEGER, num3 INTEGER, 
                num4 INTEGER, num5 INTEGER, num6 INTEGER
            )
        ''')
        conn.commit()
        conn.close()

    def import_from_npy(self, npy_path="lottery_data.npy"):
        """
        Hàm quan trọng: Đọc file .npy của bạn và nạp vào DB.
        """
        print(f"📂 [Data Loader] Đang kiểm tra file '{npy_path}'...")
        
        # 1. Kiểm tra file có tồn tại không
        if not os.path.exists(npy_path):
            print(f"❌ Lỗi: Không tìm thấy file '{npy_path}' ở thư mục gốc!")
            return

        try:
            # 2. Load dữ liệu (allow_pickle=True để đọc được cả dạng list)
            data = np.load(npy_path, allow_pickle=True)
            print(f"   -> Kiểu dữ liệu gốc: {type(data)}")
            
            # 3. Chuẩn hóa về dạng Numpy Array 2 chiều
            if isinstance(data, list):
                # Nếu lưu dạng list các array, ta gộp lại
                try:
                    data = np.vstack(data).astype(int)
                except:
                    data = np.array(data)
            
            # Xử lý trường hợp data object (thường do độ dài dòng không đều, nhưng Vietlott thì phải đều)
            if data.dtype == object:
                data = np.vstack(data).astype(int)

            shape = data.shape
            print(f"   -> Đã nhận diện Shape: {shape} (Dòng x Cột)")

            # 4. Kiểm tra dữ liệu hợp lệ (phải có ít nhất 6 cột số)
            if len(shape) != 2 or shape[1] < 6:
                print("❌ Lỗi cấu trúc: Dữ liệu phải có ít nhất 6 cột (num1...num6)")
                return
            
            # 5. Nạp vào Database
            print("   -> Đang chuyển đổi vào Database...")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Xóa sạch dữ liệu cũ (dummy data) để tránh bị lẫn
            cursor.execute("DELETE FROM kqxsmb_645")
            
            db_rows = []
            # Giả lập ngày tháng (vì file npy thường không lưu ngày)
            start_date = datetime(2016, 7, 18) 
            
            for i in range(len(data)):
                # Lấy 6 số đầu tiên, sắp xếp tăng dần cho chuẩn
                nums = sorted(data[i][:6]) 
                
                # Tạo ID và Ngày giả lập
                draw_id = i + 1
                draw_date = (start_date + timedelta(days=i*2)).strftime("%Y-%m-%d")
                
                record = (draw_id, draw_date, 
                          int(nums[0]), int(nums[1]), int(nums[2]), 
                          int(nums[3]), int(nums[4]), int(nums[5]))
                db_rows.append(record)
            
            # Insert một lần (Bulk Insert) cho nhanh
            cursor.executemany("INSERT INTO kqxsmb_645 VALUES (?,?,?,?,?,?,?,?)", db_rows)
            conn.commit()
            conn.close()
            
            print(f"✅ Thành công! Đã nạp {len(db_rows)} kỳ quay từ file .npy vào Database.")

        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng khi đọc file .npy: {e}")

    def load_data(self):
        """Hàm lấy dữ liệu sạch ra để train"""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM kqxsmb_645 ORDER BY draw_id ASC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df