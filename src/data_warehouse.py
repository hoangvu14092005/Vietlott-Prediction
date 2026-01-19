# src/data_warehouse.py
import sqlite3
import pandas as pd
import os

class LotteryWarehouse:
    def __init__(self, db_path="data/lottery_warehouse.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Khởi tạo 2 bảng dữ liệu chuẩn cho Mega và Power"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 1. BẢNG MEGA 6/45 (Nâng cấp)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mega_645 (
                draw_id INTEGER PRIMARY KEY,
                draw_date TEXT,
                num1 INTEGER, num2 INTEGER, num3 INTEGER, 
                num4 INTEGER, num5 INTEGER, num6 INTEGER,
                jackpot_val REAL DEFAULT 0,    -- Tiền Jackpot (VND)
                winners INTEGER DEFAULT 0      -- Số người trúng
            )
        ''')

        # 2. BẢNG POWER 6/55 (Chuẩn bị sẵn cho tương lai)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS power_655 (
                draw_id INTEGER PRIMARY KEY,
                draw_date TEXT,
                num1 INTEGER, num2 INTEGER, num3 INTEGER, 
                num4 INTEGER, num5 INTEGER, num6 INTEGER,
                num_bonus INTEGER,
                jp1_val REAL DEFAULT 0,
                jp1_winners INTEGER DEFAULT 0,
                jp2_val REAL DEFAULT 0,
                jp2_winners INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()

    def load_data(self, product="mega"):
        """Hàm load dữ liệu ra DataFrame để train"""
        conn = sqlite3.connect(self.db_path)
        table = "mega_645" if product == "mega" else "power_655"
        df = pd.read_sql(f"SELECT * FROM {table} ORDER BY draw_id ASC", conn)
        conn.close()
        return df