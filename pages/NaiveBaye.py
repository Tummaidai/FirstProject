from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import streamlit as st
import numpy as np

st.title("Naive Bayes Classification - Hospital Data")

# โหลดข้อมูล
df = pd.read_csv("./data/Hospital_binary.csv")
st.subheader("ตัวอย่างข้อมูล")
st.write(df.head(10))

# สมมติว่าคอลัมน์สุดท้ายเป็น target
target_col = df.columns[-1]
feature_cols = df.columns[:-1]

X = df[feature_cols]
y = df[target_col]

# แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)

# ส่วนรับข้อมูลจากผู้ใช้
st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์")

user_input = []
for col in feature_cols:
    val = st.number_input(f"กรอกค่า {col}", value=float(X[col].mean()))
    user_input.append(val)

if st.button("พยากรณ์"):
    x_input = np.array([user_input])
    y_predict2 = clf.predict(x_input)
    st.write("ผลการพยากรณ์:", y_predict2[0])
else:
    st.write("ยังไม่ได้กดปุ่มทำนาย")
