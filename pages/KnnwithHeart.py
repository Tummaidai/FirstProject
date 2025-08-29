from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title('การทำนายข้อมูลโรงพยาบาลด้วยเทคนิค K-Nearest Neighbor')

# โหลดข้อมูล
dt = pd.read_csv("./data/Hospital_binary.csv")

# แสดงข้อมูล
st.subheader("ข้อมูลส่วนแรก 10 แถว")
st.write(dt.head(10))

st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

# สถิติพื้นฐาน
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# การเลือกฟีเจอร์
target_col = dt.columns[-1]   # สมมติว่าคอลัมน์สุดท้ายคือ target
feature_cols = dt.columns[:-1]

st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", feature_cols)

# วาดกราฟ boxplot
st.write(f"### 🎯 Boxplot: {feature} แยกตาม {target_col}")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x=target_col, y=feature, ax=ax)
st.pyplot(fig)

# Pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue=target_col)
    st.pyplot(fig2)

# ส่วนทำนายข้อมูล
st.subheader("🔮 ทำนายข้อมูลใหม่")

user_input = []
for col in feature_cols:
    val = st.number_input(f"กรอกค่า {col}", value=float(dt[col].mean()))
    user_input.append(val)

if st.button("ทำนายผล"):
    X = dt[feature_cols]
    y = dt[target_col]

    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    x_input = np.array([user_input])
    out = Knn_model.predict(x_input)

    st.write("ผลการทำนาย:", out[0])

    # แสดงภาพประกอบ (อาจเปลี่ยนตามเงื่อนไขจริง)
    if out[0] == 1:
        st.image("./img/b-11.17.15-full-1024x493.jpg")
    else:
        st.image("./img/ef6d4420-e97e-11ed-a1e9-596292a5b691_webp_original.jpg")
else:
    st.write("ยังไม่ได้กดปุ่มทำนาย")
