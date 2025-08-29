import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.header("Decision Tree for classification")

# อ่านไฟล์ Hospital_binary.csv
df = pd.read_csv("./data/Hospital_binary.csv")   # <-- ชี้ไปที่ไฟล์ที่คุณอัปโหลด
st.subheader("ข้อมูลตัวอย่างจากชุดข้อมูล")
st.write(df.head(10))

# เลือก features (ให้แก้ตามคอลัมน์ที่มีในไฟล์)
features = df.columns[:-1]   # เลือกทุกคอลัมน์ ยกเว้นคอลัมน์สุดท้าย
target = df.columns[-1]      # สมมติคอลัมน์สุดท้ายคือ target

X = df[features]
y = df[target]

# แบ่ง train/test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=200)

# สร้างโมเดล
ModelDtree = DecisionTreeClassifier()
dtree = ModelDtree.fit(x_train, y_train)

# แสดงกล่องให้ใส่ค่า (ตาม features จริง)
st.subheader("กรุณาป้อนข้อมูลเพื่อพยากรณ์")

user_input = []
for col in features:
    val = st.number_input(f'กรอกค่า {col}', value=float(X[col].mean()))
    user_input.append(val)

if st.button("พยากรณ์"):
    x_input = [user_input]
    y_predict2 = dtree.predict(x_input)
    st.write("ผลการพยากรณ์:", y_predict2[0])

# คำนวณ Accuracy
y_predict = dtree.predict(x_test)
score = accuracy_score(y_test, y_predict)
st.write(f'ความแม่นยำในการพยากรณ์: {score*100:.2f}%')

# แสดงโครงสร้างต้นไม้
fig, ax = plt.subplots(figsize=(12, 8))
tree.plot_tree(dtree, feature_names=features, class_names=True, filled=True, ax=ax)
st.pyplot(fig)
