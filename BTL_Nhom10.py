import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ file CSV
data = pd.read_csv('C:\\Users\\nguye\\OneDrive\\Documents\\FileMonLapTrinh\\Mon_PyThon\Hoa_May_CNTT\\heart-disease.csv')

# Chia dữ liệu thành features (X) và target (y)
X = data.drop('target', axis=1)
y = data['target']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Logistic Regression
best_model = LogisticRegression()
best_model.fit(X_train, y_train)

# Huấn luyện mô hình SVM
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
lr_predictions = best_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)

lr_precision = precision_score(y_test, lr_predictions)
svm_precision = precision_score(y_test, svm_predictions)

lr_recall = recall_score(y_test, lr_predictions)
svm_recall = recall_score(y_test, svm_predictions)

lr_f1_score = f1_score(y_test, lr_predictions)
svm_f1_score = f1_score(y_test, svm_predictions)

# Tạo giao diện đồ họa
window = tk.Tk()
window.title("Dự đoán bệnh tim")

feature_names = X.columns.tolist()

# Định nghĩa hàm dự đoán bệnh tim sử dụng mô hình Logistic Regression
def predict_Heart_Disease_LR(model, feature_names, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    input_features = pd.DataFrame(input_features, columns=feature_names)
    prediction = model.predict(input_features)[0]
    return prediction

# Định nghĩa hàm dự đoán bệnh tim sử dụng mô hình SVM
def predict_Heart_Disease_SVM(model, feature_names, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    input_features = pd.DataFrame(input_features, columns=feature_names)
    prediction = model.predict(input_features)[0]
    return prediction

# Thêm hàm xử lý sự kiện khi nhấn nút Dự đoán sử dụng mô hình Logistic Regression
def handle_predict_button_lr():
    age = age_entry.get()
    sex = sex_entry.get()
    cp = cp_entry.get()
    trestbps = trestbps_entry.get()
    chol = chol_entry.get()
    fbs = fbs_entry.get()
    restecg = restecg_entry.get()
    thalach = thalach_entry.get()
    exang = exang_entry.get()
    oldpeak = oldpeak_entry.get()
    slope = slope_entry.get()
    ca = ca_entry.get()
    thal = thal_entry.get()

    prediction = predict_Heart_Disease_LR(best_model, feature_names, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

    result_label.config(text=f"Kết quả dự đoán (Logistic Regression): {prediction}")

    # Tính toán độ chính xác của mô hình Logistic Regression
    accuracy_label.config(text=f"Độ chính xác (Logistic Regression): {lr_accuracy}")
    precision_label.config(text=f"Precision (Logistic Regression): {lr_precision}")
    recall_label.config(text=f"Recall (Logistic Regression): {lr_recall}")
    f1_score_label.config(text=f"F1-Score (Logistic Regression): {lr_f1_score}")

# Thêm hàm xử lý sự kiện khi nhấn nút Dự đoán sử dụng mô hình SVM
def handle_predict_button_svm():
    age = age_entry.get()
    sex = sex_entry.get()
    cp = cp_entry.get()
    trestbps = trestbps_entry.get()
    chol = chol_entry.get()
    fbs = fbs_entry.get()
    restecg = restecg_entry.get()
    thalach = thalach_entry.get()
    exang = exang_entry.get()
    oldpeak = oldpeak_entry.get()
    slope = slope_entry.get()
    ca = ca_entry.get()
    thal = thal_entry.get()

    prediction = predict_Heart_Disease_SVM(svm_model, feature_names, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

    result_label.config(text=f"Kết quả dự đoán (SVM): {prediction}")

    # Tính toán độ chính xác của mô hình SVM
    accuracy_label.config(text=f"Độ chính xác (SVM): {svm_accuracy}")
    precision_label.config(text=f"Precision (SVM): {svm_precision}")
    recall_label.config(text=f"Recall (SVM): {svm_recall}")
    f1_score_label.config(text=f"F1-Score (SVM): {svm_f1_score}")

# Tạo các thành phần giao diện
age_label = tk.Label(window, text="Tuổi:")
age_label.pack()
age_entry = tk.Entry(window)
age_entry.pack()

sex_label = tk.Label(window, text="Giới tính (0: Nữ, 1: Nam):")
sex_label.pack()
sex_entry = tk.Entry(window)
sex_entry.pack()
cp_label = tk.Label(window, text="Loại đau ngực (0-3):")
cp_label.pack()
cp_entry = tk.Entry(window)
cp_entry.pack()

trestbps_label = tk.Label(window, text="Huyết áp tĩnh (mm Hg):")
trestbps_label.pack()
trestbps_entry = tk.Entry(window)
trestbps_entry.pack()

# Tạo các nhãn và ô nhập liệu cho các thuộc tính khác
chol_label = tk.Label(window, text="Cholesterol (mg/dL): ")
chol_label.pack()
chol_entry = tk.Entry(window)
chol_entry.pack()

fbs_label = tk.Label(window, text="Đường huyết nhanh (> 120 mg/dL) (0 nếu không, 1 nếu có):  ")
fbs_label.pack()
fbs_entry = tk.Entry(window)
fbs_entry.pack()

restecg_label = tk.Label(window, text="Kết quả điện tâm đồ (0-2): ")
restecg_label.pack()
restecg_entry = tk.Entry(window)
restecg_entry.pack()

thalach_label = tk.Label(window, text="Nhịp tim tối đa đạt được:  ")
thalach_label.pack()
thalach_entry = tk.Entry(window)
thalach_entry.pack()

exang_label = tk.Label(window, text="Tình trạng tập thể dục gây đau thắt ngực (0 nếu không, 1 nếu có):  ")
exang_label.pack()
exang_entry = tk.Entry(window)
exang_entry.pack()

oldpeak_label = tk.Label(window, text="ST chênh xuống do gắng sức so với nghỉ ngơi: ")
oldpeak_label.pack()
oldpeak_entry = tk.Entry(window)
oldpeak_entry.pack()

slope_label = tk.Label(window, text="Độ dốc đỉnh tập thể dục (0-2):  ")
slope_label.pack()
slope_entry = tk.Entry(window)
slope_entry.pack()

ca_label = tk.Label(window, text="Số mạch cận lân (0-3): ")
ca_label.pack()
ca_entry = tk.Entry(window)
ca_entry.pack()

thal_label = tk.Label(window, text="Thalassemia (0-3): ")
thal_label.pack()
thal_entry = tk.Entry(window)
thal_entry.pack()

# ... Tiếp tục tạo các thành phần giao diện cho các thuộc tính khác

predict_button_lr = tk.Button(window, text="Dự đoán (Logistic Regression)", command=handle_predict_button_lr)
predict_button_lr.pack()

predict_button_svm = tk.Button(window, text="Dự đoán (SVM)", command=handle_predict_button_svm)
predict_button_svm.pack()

result_label = tk.Label(window, text="")
result_label.pack()

accuracy_label = tk.Label(window, text="")
accuracy_label.pack()

precision_label = tk.Label(window, text="")
precision_label.pack()

recall_label = tk.Label(window, text="")
recall_label.pack()

f1_score_label = tk.Label(window, text="")
f1_score_label.pack()

window.mainloop()