import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Chuẩn bị dữ liệu IRIS
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 1. KNN tự cài đặt
class KNN:
    def __init__(self, k=5):
        self.k = k  # Số lượng láng giềng gần nhất

    def fit(self, X, y):
        self.X_train = X  # Lưu lại dữ liệu huấn luyện
        self.y_train = y  # Lưu lại nhãn tương ứng

    def predict(self, X):
        predictions = []
        for x in X:
            # Tính khoảng cách từ điểm x đến tất cả các điểm huấn luyện
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # Chọn k láng giềng gần nhất
            k_indices = distances.argsort()[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            # Gán nhãn theo nhãn xuất hiện nhiều nhất
            predictions.append(np.bincount(k_nearest_labels).argmax())
        return np.array(predictions)


# 2. SVM tự cài đặt (linear kernel đơn giản)
class SVM:
    def __init__(self, learning_rate=0.001, n_iters=1000, lambda_param=0.01):
        self.lr = learning_rate  # Tốc độ học
        self.n_iters = n_iters  # Số lần lặp
        self.lambda_param = lambda_param  # Tham số điều chuẩn
        self.w = None  # Trọng số
        self.b = 0  # Bias

    def fit(self, X, y):
        # Đổi nhãn y thành +1 và -1
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(X.shape[1])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Kiểm tra điều kiện margin (biên)
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # Cập nhật nếu đúng điều kiện
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Điều chỉnh trọng số và bias nếu không đạt điều kiện
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        # Tính giá trị dự đoán
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


# 3. MLPClassifier (Neural Network) tự cài đặt
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.lr = learning_rate  # Tốc độ học
        # Khởi tạo trọng số và bias cho lớp ẩn
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        # Khởi tạo trọng số và bias cho lớp đầu ra
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        # Hàm sigmoid dùng để kích hoạt
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Đạo hàm sigmoid để tính gradient
        return x * (1 - x)

    def fit(self, X, y, n_epochs=1000):
        # Chuyển đổi y thành one-hot encoding
        y_one_hot = np.zeros((y.size, y.max() + 1))
        y_one_hot[np.arange(y.size), y] = 1

        for _ in range(n_epochs):
            # Forward pass
            hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_output = self.sigmoid(hidden_input)
            final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
            y_pred = self.sigmoid(final_input)

            # Tính lỗi và backpropagation
            output_error = y_one_hot - y_pred
            d_output = output_error * self.sigmoid_derivative(y_pred)
            hidden_error = d_output.dot(self.weights_hidden_output.T)
            d_hidden = hidden_error * self.sigmoid_derivative(hidden_output)

            # Cập nhật trọng số và bias
            self.weights_hidden_output += hidden_output.T.dot(d_output) * self.lr
            self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.lr
            self.weights_input_hidden += X.T.dot(d_hidden) * self.lr
            self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.lr

    def predict(self, X):
        # Forward pass để dự đoán
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        y_pred = self.sigmoid(final_input)
        # Lấy nhãn có xác suất cao nhất
        return np.argmax(y_pred, axis=1)


# Đánh giá từng mô hình
models = {
    "KNN": KNN(k=5),
    "SVM": SVM(learning_rate=0.001, n_iters=1000),
    "ANN": NeuralNetwork(input_size=X_train.shape[1], hidden_size=5, output_size=3)
}

results = {}
for model_name, model in models.items():
    # Huấn luyện mô hình và đo thời gian
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Dự đoán và đo thời gian
    start_time = time.time()
    predictions = model.predict(X_test)
    test_time = time.time() - start_time

    # Tính toán độ chính xác
    accuracy = accuracy_score(y_test, predictions)
    results[model_name] = {
        "accuracy": accuracy,
        "train_time": train_time,
        "test_time": test_time
    }

# Hiển thị kết quả
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"  Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"  Training Time: {metrics['train_time']:.4f}s")
    print(f"  Testing Time: {metrics['test_time']:.4f}s")
    print()
