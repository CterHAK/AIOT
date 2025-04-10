# ĐỒ ÁN GIỮA KỲ: XÂY DỰNG MÔ HÌNH CVAE CHO TỔNG HỢP TÍN HIỆU PPG

## 1. TỔNG QUAN ĐỒ ÁN

### 1.1. Mô tả bài toán
Tín hiệu Photoplethysmography (PPG) là một kỹ thuật đo lường quang học không xâm lấn, được sử dụng để ghi nhận sự thay đổi thể tích máu trong các mạch máu ngoại vi. PPG thường được ứng dụng trong việc theo dõi các chỉ số sinh lý quan trọng như nhịp tim (Heart Rate - HR) và nhịp thở (Breathing Rate - BR). 

Trong đồ án này, sinh viên sẽ xây dựng mô hình Conditional Variational Autoencoder (CVAE) để tổng hợp tín hiệu PPG dựa trên các thông số sinh lý cho trước (nhịp tim và tốc độ hô hấp). Dữ liệu đã được tiền xử lý thành các đoạn tín hiệu dài 1250 điểm (khoảng 5 giây).

### 1.2. Mục tiêu
- Hiểu nguyên lý hoạt động của mô hình CVAE
- Xây dựng và huấn luyện mô hình CVAE để tổng hợp tín hiệu PPG
- Nâng cao: Xây dựng mô hình CNN-CVAE cải tiến
- Đánh giá hiệu quả mô hình thông qua các chỉ số định lượng và định tính

### 1.3. Ứng dụng
- Mở rộng tập dữ liệu huấn luyện cho các mô hình học máy trong lĩnh vực y tế
- Mô phỏng tín hiệu sinh lý phục vụ nghiên cứu hoặc kiểm tra thiết bị y tế
- Tạo dữ liệu tổng hợp cho các ứng dụng theo dõi sức khỏe

## 2. YÊU CẦU VỀ TỔ CHỨC CODE

### 2.1. Nguyên tắc tổ chức notebook
- Notebook phải được tổ chức theo các section rõ ràng, có tiêu đề và mô tả cho từng phần
- Các cell phải được chạy theo thứ tự từ trên xuống dưới và không có lỗi
- Mỗi cell nên thực hiện một chức năng cụ thể và có comment đầy đủ
- Kết quả của mỗi bước xử lý cần được hiển thị ngay sau cell tương ứng

### 2.2. Kiểm tra kết quả trước khi lưu
- **Nguyên tắc quan trọng**: Luôn hiển thị và kiểm tra kết quả trước khi lưu thành file hoặc hình ảnh
- Tổ chức thành 2 cell riêng biệt:
  - Cell 1: Thực hiện xử lý và hiển thị kết quả để kiểm tra
  - Cell 2: Lưu kết quả đã kiểm tra thành file hoặc hình ảnh

Ví dụ:
```python
# Cell 1: Tạo và hiển thị biểu đồ để kiểm tra
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()  # Hiển thị để kiểm tra trước

# Cell 2: Sau khi đã kiểm tra và kết quả OK, lưu biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model_loss.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 2.3. Hiển thị thông tin chi tiết
- Hiển thị thông tin về cấu trúc dữ liệu sau mỗi bước xử lý (shape, type, min/max values)
- In ra các thông số chính của mô hình sau khi định nghĩa (summary)
- Hiển thị các metrics trong quá trình huấn luyện để theo dõi
- Visualize các kết quả trung gian để kiểm tra tính đúng đắn

### 2.4. Xử lý lỗi
- Thêm các khối try-except để xử lý các lỗi có thể xảy ra
- In ra thông báo lỗi rõ ràng giúp dễ dàng debug
- Kiểm tra các điều kiện đầu vào trước khi thực hiện các phép tính phức tạp

## 3. DỮ LIỆU

### 3.1. Bộ dữ liệu
- Dữ liệu huấn luyện: 3724 đoạn tín hiệu PPG
- Dữ liệu kiểm thử: 931 đoạn tín hiệu PPG
- Mỗi đoạn tín hiệu có độ dài 1250 điểm
- Thông số điều kiện: HR (nhịp tim) và BR (tốc độ hô hấp)

### 3.2. Tiền xử lý dữ liệu

#### 3.2.1. Chuẩn hóa tín hiệu PPG
Tín hiệu PPG cần được chuẩn hóa về khoảng [-1, 1] để phù hợp với hàm kích hoạt tanh ở đầu ra của mô hình CVAE. Sử dụng phương pháp Min-Max Normalization:

```python
# Bước 3: Tiền xử lý tín hiệu PPG
fs = 125  # Tần số lấy mẫu (Hz)
segment_length = int(fs * 10)  # Độ dài đoạn tín hiệu: 10 giây (1250 mẫu)

# Thiết kế bộ lọc Butterworth bandpass (0.1-8 Hz)
nyquist = fs / 2
lowcut = 0.1
highcut = 8.0
low_cutoff = lowcut / nyquist
high_cutoff = highcut / nyquist
b, a = signal.butter(2, [low_cutoff, high_cutoff], btype='band')  # order=2

# Bộ chuẩn hóa tín hiệu PPG về [0, 1]
ppg_scaler = MinMaxScaler(feature_range=(0, 1))

# Bước 5: Phân đoạn tín hiệu PPG
print("Đang phân đoạn tín hiệu PPG...")
step = int(segment_length * (1 - 0.5))  # Bước trượt: 50% overlap
num_segments = 0

for j, record in enumerate(valid_records):
    try:
        ppg = record['ppg']
        hr_values = record['hr']
        rr_values = record['rr']
        ppg_fs = record['ppg_fs']

        if len(ppg) == 0 or len(hr_values) == 0 or len(rr_values) == 0:
            print(f"Bản ghi {j} thiếu dữ liệu, bỏ qua")
            continue

        # Lọc tín hiệu PPG
        ppg_filtered = signal.filtfilt(b, a, ppg)

        # Chuẩn hóa tín hiệu PPG về [0, 1]
        ppg_normalized = ppg_scaler.fit_transform(ppg_filtered.reshape(-1, 1)).flatten()

        # Chuẩn hóa HR và RR cho bản ghi này
        hr_values = hr_values.reshape(-1, 1)
        rr_values = rr_values.reshape(-1, 1)
        hr_norm = hr_scaler.transform(hr_values).flatten()
        rr_norm = rr_scaler.transform(rr_values).flatten()
        hr_norm = np.nan_to_num(hr_norm, nan=0.0)
        rr_norm = np.nan_to_num(rr_norm, nan=0.0)

        start = 0
        end = start + segment_length

        while end <= len(ppg):
            # Trích xuất đoạn tín hiệu PPG
            ppg_seg = ppg_normalized[start:end]
            if len(ppg_seg) != segment_length:
                break  # Bỏ qua nếu đoạn không đủ dài

            # Trích xuất HR và RR tương ứng
            time_start = start / ppg_fs  # Thời gian bắt đầu (giây)
            time_end = end / ppg_fs  # Thời gian kết thúc (giây)
            hr_indices = np.where((np.arange(len(hr_values)) * (1.0 / ppg_fs) >= time_start) & (np.arange(len(hr_values)) * (1.0 / ppg_fs) < time_end))[0]
            rr_indices = np.where((np.arange(len(rr_values)) * (1.0 / ppg_fs) >= time_start) & (np.arange(len(rr_values)) * (1.0 / ppg_fs) < time_end))[0]

            hr_seg = np.mean(hr_norm[hr_indices]) if len(hr_indices) > 0 else hr_norm[0]
            rr_seg = np.mean(rr_norm[rr_indices]) if len(rr_indices) > 0 else rr_norm[0]

            ppg_segments.append(ppg_seg)
            hr_segments.append(hr_seg)
            rr_segments.append(rr_seg)

            start += step
            end += step
            num_segments += 1

        print(f"Đã xử lý bản ghi {j}")

    except Exception as e:
        print(f"Lỗi khi phân đoạn tín hiệu PPG cho bản ghi {j}: {e}")

print(f"Số đoạn tín hiệu: {num_segments}")

# Kiểm tra nếu không có đoạn tín hiệu nào được tạo
if num_segments == 0:
    print("Không có đủ dữ liệu để tiếp tục. Sử dụng dữ liệu giả lập để minh họa.")
    num_samples = 100
    ppg_segments = np.random.rand(num_samples, segment_length)  # Giả lập tín hiệu PPG từ [0, 1]
    hr_segments = np.random.uniform(0, 1, num_samples)  # HR đã chuẩn hóa
    rr_segments = np.random.uniform(0, 1, num_samples)  # RR đã chuẩn hóa
    print(f"Đã tạo {num_samples} mẫu dữ liệu giả lập.")


```

#### 3.2.2. Chuẩn hóa các thông số điều kiện (HR và BR)
Đối với các thông số HR và BR, sinh viên phải sử dụng phương pháp Z-score normalization cho toàn bộ dữ liệu:

```python
# Bước 2: Chuẩn hóa HR và RR về [0, 1] bằng MinMaxScaler
hr_scaler = MinMaxScaler(feature_range=(0, 1))
rr_scaler = MinMaxScaler(feature_range=(0, 1))

# Reshape dữ liệu để phù hợp với MinMaxScaler
all_hr_values = all_hr_values.reshape(-1, 1)
all_rr_values = all_rr_values.reshape(-1, 1)

# Chuẩn hóa
hr_normalized = hr_scaler.fit_transform(all_hr_values).flatten()
rr_normalized = rr_scaler.fit_transform(all_rr_values).flatten()

# Xử lý giá trị NaN (nếu có)
hr_normalized = np.nan_to_num(hr_normalized, nan=0.0)
rr_normalized = np.nan_to_num(rr_normalized, nan=0.0)

# Thống kê sau khi chuẩn hóa HR và RR
print("\nThống kê sau khi chuẩn hóa HR và RR:")
print(f"HR (normalized): mean = {np.mean(hr_normalized):.4f}, std = {np.std(hr_normalized):.4f}, min = {np.min(hr_normalized):.4f}, max = {np.max(hr_normalized):.4f}")
print(f"RR (normalized): mean = {np.mean(rr_normalized):.4f}, std = {np.std(rr_normalized):.4f}, min = {np.min(rr_normalized):.4f}, max = {np.max(rr_normalized):.4f}")

# Vẽ biểu đồ phân bố của HR và RR (sau khi chuẩn hóa) - giống với biểu đồ của thầy bạn
plt.figure(figsize=(15, 5))

# Biểu đồ phân bố của HR (chuẩn hóa)
plt.subplot(1, 2, 1)
counts, bins, _ = plt.hist(hr_normalized, bins=20, alpha=0.7, density=True, color='blue')
plt.axvline(x=np.mean(hr_normalized), color='red', linestyle='--', label='Mean')
plt.title('Heart Rate Distribution (MinMax [0, 1])')
plt.xlabel('HR (normalized)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.legend()

# Biểu đồ phân bố của RR (chuẩn hóa)
plt.subplot(1, 2, 2)
counts, bins, _ = plt.hist(rr_normalized, bins=20, alpha=0.7, density=True, color='green')
plt.axvline(x=np.mean(rr_normalized), color='red', linestyle='--', label='Mean')
plt.title('Respiratory Rate Distribution (MinMax [0, 1])')
plt.xlabel('RR (normalized)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
```

## 4. YÊU CẦU CƠ BẢN: XÂY DỰNG MÔ HÌNH CVAE GỐC

### 4.1. Định nghĩa lớp Sampling

```python
# Định nghĩa lớp Sampling
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

### 4.2. Xây dựng Encoder

```python
# Encoder cải tiến
        encoder_inputs = layers.Input(shape=(input_dim,), name='encoder_input')
        condition_inputs = layers.Input(shape=(condition_dim,), name='condition_input')
        x = layers.Concatenate()([encoder_inputs, condition_inputs])
        for units in hidden_units:
            x = layers.Dense(units, kernel_initializer='glorot_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(negative_slope=0.2)(x)
            x = layers.Dropout(0.2)(x)
        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = Model([encoder_inputs, condition_inputs], [z_mean, z_log_var, z], name='encoder')
```

### 4.3. Xây dựng Decoder

```python
# Decoder cải tiến
        latent_inputs = layers.Input(shape=(latent_dim,), name='latent_input')
        condition_inputs_decoder = layers.Input(shape=(condition_dim,), name='condition_input_decoder')
        x = layers.Concatenate()([latent_inputs, condition_inputs_decoder])
        for units in reversed(hidden_units):
            x = layers.Dense(units, kernel_initializer='glorot_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(negative_slope=0.2)(x)
            x = layers.Dropout(0.2)(x)
        decoder_outputs = layers.Dense(input_dim, activation='sigmoid')(x)
        self.decoder = Model([latent_inputs, condition_inputs_decoder], decoder_outputs, name='decoder')
```

### 4.4. Tạo mô hình CVAE hoàn chỉnh

```python
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs):
        x, condition = inputs
        z_mean, z_log_var, z = self.encoder([x, condition])
        return self.decoder([z, condition])

    def train_step(self, data):
        x, condition = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([x, condition])
            reconstruction = self.decoder([z, condition])
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(x, reconstruction)) * self.input_dim
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = reconstruction_loss + 0.5 * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    def test_step(self, data):
        x, condition = data
        z_mean, z_log_var, z = self.encoder([x, condition])
        reconstruction = self.decoder([z, condition])
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(x, reconstruction)) * self.input_dim
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        total_loss = reconstruction_loss + 0.5 * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    def generate(self, condition):
        condition = tf.convert_to_tensor(condition, dtype=tf.float32)
        batch_size = tf.shape(condition)[0]
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        return self.decoder([z, condition])
```

### 4.5. Callback để ghi lại learning rate

```python
# Callback để ghi lại learning rate
class LrHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LrHistory, self).__init__()
        self.lr_history = []

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.lr_history.append(lr)

lr_history_callback = LrHistory()
```

### 4.6. Huấn luyện mô hình

```python
# Huấn luyện mô hình
print("\nBắt đầu huấn luyện mô hình Standard CVAE...")
start_time = time.time()
history = cvae.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset,
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback, lr_history_callback],
    verbose=1
)
training_time = time.time() - start_time
print(f"\nHuấn luyện Standard CVAE hoàn tất trong {training_time:.2f} giây.")

# Lưu mô hình
cvae.save_weights(os.path.join(model_path, 'cvae_standard_final.weights.h5'))
print(f"Đã lưu mô hình Standard CVAE tại: {os.path.join(model_path, 'cvae_standard_final.weights.h5')}")
```

### 4.7. Tạo dữ liệu mới từ mô hình đã huấn luyện

```python
def generate(self, condition):
        condition = tf.convert_to_tensor(condition, dtype=tf.float32)
        batch_size = tf.shape(condition)[0]
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        return self.decoder([z, condition])
 Tải mô hình đã huấn luyện
print("Đang tải mô hình Standard CVAE đã huấn luyện...")
cvae = StandardCVAE(input_dim, condition_dim, latent_dim, hidden_units)
cvae.build(input_shape=[(None, input_dim), (None, condition_dim)])
cvae.load_weights(os.path.join(model_path, 'cvae_standard_final.weights.h5'))

# Tạo tín hiệu PPG và không gian tiềm ẩn
num_samples = 20
test_indices = np.random.choice(len(X_test), num_samples, replace=False)
test_conditions = tf.gather(condition_test, test_indices)
original_ppg = X_test[test_indices]
generated_ppg = cvae.generate(test_conditions).numpy()
original_ppg_tensor = tf.convert_to_tensor(original_ppg, dtype=tf.float32)
z_mean, z_log_var, z = cvae.encode([original_ppg_tensor, test_conditions])

# Tải kết quả phân tích Fourier (giả định file đã được tạo trước)
print("Đang tải kết quả phân tích Fourier...")
fourier_results_path = os.path.join(results_path, 'frequency_analysis_results_v2.csv')
if os.path.exists(fourier_results_path):
    fourier_results = pd.read_csv(fourier_results_path)
    print(f"Đã tải kết quả phân tích Fourier: {len(fourier_results)} mẫu")
else:
    print("Không tìm thấy kết quả phân tích Fourier! Vui lòng tạo file trước.")

# Hàm phân tích phổ tần số
def analyze_frequency_spectrum(signal, fs):
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)[:n//2]
    yf_abs = 2.0/n * np.abs(yf[0:n//2])
    return xf, yf_abs

```

## 5. YÊU CẦU NÂNG CAO: XÂY DỰNG MÔ HÌNH CNN-CVAE

### 5.1. Chuẩn bị dữ liệu cho DNN

```python
# Chuẩn hóa dữ liệu kiểm thử (khớp với train)
X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))
condition_test = tf.convert_to_tensor(np.column_stack((hr_test, rr_test)), dtype=tf.float32)
condition_test = (condition_test - tf.reduce_min(condition_test, axis=0)) / (tf.reduce_max(condition_test, axis=0) - tf.reduce_min(condition_test, axis=0))

# Giải chuẩn hóa HR và RR về giá trị thực tế
hr_min, hr_max = np.min(hr_train), np.max(hr_train)
rr_min, rr_max = np.min(rr_train), np.max(rr_train)
hr_test_real = hr_test * (hr_max - hr_min) + hr_min
rr_test_real = rr_test * (rr_max - rr_min) + rr_min
```

### 5.2. Xây dựng Encoder

```python
encoder_inputs = layers.Input(shape=(input_dim,), name='encoder_input')
        condition_inputs = layers.Input(shape=(condition_dim,), name='condition_input')
        x = layers.Concatenate()([encoder_inputs, condition_inputs])
        for units in hidden_units:
            x = layers.Dense(units, kernel_initializer='glorot_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(negative_slope=0.2)(x)
            x = layers.Dropout(0.2)(x)
        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = Model([encoder_inputs, condition_inputs], [z_mean, z_log_var, z], name='encoder')
```

### 5.3. Xây dựng CNN Decoder

```python
# Decoder
        latent_inputs = layers.Input(shape=(latent_dim,), name='latent_input')
        condition_inputs_decoder = layers.Input(shape=(condition_dim,), name='condition_input_decoder')
        x = layers.Concatenate()([latent_inputs, condition_inputs_decoder])
        for units in reversed(hidden_units):
            x = layers.Dense(units, kernel_initializer='glorot_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(negative_slope=0.2)(x)
            x = layers.Dropout(0.2)(x)
        decoder_outputs = layers.Dense(input_dim, activation='sigmoid')(x)
        self.decoder = Model([latent_inputs, condition_inputs_decoder], decoder_outputs, name='decoder')
```

### 5.4. Tạo mô hình CVAE hoàn chỉnh và hàm mất mát

```python
# Xây dựng mô hình với kích thước đầu vào
cvae.build(input_shape=[(None, input_dim), (None, condition_dim)])
cvae.compile(optimizer=Adam(learning_rate=lr_schedule))

# Tạo dataset với shuffle
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, condition_train)).shuffle(buffer_size=1024).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, condition_test)).batch(batch_size)

# Callbacks
log_dir = os.path.join(model_path, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Sửa checkpoint path để tránh xung đột
checkpoint_path = os.path.join(model_path, f"cvae_standard_checkpoint_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.weights.h5")
old_checkpoint = os.path.join(model_path, "cvae_standard_checkpoint.weights.h5")
if os.path.exists(old_checkpoint):
    backup_path = old_checkpoint + f".backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.move(old_checkpoint, backup_path)
    print(f"Đã đổi tên tệp checkpoint cũ thành: {backup_path}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True
)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=20,
    restore_best_weights=True
)

# Callback để ghi lại learning rate
class LrHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LrHistory, self).__init__()
        self.lr_history = []

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.lr_history.append(lr)

lr_history_callback = LrHistory()

# Huấn luyện mô hình
print("\nBắt đầu huấn luyện mô hình Standard CVAE...")
start_time = time.time()
history = cvae.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset,
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback, lr_history_callback],
    verbose=1
)
training_time = time.time() - start_time
print(f"\nHuấn luyện Standard CVAE hoàn tất trong {training_time:.2f} giây.")

# Lưu mô hình
cvae.save_weights(os.path.join(model_path, 'cvae_standard_final.weights.h5'))
print(f"Đã lưu mô hình Standard CVAE tại: {os.path.join(model_path, 'cvae_standard_final.weights.h5')}")
```

### 5.5. Huấn luyện mô hình CNN-CVAE

```python
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True
)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=20,
    restore_best_weights=True
)

```

### 5.6. So sánh hiệu suất giữa hai mô hình

```python
# Đánh giá và vẽ biểu đồ
def evaluate_and_plot(cvae, X_test, condition_test, figures_path, model_type='standard', fs=125):
    ppg_generated = cvae.generate(condition_test).numpy()
    mse = np.mean(np.square(X_test - ppg_generated))
    hr_extracted = [len(find_peaks(ppg, distance=fs*0.6)[0]) * (60 / 10) for ppg in ppg_generated]
    hr_mae = np.mean(np.abs(hr_extracted - hr_test))
    br_extracted = []
    for ppg in ppg_generated:
        fft_vals = np.abs(fft(ppg))
        freqs = np.fft.fftfreq(len(ppg), 1/fs)
        br_idx = (freqs > 0.1) & (freqs < 0.5)
        br = freqs[np.argmax(fft_vals[br_idx])] * 60 if np.any(br_idx) else 0
        br_extracted.append(br)
    br_mae = np.mean(np.abs(br_extracted - rr_test))
    print(f"STANDARD - MSE: {mse:.4f}, HR MAE: {hr_mae:.2f} bpm, BR MAE: {br_mae:.2f} breaths/min")
    plt.figure(figsize=(15, 10))
    for i in range(3):
        plt.subplot(3, 2, 2*i+1)
        plt.plot(X_test[i], label=f'Real PPG (HR={hr_test[i]:.1f}, BR={rr_test[i]:.1f})')
        plt.legend()
        plt.subplot(3, 2, 2*i+2)
        plt.plot(ppg_generated[i], label=f'Gen PPG (HR={hr_extracted[i]:.1f}, BR={br_extracted[i]:.1f})', color='orange')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'standard_ppg_comparison.png'))
    plt.show()  # Hiển thị biểu đồ trên output
    return mse, hr_mae, br_mae
```

## 6. ĐÁNH GIÁ MÔ HÌNH

### 6.1. Đánh giá định lượng

```python
# Cell 1: Định nghĩa các hàm trích xuất nhịp tim và nhịp thở từ tín hiệu PPG
# 1. Trực quan hóa tín hiệu PPG gốc và tín hiệu tổng hợp
print("\n1. Trực quan hóa tín hiệu PPG gốc và tín hiệu tổng hợp")
plt.figure(figsize=(15, 20))
for i in range(min(10, num_samples)):
    plt.subplot(10, 2, 2*i+1)
    plt.plot(original_ppg[i], label='Original PPG', color='blue')
    plt.title(f'Sample {i+1}: HR={hr_test_real[test_indices[i]]:.1f}, RR={rr_test_real[test_indices[i]]:.1f}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(10, 2, 2*i+2)
    plt.plot(generated_ppg[i], label='Generated PPG', color='orange', alpha=0.7)
    plt.title(f'Sample {i+1}: HR={hr_test_real[test_indices[i]]:.1f}, RR={rr_test_real[test_indices[i]]:.1f}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'original_vs_generated_comparison_standard.png'))
plt.show()  # Hiển thị biểu đồ trên output

# Cell 2: Hàm đánh giá các mô hình dựa trên MSE và độ chính xác HR, BR
# 2. Trực quan hóa phân bố HR và RR thực tế
print("\n2. Trực quan hóa phân bố HR và RR")
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(hr_test_real, rr_test_real, alpha=0.5)
plt.title('HR vs RR Distribution (Real)')
plt.xlabel('HR (bpm)')
plt.ylabel('RR (breaths/min)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.hist(hr_test_real, bins=20, alpha=0.7)
plt.title('HR Distribution')
plt.xlabel('HR (bpm)')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(rr_test_real, bins=20, alpha=0.7)
plt.title('RR Distribution')
plt.xlabel('RR (breaths/min)')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'hr_rr_distribution_standard.png'))
plt.show()  # Hiển thị biểu đồ trên output

# 2. Trực quan hóa phân bố HR và RR thực tế
print("\n2. Trực quan hóa phân bố HR và RR")
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(hr_test_real, rr_test_real, alpha=0.5)
plt.title('HR vs RR Distribution (Real)')
plt.xlabel('HR (bpm)')
plt.ylabel('RR (breaths/min)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.hist(hr_test_real, bins=20, alpha=0.7)
plt.title('HR Distribution')
plt.xlabel('HR (bpm)')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(rr_test_real, bins=20, alpha=0.7)
plt.title('RR Distribution')
plt.xlabel('RR (breaths/min)')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'hr_rr_distribution_standard.png'))
plt.show()  # Hiển thị biểu đồ trên output

# Cell 3: Đánh giá cả hai mô hình
# 7. Tạo bảng tóm tắt kết quả đánh giá
    print("\n7. Tạo bảng tóm tắt kết quả đánh giá")
    summary_stats = {
        'MSE_Time': fourier_results['MSE_Time'].describe().to_dict(),
        'PSNR': fourier_results['PSNR'].describe().to_dict(),
        'Corr': fourier_results['Corr'].describe().to_dict(),
        'MSE_Freq': fourier_results['MSE_Freq'].describe().to_dict(),
        'HR_Error': (fourier_results['HR_Gen'] - fourier_results['HR_Real']).describe().to_dict(),
        'RR_Error': (fourier_results['RR_Gen'] - fourier_results['RR_Real']).describe().to_dict()
    }
    summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')
    summary_df.to_csv(os.path.join(results_path, 'evaluation_summary_standard.csv'))

    # Lưu báo cáo
    with open(os.path.join(results_path, 'model_evaluation_results_standard.txt'), 'w') as f:
        f.write("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH Standard CVAE\n")
        f.write("===================================\n\n")
        f.write(f"MSE (Time): Mean={summary_stats['MSE_Time']['mean']:.4f}, Std={summary_stats['MSE_Time']['std']:.4f}\n")
        f.write(f"PSNR: Mean={summary_stats['PSNR']['mean']:.2f}, Std={summary_stats['PSNR']['std']:.2f}\n")
        f.write(f"Corr: Mean={summary_stats['Corr']['mean']:.4f}, Std={summary_stats['Corr']['std']:.4f}\n")
        f.write(f"MSE (Freq): Mean={summary_stats['MSE_Freq']['mean']:.4f}, Std={summary_stats['MSE_Freq']['std']:.4f}\n")
        f.write(f"HR Error: Mean={summary_stats['HR_Error']['mean']:.2f}, Std={summary_stats['HR_Error']['std']:.2f}\n")
        f.write(f"RR Error: Mean={summary_stats['RR_Error']['mean']:.2f}, Std={summary_stats['RR_Error']['std']:.2f}\n\n")
        f.write("Nhận xét: Mô hình tái tạo tín hiệu PPG với chất lượng trung bình, cần cải thiện thêm để phản ánh tốt hơn HR/RR thực tế.\n")
else:
    print("Bỏ qua bước 6 và 7 vì không có file kết quả Fourier.")

print(f"Kết quả trực quan hóa đã được lưu tại: {figures_path}")
```

### 6.2. Đánh giá định tính

```python
# Cell 1: Phân tích phổ tần số của tín hiệu PPG thực tế và tạo ra
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def plot_frequency_spectrum(signal, fs=250, title="Frequency Spectrum"):
    """
    Vẽ phổ tần số của tín hiệu.
    
    Parameters:
    -----------
    signal : ndarray
        Tín hiệu cần phân tích
    fs : int
        Tần số lấy mẫu (Hz)
    title : str
        Tiêu đề biểu đồ
    """
    # Thực hiện FFT
    fft_result = fft(signal)
    freq = fftfreq(len(signal), 1/fs)
    
    # Chỉ quan tâm đến nửa đầu và tần số dương
    n = len(signal)
    fft_magnitude = np.abs(fft_result[:n//2])
    pos_freq = freq[:n//2]
    
    plt.plot(pos_freq, fft_magnitude)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim([0, 5])  # Giới hạn tần số hiển thị đến 5 Hz
    
# Lấy một mẫu PPG thực từ tập test
sample_idx = 5
real_ppg = X_test_normalized[sample_idx]

# Tạo tín hiệu PPG sử dụng cả hai mô hình với cùng HR và BR
hr_actual = y_test_normalized[sample_idx, 0] * hr_br_params['std'][0] + hr_br_params['mean'][0]
br_actual = y_test_normalized[sample_idx, 1] * hr_br_params['std'][1] + hr_br_params['mean'][1]

# Tạo tín hiệu
_, cvae_ppg = generate_ppg(hr_actual, br_actual, num_samples=1)
_, cnn_ppg = generate_cnn_ppg(hr_actual, br_actual, num_samples=1)

cvae_ppg_signal = cvae_ppg[0]
cnn_ppg_signal = cnn_ppg[0] if isinstance(cnn_ppg, np.ndarray) and cnn_ppg.ndim > 1 else cnn_ppg

# Vẽ tín hiệu và phổ tần số
plt.figure(figsize=(15, 10))

# Vẽ tín hiệu
plt.subplot(3, 2, 1)
plt.plot(real_ppg)
plt.title(f'Original PPG (HR={hr_actual:.1f}, BR={br_actual:.1f})')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 3)
plt.plot(cvae_ppg_signal)
plt.title('CVAE Generated PPG')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 5)
plt.plot(cnn_ppg_signal)
plt.title('CNN-CVAE Generated PPG')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

# Vẽ phổ tần số
plt.subplot(3, 2, 2)
plot_frequency_spectrum(real_ppg, fs=250, title="Original PPG Frequency Spectrum")

plt.subplot(3, 2, 4)
plot_frequency_spectrum(cvae_ppg_signal, fs=250, title="CVAE PPG Frequency Spectrum")

plt.subplot(3, 2, 6)
plot_frequency_spectrum(cnn_ppg_signal, fs=250, title="CNN-CVAE PPG Frequency Spectrum")

plt.tight_layout()
plt.show()

# Cell 2: Sau khi kiểm tra, lưu biểu đồ
plt.figure(figsize=(15, 10))

# Vẽ tín hiệu
plt.subplot(3, 2, 1)
plt.plot(real_ppg)
plt.title(f'Original PPG (HR={hr_actual:.1f}, BR={br_actual:.1f})')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 3)
plt.plot(cvae_ppg_signal)
plt.title('CVAE Generated PPG')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 5)
plt.plot(cnn_ppg_signal)
plt.title('CNN-CVAE Generated PPG')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

# Vẽ phổ tần số
plt.subplot(3, 2, 2)
plot_frequency_spectrum(real_ppg, fs=250, title="Original PPG Frequency Spectrum")

plt.subplot(3, 2, 4)
plot_frequency_spectrum(cvae_ppg_signal, fs=250, title="CVAE PPG Frequency Spectrum")

plt.subplot(3, 2, 6)
plot_frequency_spectrum(cnn_ppg_signal, fs=250, title="CNN-CVAE PPG Frequency Spectrum")

plt.tight_layout()
plt.savefig('models/signal_frequency_analysis.png', dpi=300, bbox_inches='tight')
print("Signal frequency analysis saved to 'models/signal_frequency_analysis.png'")

# Cell 3: Kiểm tra tính nhất quán của các tín hiệu được tạo ra
def plot_consistency_check(hr, rr, model_type='CVAE', num_samples=5):
    """
    Tạo và vẽ nhiều mẫu tín hiệu PPG từ cùng một bộ thông số HR, BR.
    """
    plt.figure(figsize=(12, 10))
    
    for i in range(num_samples):
        if model_type == 'CVAE':
            _, ppg = generate_ppg(hr, rr, num_samples=1)
            ppg_signal = ppg[0]
        else:  # CNN-CVAE
            _, ppg = generate_cnn_ppg(hr, rr, num_samples=1)
            ppg_signal = ppg[0] if isinstance(ppg, np.ndarray) and ppg.ndim > 1 else ppg
            
        plt.subplot(num_samples, 1, i+1)
        plt.plot(ppg_signal)
        plt.title(f'{model_type} Sample #{i+1} (HR={hr}, RR={rr})')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()

# Kiểm tra CVAE
plot_consistency_check(hr=75, rr=15, model_type='CVAE', num_samples=5)

# Kiểm tra CNN-CVAE
plot_consistency_check(hr=75, rr=15, model_type='CNN-CVAE', num_samples=5)

# Cell 4: Lưu kết quả kiểm tra tính nhất quán
# CVAE
plt.figure(figsize=(12, 10))
for i in range(5):
    _, ppg = generate_ppg(75, 15, num_samples=1)
    ppg_signal = ppg[0]
    
    plt.subplot(5, 1, i+1)
    plt.plot(ppg_signal)
    plt.title(f'CVAE Sample #{i+1} (HR=75, RR=15)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig('models/cvae_consistency_check.png', dpi=300, bbox_inches='tight')
print("CVAE consistency check saved to 'models/cvae_consistency_check.png'")

# CNN-CVAE
plt.figure(figsize=(12, 10))
for i in range(5):
    _, ppg = generate_cnn_ppg(75, 15, num_samples=1)
    ppg_signal = ppg[0] if isinstance(ppg, np.ndarray) and ppg.ndim > 1 else ppg
    
    plt.subplot(5, 1, i+1)
    plt.plot(ppg_signal)
    plt.title(f'CNN-CVAE Sample #{i+1} (HR=75, RR=15)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig('models/cnn_cvae_consistency_check.png', dpi=300, bbox_inches='tight')
print("CNN-CVAE consistency check saved to 'models/cnn_cvae_consistency_check.png'")
```

## 7. TIÊU CHÍ ĐÁNH GIÁ

### 7.1. Mô hình CVAE gốc (70%)
- Cài đặt đúng cấu trúc encoder và decoder (20%)
- Định nghĩa và áp dụng hàm mất mát phù hợp (15%)
- Tiền xử lý dữ liệu đúng và chuẩn hóa về khoảng [-1, 1] cho PPG và z-score cho HR, BR (15%)
- Huấn luyện mô hình đến khi hội tụ (10%)
- Tạo được tín hiệu PPG mới từ các thông số HR, BR cho trước (10%)

### 7.2. Mô hình CNN-CVAE (30%)
- Cài đặt đúng cấu trúc CNN encoder và decoder (15%)
- Huấn luyện mô hình đến khi hội tụ (10%)
- So sánh hiệu suất giữa hai mô hình (5%)

## 8. YÊU CẦU NỘP BÀI

1. **File notebook** (Jupyter/Colab) chứa toàn bộ mã nguồn có chú thích:
   - Code phải được tổ chức theo các section rõ ràng
   - Mỗi phần xử lý phải hiển thị kết quả để kiểm tra trước khi lưu
   - Sử dụng cell riêng biệt để kiểm tra kết quả và lưu file/hình ảnh

2. **Báo cáo** ngắn (1-2 trang) mô tả:
   - Phương pháp tiền xử lý dữ liệu
   - Kiến trúc mô hình CVAE và CNN-CVAE
   - Kết quả đánh giá và so sánh giữa hai mô hình
   - Ưu nhược điểm của từng phương pháp

3. **File mô hình đã huấn luyện** (.h5) cho CVAE gốc và CNN-CVAE

## 9. TÀI NGUYÊN THAM KHẢO

- TensorFlow VAE Tutorial: https://www.tensorflow.org/tutorials/generative/cvae
- Keras CNN Documentation: https://keras.io/api/layers/convolution_layers/
- Bài giảng về Variational Autoencoders
- BIDMC PPG Dataset: https://physionet.org/content/bidmc/1.0.0/
