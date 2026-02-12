import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os

# =================配置参数=================
SAMPLES_PER_VOICE = 30  # 每个声音样本包含30帧
FEATURE_NUM = 8  # 每帧包含8个频率特征
EPOCHS = 1000  # 训练轮数
BATCH_SIZE = 4  # 【优化】稍微调大Batch size，让训练更稳定，原为1
LEARNING_RATE = 0.00001  # 学习率

# 【修改1】增加 Noise 的标签定义
LABEL_SILENCE = 0
LABEL_OPEN = 1
LABEL_CLOSE = 2
LABEL_NOISE = 3  # 新增标签


# =================1. 数据预处理函数=================
def load_and_process_data(csv_path, label):
    """
    读取CSV，归一化，并重塑为模型需要的输入格式 (N, 240)
    假设CSV格式为：每30行代表一个样本，每行8列数据
    """
    if not os.path.exists(csv_path):
        print(f"警告: 找不到文件 {csv_path}，将跳过此类别")
        return np.empty([0, SAMPLES_PER_VOICE * FEATURE_NUM]), np.empty([0])

    # 读取CSV，无表头
    try:
        df = pd.read_csv(csv_path, header=None)
        raw_data = df.values
    except Exception as e:
        print(f"错误: 读取 {csv_path} 失败: {e}")
        return np.empty([0, SAMPLES_PER_VOICE * FEATURE_NUM]), np.empty([0])

    # 计算有多少个完整的声音样本 (总行数 // 30)
    # 确保行数是30的倍数，多余的行丢弃
    num_samples = raw_data.shape[0] // SAMPLES_PER_VOICE

    # 截取有效数据 (防止最后几行不够30行报错)
    raw_data = raw_data[:num_samples * SAMPLES_PER_VOICE, :]

    # 初始化数据容器
    data_x = np.zeros((num_samples, SAMPLES_PER_VOICE * FEATURE_NUM))
    data_y = np.full((num_samples,), label)  # 填充标签

    print(f"处理文件: {csv_path}, 样本数: {num_samples}")

    for i in range(num_samples):
        # 取出当前样本的30行8列数据
        sample_block = raw_data[i * SAMPLES_PER_VOICE: (i + 1) * SAMPLES_PER_VOICE, :]

        # 归一化: 除以128.0 (将范围压缩到 0-1 之间)
        sample_norm = sample_block / 128.0

        # 展平: 将 (30, 8) 展平为 (240,)
        data_x[i, :] = sample_norm.flatten()

    return data_x, data_y


# =================2. 加载并合并数据集 (升级版)=================
print("--- 开始加载数据 ---")


# 1. 定义增强函数
def augment_data(x_data, y_label):
    """
    输入原始数据，返回一个包含 [0.6倍, 1.0倍, 1.4倍] 的混合数据集
    """
    if x_data.shape[0] == 0:
        return x_data, y_label

    # 系数列表：
    # 0.6: 模拟小声，提高灵敏度
    # 1.0: 保持原样
    # 1.4: 模拟大声截断，提高抗拍手干扰能力
    factors = [0.6, 1.0, 1.4]

    augmented_x_list = []
    augmented_y_list = []

    for f in factors:
        # 乘系数
        x_aug = x_data * f
        # 截断在 0.0 到 1.0 之间 (对应 Arduino 的 0-128)
        x_aug = np.clip(x_aug, 0.0, 1.0)

        augmented_x_list.append(x_aug)

        # 对应的标签也要复制
        y_aug = np.full((x_aug.shape[0],), y_label)
        augmented_y_list.append(y_aug)

    # 将列表合并成一个大数组
    return np.concatenate(augmented_x_list, axis=0), np.concatenate(augmented_y_list, axis=0)


# 2. 加载原始数据
raw_silence_x, raw_silence_y = load_and_process_data('data/silence.csv', LABEL_SILENCE)
raw_open_x, raw_open_y = load_and_process_data('data/open.csv', LABEL_OPEN)
raw_close_x, raw_close_y = load_and_process_data('data/close.csv', LABEL_CLOSE)
raw_noise_x, raw_noise_y = load_and_process_data('data/noise.csv', LABEL_NOISE)

# 3. 对不同类别应用不同的策略

# Silence 和 Noise 不需要增强 (或者只做简单的截断)
# Noise 已经有150条了，足够多，不需要再乘倍数，否则训练时间太长
final_silence_x, final_silence_y = raw_silence_x, raw_silence_y
final_noise_x, final_noise_y = raw_noise_x, raw_noise_y

# Open 和 Close 需要重点增强！！！
# 100条 -> 变成 300条 (包含小声、正常、大声)
final_open_x, final_open_y = augment_data(raw_open_x, LABEL_OPEN)
final_close_x, final_close_y = augment_data(raw_close_x, LABEL_CLOSE)

# 4. 合并所有数据
X = np.concatenate((final_silence_x, final_open_x, final_close_x, final_noise_x), axis=0)
Y = np.concatenate((final_silence_y, final_open_y, final_close_y, final_noise_y), axis=0)

print(f"--- 数据准备完成 ---")
print(f"Silence 样本数: {final_silence_x.shape[0]}")
print(f"Open    样本数: {final_open_x.shape[0]} (已增强)")
print(f"Close   样本数: {final_close_x.shape[0]} (已增强)")
print(f"Noise   样本数: {final_noise_x.shape[0]}")
print(f"总数据量 X: {X.shape}")

# ... 接下来的代码 (打乱、模型构建) 保持不变 ...


# =================3. 数据洗牌与切分=================
# 生成随机索引进行洗牌
indices = np.random.permutation(X.shape[0])
X_shuffled = X[indices]
Y_shuffled = Y[indices]

# 切分训练集(80%) 和 测试集(20%)
split_idx = int(X.shape[0] * 0.2)
x_test = X_shuffled[:split_idx]
y_test = Y_shuffled[:split_idx]
x_train = X_shuffled[split_idx:]
y_train = Y_shuffled[split_idx:]

print(f"训练集大小: {x_train.shape[0]}, 测试集大小: {x_test.shape[0]}")

# =================4. 构建模型 (Keras)=================
model = keras.Sequential([
    # 输入层：30帧 * 8特征 = 240个输入节点
    keras.layers.Dense(32, input_shape=(SAMPLES_PER_VOICE * FEATURE_NUM,), activation='relu'),
    # 隐藏层
    keras.layers.Dense(16, activation='relu'),
    # 【修改4】输出层：改为 4 个类别 (Silence, Open, Close, Noise)
    keras.layers.Dense(4, activation='softmax')
])

# 编译模型
adam = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['sparse_categorical_accuracy'])

model.summary()

# =================5. 模型训练=================
print("\n--- 开始训练 ---")
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    verbose=1
)

# =================6. 导出 TFLite 模型=================
print("\n--- 导出 TFLite 模型 ---")

# 1. 先保存为 SavedModel
export_dir = "saved_model"
try:
    model.export(export_dir)  # Keras 3
except AttributeError:
    model.save(export_dir)  # Keras 2 / Legacy

# 2. 转换
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

# 保存 .tflite 文件
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print("已保存: model.tflite")


# =================7. 生成 C 头文件=================
def convert_to_c_array(bytes_data, model_name="model"):
    hex_str = ", ".join([f"0x{b:02x}" for b in bytes_data])
    # 为了美观，每行放12个字节
    lines = []
    hex_list = [f"0x{b:02x}" for b in bytes_data]
    for i in range(0, len(hex_list), 12):
        lines.append(", ".join(hex_list[i:i + 12]))
    formatted_hex = ",\n  ".join(lines)

    c_str = f"const unsigned char {model_name}[] = {{\n  {formatted_hex}\n}};\n"
    c_str += f"const unsigned int {model_name}_len = {len(bytes_data)};"
    return c_str


c_code = convert_to_c_array(tflite_model)
with open("model.h", "w") as f:
    f.write(c_code)

print(f"已保存: model.h (大小: {len(tflite_model)} 字节)")
print("请将 model.h 复制到你的 Arduino 项目文件夹中，并重新编译上传。")
