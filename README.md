文件结构
PLJC（对应的文件名）/
│
├── dataset/                    # 数据集目录（被.gitignore忽略）-- 需要自己在群里下载
│    └── ...                    # 实际数据文件，
│
├── data/                       # 预处理后的数据缓存目录（被.gitignore忽略） -- 会自动生成
│    ├── X_<fps>.npy            # 特征数据（按帧率保存）
│    └── y_<fps>.npy            # 标签数据（按帧率保存）
│
├── models/                     # 模型保存目录（被.gitignore忽略） -- 会自动生成
│    ├── best_fatigue_model.pth # 最佳模型权重
│    └── final_fatigue_model.pth# 最终训练完成的模型权重
│
├── shape_predictor_68_face_landmarks.dat  # 人脸关键点检测模型文件（dlib使用） -- 需要自己在群里下载
└── ...                         # 其他可能的辅助文件或脚本
