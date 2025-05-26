


# Object-Detector-App

[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/helloworldpxy/object_detector_app/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/helloworldpxy/object_detector_app)](https://github.com/helloworldpxy/object_detector_app/stargazers)

## 项目简介  
**Object-Detector-App** 是一个基于 TensorFlow 和 OpenCV 的实时对象检测与跟踪应用。该项目支持从摄像头或视频文件输入流中实时检测常见物体（基于 COCO 数据集），并绘制边界框、类别标签及置信度。检测结果可实时显示、保存为视频文件（`output.mp4`），并生成跟踪报告（`report.csv`）。项目参考自 [datitran/object_detector_app](https://github.com/datitran/object_detector_app)，采用 **MIT 许可证**。
---

## 功能特性

- **实时对象检测**：基于 COCO 数据集，支持 90 种常见物体（如人、车辆、动物等）的检测。
- **多线程处理**：分离视频流读取与模型推理任务，提升实时性能。
- **结果可视化**：实时显示带边界框、类别标签和置信度的视频画面。
- **视频保存**：将处理后的视频保存为 `output.mp4`。
- **跟踪报告**：生成 `report.csv` 记录检测到的物体及其位置信息。
- **灵活输入源**：支持摄像头设备或本地视频文件输入。

---

## 环境安装

### 依赖库安装
```bash
pip install -r requirements.txt
```

### 预训练模型
1. 下载 [frozen_inference_graph.pb](https://github.com/helloworldpxy/object_detector_app/frozen_inference_graph.pb) 文件。
2. 确保标签文件 `detection/data/mscoco_label_map.pbtxt` 已存在。

---

## 使用说明

### 命令行参数
| 参数 | 说明 | 示例 |
|------|------|------|
| `-src` | 摄像头设备索引或视频文件路径 | `-src 0`（默认摄像头） |
| `-wd` | 帧宽度（默认 1280） | `-wd 1920` |
| `-ht` | 帧高度（默认 720） | `-ht 1080` |

### 运行程序
```bash
python objection_detection_app.py -src 0 -wd 1280 -ht 720
```

### 操作指南
1. 按下 `q` 键退出程序。
2. 检测结果实时显示在窗口中，并自动保存到 `output.mp4`。
3. 跟踪报告 `report.csv` 会记录每帧的检测结果。

---

## 项目结构
```
object_detector_app/
├── objection_detection_app.py  # 主程序（视频流处理与多线程）
├── detect_object.py            # 对象检测模块（模型推理）
├── video_writer.py             # 视频写入模块（结果保存）
├── utils/                      # 工具库
│   ├── webcam.py               # 摄像头流处理
│   └── label_map_util.py       # 标签映射解析
├── detection/                  # 检测相关文件
│   ├── data/mscoco_label_map.pbtxt  # COCO 标签映射
│   └── tf_models/              # 预训练模型目录
├── analytics/                  # 数据分析
│   └── tracking.py             # 对象跟踪与报告生成
└── frozen_inference_graph.pb   # TensorFlow 冻结模型
```

---

## 许可证
本项目基于 [MIT License](LICENSE) 开源。  
参考项目：[datitran/object_detector_app](https://github.com/datitran/object_detector_app)。

---

## 致谢
- 感谢 [datitran](https://github.com/datitran) 提供原始项目参考。
- 模型基于 [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)。

---

## 反馈与贡献
欢迎提交 Issue 或 Pull Request！  
项目主页：[https://github.com/helloworldpxy/object_detector_app](https://github.com/helloworldpxy/object_detector_app)
