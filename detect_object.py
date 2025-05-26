import os
import numpy as np
from utils import label_map_util
from utils.webcam import draw_boxes_and_labels

CWD_PATH = os.getcwd()
PATH_TO_LABELS = os.path.join(CWD_PATH, 'detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# 加载标签映射
try:
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
except Exception as e:
    print(f"加载标签映射时出错: {e}")
    raise

# 通过给定的图像和会话检测物体，返回检测结果
def detect_objects(image_np, sess, detection_graph):
    # 扩展图像数组维度
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # 获取输入的图像张量
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # 获取输出的框坐标、分数、类别
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # 执行检测/模型预测
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # 解压并绘制边界框和标签
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=0.5
    )

    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)