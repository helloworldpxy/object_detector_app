import struct
import six
import collections
import cv2
import datetime
from threading import Thread
from matplotlib import colors

class FPS:
    def __init__(self):
        # 初始化FPS类，记录开始时间、结束时间和帧数
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # 记录视频分析的开始时间
        self._start = datetime.datetime.now()
        return self
    
    def stop(self):
        # 记录视频分析的结束时间
        self._end = datetime.datetime.now()

    def update(self):
        # 更新帧数
        self._numFrames += 1

    def elapsed(self):
        # 返回总的分析时间（秒）
        return (self._end - self._start).total_seconds()

    def fps(self):
        # 计算近似每秒帧数
        return self._numFrames / self.elapsed()

    def get_numFrames(self):
        # 返回总的帧数
        return self._numFrames


class WebcamVideoStream:
    def __init__(self, src, width, height):
        # 初始化摄像头视频流，并读取第一帧
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False


    def start(self):
        # 启动线程以从视频流中读取帧
        Thread(target=self.update, args=()).start()
        return self


    def update(self):
        # 循环直到线程停止，不断读取下一帧
        while True:
            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()


    def read(self):
        # 返回最近读取的帧
        return self.frame


    def stop(self):
        # 停止线程并释放视频流
        self.stopped = True
        self.stream.release()


def standard_colors():
    # 返回一组标准颜色名称的列表
    colors = [
        'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
        'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
        'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
        'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
        'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
        'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
        'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
        'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
        'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
        'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
        'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
        'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
        'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
        'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
        'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
        'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
        'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
        'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
        'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
        'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
        'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
        'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
        'WhiteSmoke', 'Yellow', 'YellowGreen'
    ]
    return colors

def color_name_to_rgb():
    # 将颜色名称转换为RGB值
    colors_rgb = []
    for key, value in colors.cnames.items():
        colors_rgb.append((key, struct.unpack('BBB', bytes.fromhex(value.replace('#', '')))))
    return dict(colors_rgb)


def draw_boxes_and_labels(
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        keypoints=None,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        agnostic_mode=False):
    """Returns boxes coordinates, class names and colors
    Args:
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      instance_masks: a numpy array of shape [N, image_height, image_width], can
        be None
      keypoints: a numpy array of shape [N, num_keypoints, 2], can
        be None
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
      min_score_thresh: minimum score threshold for a box to be visualized
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
    """
    # 创建每个框的位置、类别名称和颜色的显示字符串
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = 'black'
            else:
                if not agnostic_mode:
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = '{}: {}%'.format(
                        class_name,
                        int(100 * scores[i]))
                else:
                    display_str = 'score: {}%'.format(int(100 * scores[i]))
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = standard_colors()[
                        classes[i] % len(standard_colors())]

    # 存储所有框的坐标、类别名称和颜色
    color_rgb = color_name_to_rgb()
    rect_points = []
    class_names = []
    class_colors = []
    for box, color in six.iteritems(box_to_color_map):
        ymin, xmin, ymax, xmax = box
        rect_points.append(dict(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax))
        class_names.append(box_to_display_str_map[box])
        class_colors.append(color_rgb[color.lower()])
    return rect_points, class_names, class_colors