import cv2

class VideoWriter(object):
    # 定义VideoWriter，设置视频保存路径、尺寸、编码器、帧率和颜色格式等
    def __init__(self, path, size):
        self.path = path
        self.size = size
        self.fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
        self.fps = 30.0
        self.is_color = True

    # 将帧写入视频文件
    def __call__(self, frame):
        self.writer.write(frame)

    # 释放VideoWriter资源，关闭视频文件
    def close(self):
        self.writer.release()