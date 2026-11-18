import time

import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from chapt4_interfaces.srv import FaceDetector


class FaceDetectorClient(Node):
    def __init__(self):
        super().__init__('face_client_node')

        self.bridge = CvBridge()

        self.client = self.create_client(
            FaceDetector,
            'face_detect',
        )

        # 0表示尝试打开默认摄像头，通常是笔记本内置摄像头
        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            self.camera.release()
            raise RuntimeError(
                '无法打开摄像头，请检查隐私挡板及其他占用摄像头的程序'
            )

        # 请求摄像头使用640×480分辨率，减轻传输和检测负担
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 是否已有请求正在等待结果
        self.request_pending = False

        # 保存发给服务端的那一帧
        self.request_frame = None
        self.future = None

        # 保存最近一次检测完成、画好框的图片
        self.display_frame = None

        self.request_start_time = 0.0
        self.last_wait_log = 0.0
        self.running = True

        self.window_name = 'Camera Face Detection'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # 每隔约30毫秒处理一次摄像头和窗口
        self.timer = self.create_timer(0.03, self.send_request)

        self.get_logger().info(
            '摄像头已打开，等待检测；点击图片窗口后按q退出'
        )

    def send_request(self):
        # 1. 从摄像头读取当前画面
        success, cv_image = self.camera.read()

        if not success:
            self.get_logger().error('摄像头画面读取失败')
            self.running = False
            return

        # 第一次还没有检测结果时，先显示摄像头画面
        if self.display_frame is None:
            show_image = cv_image.copy()
        else:
            # 显示的是对应坐标已画好框的原始帧
            show_image = self.display_frame.copy()

        cv2.putText(
            show_image,
            'Press q to quit',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow(self.window_name, show_image)

        # 短暂等待，处理窗口事件，不再无限等待按键
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.running = False
            return

        if cv2.getWindowProperty(
            self.window_name,
            cv2.WND_PROP_VISIBLE,
        ) < 1:
            self.running = False
            return

        # 2. 上一帧还没返回，不继续发送新请求
        if self.request_pending:
            if time.monotonic() - self.request_start_time > 10.0:
                self.get_logger().error(
                    '等待检测结果超过10秒，请检查服务端'
                )
                self.running = False
            return

        # 服务未上线时保持预览，并间隔打印提示
        if not self.client.service_is_ready():
            now = time.monotonic()

            if now - self.last_wait_log >= 2.0:
                self.get_logger().info('等待服务端上线……')
                self.last_wait_log = now

            return

        # 3. 保存要发送的这一帧，构造请求
        self.request_frame = cv_image.copy()

        request = FaceDetector.Request()
        request.image = self.bridge.cv2_to_imgmsg(
            self.request_frame,
            encoding='bgr8',
        )

        # 4. 发送异步请求，登记结果回调
        self.future = self.client.call_async(request)
        self.request_pending = True
        self.request_start_time = time.monotonic()

        self.future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        try:
            response = future.result()

            # 在发送给服务端的那一帧上画框
            cv_image = self.request_frame.copy()

            for i in range(response.number):
                cv2.rectangle(
                    cv_image,
                    (response.left[i], response.top[i]),
                    (response.right[i], response.bottom[i]),
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                cv_image,
                f'Faces: {response.number}',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # 留给下一次定时器回调显示
            self.display_frame = cv_image

        except Exception as error:
            self.get_logger().error(f'检测请求失败：{error}')
            self.running = False

        finally:
            # 本次请求已处理完，允许下一次发送
            self.request_pending = False

    def close_camera(self):
        self.timer.cancel()

        if self.future is not None and not self.future.done():
            self.future.cancel()

        # 释放摄像头，让其他程序可以继续使用
        self.camera.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = FaceDetectorClient()

        # 每次处理一项ROS任务，并检查是否需要退出
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close_camera()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()