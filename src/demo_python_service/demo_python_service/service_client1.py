import rclpy
from rclpy.node import Node
from chapt4_interfaces.srv import FaceDetector
from cv_bridge import CvBridge
import cv2
import os

class FaceDetectorClient(Node):
    def __init__(self):
        super().__init__('face_client_node')
        self.bridge = CvBridge()

        #创建客户端，找到/face_detect的服务
        self.client = self.create_client(
            FaceDetector,
            'face_detect'
        )
        #如果服务端没有启动，每秒查看一次
        while not self.client.wait_for_service(timeout_sec = 1.0):
            self.get_logger().info("等待服务器上线")
        #设置图片路径
        self.image_path = os.path.expanduser('~/图片/test.jpg')

    def send_request(self):
        #1.加载本地图片，你要吃什么
        cv_image = cv2.imread(self.image_path) #cv2.imread()：读取图片

        #2.构造请求，填写菜单
        #FaceDetector服务类型提供两种对象 请求.Request() 响应.Response()
        request = FaceDetector.Request()

        # 把 OpenCV 的图翻译成 ROS 的图，塞进请求的 image 字段
        request.image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

         # 3. 点菜（发送异步请求）
        # call_async 会立刻返回一个小票 (future)
        #发出请求，不等待人脸检测完成，就返回一个Future对象。
        self.future = self.client.call_async(request)
        # 给小票绑个服务员（回调）：等后厨把菜做好了，触发 self.result_callback
        self.future.add_done_callback(self.result_callback)



    def result_callback(self, future):
        try:
            # 拿菜（拿到最终响应结果）
            response = future.result()
            self.get_logger().info(f"收到结果：人脸数 {response.number}")
            
            # 吃菜（把服务端返回的坐标画在图上）
            cv_image = cv2.imread(self.image_path)
            # 遍历所有人脸，画绿色的框
            for i in range(response.number):
                cv2.rectangle(cv_image, 
                              (response.left[i], response.top[i]), # 左上角
                              (response.right[i], response.bottom[i]), # 右下角
                              (0, 255, 0), 2) # 绿色，线宽2
            
            # 展示图片，按任意键关闭
            cv2.imshow("Face Detection Client", cv_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.get_logger().error(f"调用失败: {e}")
        finally:
            rclpy.shutdown() # 吃完擦嘴（关闭 ROS）

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectorClient()
    node.send_request() # 一启动就点菜（发请求）
    rclpy.spin(node)    # 等待服务员上菜（让节点保持运行，以便回调函数执行）

if __name__ == '__main__':
    main()