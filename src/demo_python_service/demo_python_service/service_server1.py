import rclpy
from rclpy.node import Node

#FaceDetector	你自定义的 Service 接口类型
from chapt4_interfaces.srv import FaceDetector

from cv_bridge import CvBridge
import cv2  #OpenCV 的 Python 模块
import time
import os

class FaceDetectorServer(Node):
    def __init__ (self):
        #给节点取名字face_server_node
        super().__init__('face_server_node')

        #创建服务
        self.srv = self.create_service(
            FaceDetector,    #参数1：用什么接口（菜单）
            'face_detect', #参数2：服务名叫什么（客人找这个名字）
            self.detect_callback   #参数3：接到客人后找谁处理 
        )
        #把ros图像和opencv图互相转化,CvBridge图像转换工具类
        self.bridge = CvBridge()

        #获取opencv自带的人脸检测模型路径
        cascade_path = (
    '/usr/share/opencv4/haarcascades/'
    'haarcascade_frontalface_default.xml'
)
                #OpenCV 提供的 Haar 级联模型目录路径 #OpenCV 提供的一个预训练正面人脸检测模型文件
                #+ 是字符串拼接
        
        #创建人脸检测器并加载模型
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
                                    #创建分类器对象并加载模型
        
           
        # 为命令行测试准备一张默认图
        self.default_image_path = os.path.expanduser('~/图片/test.jpg')
        self.get_logger().info("人脸检测服务端已启动，等待请求...")

    # 客人来了（接到请求），触发这个函数
    def detect_callback(self,request,response):
        start_time = time.time() #先记录开始的时间，后面计算总共花费的时间

        #1.拿图片
        if request.image.data :# 如果客人自己带了图
            # 翻译成 OpenCV 能懂的格式 (bgr8 是彩色图的格式)
            cv_image = self.bridge.imgmsg_to_cv2(
                request.image,
                 desired_encoding='bgr8')
            
        else: # 如果客人没带图（比如用命令行调用），用我们准备的默认图
            cv_image = cv2.imread(self.default_image_path)

        #2.检测人脸
        # 先把彩图变灰度图，因为检测模型只认灰度
        #第一个参数 cv_image：原图，第二个参数 cv2.COLOR_BGR2GRAY：从BGR彩色图转换为灰度图。
        gray = cv2.cvtColor(cv_image , cv2.COLOR_BGR2GRAY)

        #找脸,`detectMultiScale()` 在不同大小的尺度上寻找人脸
        #faces 保存检测到的人脸矩形，每张脸包含四个数
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5

        )

        response.number = len(faces) #找到几张脸
        response.use_time = time.time() - start_time #耗时

        ## 把每个脸的坐标塞进对应的数组里
        #x左上角横坐标 y左上角纵坐标 w宽度 h高度
        #.append()把一个数追加到数组末尾
        for (x , y , w , h) in faces :
            response.top.append(int(y)) #上
            response.right.append(int(x + w))#右
            response.bottom.append(int(y + h))#下
            response.left.append(int(x))       #左
        self.get_logger().info(f"检测完成！发现 {response.number} 张脸")
        return response

def main():
    rclpy.init()
    node = FaceDetectorServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()