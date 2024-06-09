import cv2
import os
 
def txtShow(img,txt,img_name,savedir_path,save=True):
    image = cv2.imread(img)
    height,width = image.shape[:2]      # 获取原始图像的高和宽
 
    # 读取classes类别信息
    with open('class.txt','r') as f:
        classes = f.read().splitlines()
    # ['person', 'car']
 
    # 读取yolo格式标注的txt信息
    with open(txt,'r') as f:
        labels = f.read().splitlines()
    # ['0 0.403646 0.485491 0.103423 0.110863', '1 0.658482 0.425595 0.09375 0.099702', '2 0.482515 0.603795 0.061756 0.045387', '3 0.594122 0.610863 0.063244 0.052083', '4 0.496652 0.387649 0.064732 0.049107']
 
    ob = []         # 存放目标信息
    for i in labels:
        cl, x_centre, y_centre, w, h = i.split(' ')
 
        # 需要将数据类型转换成数字型
        cl, x_centre, y_centre, w, h = int(cl), float(x_centre), float(y_centre), float(w),float(h)
        name = classes[cl]      # 根据classes文件获取真实目标
        xmin = int(x_centre * width - w * width / 2)        # 坐标转换
        ymin = int(y_centre * height - h * height / 2)
        xmax = int(x_centre * width + w * width / 2)
        ymax = int(y_centre * height + h * height / 2)
 
        tmp = [name, xmin, ymin, xmax, ymax]  # 单个检测框
        ob.append(tmp)
 
    # 绘制检测框
    for name, x1, y1, x2, y2 in ob:
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)  # 绘制矩形框
        cv2.putText(image, name, (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, thickness=1, color=(0, 0, 255))
 
        # 保存图像
    if save:
        cv2.imwrite(os.path.join(savedir_path,img_name), image)
 
        # 展示图像
    # cv2.imshow('test', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
 
 
if __name__=='__main__':

    imgdir_path = " "      #图片文件夹路径   文件夹中需包含images和labels
    savedir_path = " "     #保存可视化结果的文件夹路径
    
    for img_name in os.listdir(imgdir_path):
        img_path = os.path.join(imgdir_path,img_name)
        label_path = img_path.replace('images','labels')
        label_path = label_path.replace('.jpg','.txt')
        txtShow(img=img_path,txt=label_path,img_name=img_name,savedir_path=savedir_path,save=True)
        
        
        