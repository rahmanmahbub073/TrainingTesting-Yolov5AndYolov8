import os
 
def calculate_iou(box1, box2):
    
    Xcenter1, Ycenter1, W1, H1 = box1[0],box1[1],box1[2],box1[3]
    Xcenter2, Ycenter2, W2, H2 = box2[0],box2[1],box2[2],box2[3]
    Xmin1, Ymin1 = Xcenter1 - W1 / 2, Ycenter1 - H1 / 2
    Xmax1, Ymax1 = Xcenter1 + W1 / 2, Ycenter1 + H1 / 2
    Xmin2, Ymin2 = Xcenter2 - W2 / 2, Ycenter2 - H2 / 2
    Xmax2, Ymax2 = Xcenter2 + W2 / 2, Ycenter2 + H2 / 2


    inter_Xmin = max(Xmin1, Xmin2)
    inter_Ymin = max(Ymin1, Ymin2)
    inter_Xmax = min(Xmax1, Xmax2)
    inter_Ymax = min(Ymax1, Ymax2)
 
    # 以免不相交
    W = max(0, inter_Xmax - inter_Xmin)
    H = max(0, inter_Ymax - inter_Ymin)
 
    # 计算相交区域面积
    inter_area = W * H
 
    # 计算并集面积
    merge_area = (Xmax1 - Xmin1) * (Ymax1 - Ymin1) + (Xmax2 - Xmin2) * (Ymax2 - Ymin2)
 
    # 计算IOU
    IOU = inter_area / (merge_area - inter_area + 1e-6)


    return IOU
 
def calculate_metrics(ground_truth_boxes, detected_boxes):
    true_positives = 0  #真正例，预测为正例而且实际上也是正例；
    false_negatives = 0 #假负例，预测为负例然而实际上却是正例；
    false_positives = 0 #假正例，预测为正例然而实际上却是负例；
    true_negatives = 0  # 真负例，预测为负例而且实际上也是负例。目标检测默认为0 
 
    # 遍历每个真实框
    for truth_box in ground_truth_boxes:
        found_match = False
        # 遍历每个检测到的框
        for detected_box in detected_boxes:
            iou = calculate_iou(truth_box, detected_box) # 计算真实框和检测框的交并比
            if iou >= 0.5 and truth_box[4] == detected_box[4]:  # 如果交并比大于等于0.5且类别匹配
                true_positives += 1 # 则增加真正例计数
                found_match = True
                break
        if not found_match:
            false_negatives += 1  # 如果未找到匹配的检测框，则增加假负例计数
 
    unique_detected_classes = set(box[4] for box in detected_boxes) # 获取检测到的框中的唯一类别
    unique_ground_truth_classes = set(box[4] for box in ground_truth_boxes) # 获取真实框中的唯一类别
 
    for c in unique_detected_classes:
        if c not in unique_ground_truth_classes:
            false_positives += sum(box[4] == c for box in detected_boxes) # 计算假正例
 
    for c in unique_ground_truth_classes:
        if c not in unique_detected_classes:
            false_negatives += sum(box[4] == c for box in ground_truth_boxes)  # 根据未检测到的类别，增加假负例的计数
 
    total_instances = true_positives + false_negatives + false_positives + true_negatives
    accuracy = (true_positives + true_negatives) / total_instances if total_instances > 0 else 0 # 准确率计算
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0 # 召回率计算
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0 # 精确率计算
    miss_rate = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0 # 漏检率计算
    false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0 # 误检率计算
 
    return recall, miss_rate, false_positive_rate, precision, accuracy
 
def read_boxes_from_txt(file_path):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            box = [float(values[1]), float(values[2]), float(values[3]), float(values[4]), values[0]]
            boxes.append(box)
    return boxes
 
# 文件夹路径
ground_truth_folder = '/data01/user1/FinalProject/ground_truth_dataset/labels'  # 真值laebl文件夹路径
detected_folder = '/data01/user1/FinalProject/yolov5/runs/detect/exp7/labels/'  # 预测laebl文件夹路径3
 
# 计算目标检测指标
total_recall = 0
total_miss_rate = 0
total_false_positive_rate = 0
total_precision = 0
total_accuracy = 0
num_images = 0
 
for filename in os.listdir(ground_truth_folder):
    if filename.endswith(".txt"):
        ground_truth_boxes = read_boxes_from_txt(os.path.join(ground_truth_folder, filename))
        detected_boxes = read_boxes_from_txt(os.path.join(detected_folder, filename))
 
        recall, miss_rate, false_positive_rate, precision, accuracy = calculate_metrics(ground_truth_boxes, detected_boxes)
        total_recall += recall
        total_miss_rate += miss_rate
        total_false_positive_rate += false_positive_rate
        total_precision += precision
        total_accuracy += accuracy
        num_images += 1
        
 
average_recall = total_recall / num_images
average_miss_rate = total_miss_rate / num_images
average_false_positive_rate = total_false_positive_rate / num_images
average_precision = total_precision / num_images
average_accuracy = total_accuracy / num_images
 
print("Average Recall: ", average_recall)
print("Average Miss Rate: ", average_miss_rate)
print("Average False Positive Rate: ", average_false_positive_rate)
print("Average Precision: ", average_precision)
print("Average Accuracy: ", average_accuracy)

print("测试图片数量：{}".format(num_images))