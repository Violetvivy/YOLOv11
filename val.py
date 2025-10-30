from ultralytics import YOLO
import numpy as np

def main():
    # 加载训练好的模型
    model = YOLO('runs/detect/model_based_m/weights/best.pt')

    # 在测试集上评估模型
    results = model.val(
    data='E:/Dataset/Fruits 2.v1-three.yolov11/data.yaml',
    split='test',
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.6,
    device=0
    )

    print("=== 测试集评估结果 ===")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")

    # 处理精确率和召回率的数组格式
    if hasattr(results.box, 'p'):
        precision = results.box.p
        if isinstance(precision, (np.ndarray, list)):
            # 取所有类别的平均精确率
            mean_precision = np.mean(precision)
            print(f"平均精确率 (Precision): {mean_precision:.4f}")
        else:
            print(f"精确率 (Precision): {precision:.4f}")
    else:
        print("精确率数据不可用")

    if hasattr(results.box, 'r'):
        recall = results.box.r
        if isinstance(recall, (np.ndarray, list)):
            # 取所有类别的平均召回率
            mean_recall = np.mean(recall)
            print(f"平均召回率 (Recall): {mean_recall:.4f}")
        else:
            print(f"召回率 (Recall): {recall:.4f}")
    else:
        print("召回率数据不可用")

    # 检查是否达到0.95的目标
    target_accuracy = 0.95
    current_accuracy = results.box.map50

    print(f"\n目标mAP@0.5: {target_accuracy:.2f}")
    print(f"当前mAP@0.5: {current_accuracy:.4f}")

    if current_accuracy >= target_accuracy:
        print("✅ 恭喜！模型已达到目标准确率！")
    else:
        print(f"❌ 模型未达到目标准确率，差距: {target_accuracy - current_accuracy:.4f}")

if __name__ == "__main__":
    main()