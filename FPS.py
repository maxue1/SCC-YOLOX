import torch
import time
from nets.yolo import YoloBody

def measure_fps(model, input_shape=(640, 640), num_classes=1, device='cuda', num_trials=100):
    # 初始化模型
    model = model(num_classes, 's').to(device)
    model.eval()

    # 创建随机输入
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)

    # 预热模型，消除首次运行的影响
    for _ in range(10):
        _ = model(dummy_input)

    # 正式计时
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()

    for _ in range(num_trials):
        with torch.no_grad():
            _ = model(dummy_input)

    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()

    # 计算FPS
    total_time = end_time - start_time
    avg_time_per_image = total_time / num_trials
    fps = 1 / avg_time_per_image

    print(f"Average Inference Time per Image: {avg_time_per_image:.4f} s")
    print(f"FPS: {fps:.2f}")
    return fps

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    measure_fps(YoloBody, device=device)
