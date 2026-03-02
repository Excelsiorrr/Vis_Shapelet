import torch
import os

# 定义数据文件夹路径
data_folder = r"d:\shapelet\ShapeX\datasets\mitecg"

print("=" * 60)
print("MIT-ECG 数据集文件检查")
print("=" * 60)

# 检查根目录下的文件
root_files = ["mitecg.pt", "saliency.pt", "X.pt", "y.pt"]

for file_name in root_files:
    file_path = os.path.join(data_folder, file_name)
    if os.path.exists(file_path):
        print(f"\n📁 文件: {file_name}")
        print("-" * 60)
        try:
            data = torch.load(file_path)
            print(f"数据类型: {type(data)}")
            
            if isinstance(data, torch.Tensor):
                print(f"张量形状: {data.shape}")
                print(f"数据类型: {data.dtype}")
                print(f"数值范围: [{data.min():.4f}, {data.max():.4f}]")
                print(f"前5个元素:\n{data[:5]}")
            elif isinstance(data, dict):
                print(f"字典键: {list(data.keys())}")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
                    else:
                        print(f"  {key}: 类型={type(value)}")
            elif isinstance(data, list):
                print(f"列表长度: {len(data)}")
                if len(data) > 0:
                    print(f"第一个元素类型: {type(data[0])}")
                    if isinstance(data[0], torch.Tensor):
                        print(f"第一个元素形状: {data[0].shape}")
            else:
                print(f"数据内容: {data}")
        except Exception as e:
            print(f"❌ 加载失败: {e}")
    else:
        print(f"\n⚠️  文件不存在: {file_name}")

# 检查 all_data 子文件夹
all_data_folder = os.path.join(data_folder, "all_data")
if os.path.exists(all_data_folder):
    print("\n" + "=" * 60)
    print("all_data 文件夹")
    print("=" * 60)
    
    all_data_files = ["saliency.pt", "X.pt", "y.pt"]
    for file_name in all_data_files:
        file_path = os.path.join(all_data_folder, file_name)
        if os.path.exists(file_path):
            print(f"\n📁 文件: all_data/{file_name}")
            print("-" * 60)
            try:
                data = torch.load(file_path)
                print(f"数据类型: {type(data)}")
                
                if isinstance(data, torch.Tensor):
                    print(f"张量形状: {data.shape}")
                    print(f"数据类型: {data.dtype}")
                    print(f"数值范围: [{data.min():.4f}, {data.max():.4f}]")
                    print(f"前5个元素:\n{data[:5]}")
                elif isinstance(data, dict):
                    print(f"字典键: {list(data.keys())}")
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
            except Exception as e:
                print(f"❌ 加载失败: {e}")

print("\n" + "=" * 60)
print("检查完成!")
print("=" * 60)