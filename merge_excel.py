import pandas as pd
import os
import glob
import numpy as np

def merge_excel_files_advanced(input_folder=None, file_pattern="*.xlsx", output_file="合并结果.xlsx"):
    """
    高级版本：支持文件夹扫描和更多选项
    
    参数:
    input_folder: 输入文件夹路径（如果为None，则使用当前目录）
    file_pattern: 文件匹配模式
    output_file: 输出文件路径
    """
    
    # 获取文件列表
    if input_folder is None:
        input_folder = "."
    
    file_paths = glob.glob(os.path.join(input_folder, file_pattern))
    
    if not file_paths:
        print("未找到匹配的Excel文件")
        return
    
    print(f"找到 {len(file_paths)} 个Excel文件:")
    for file_path in file_paths:
        print(f"  - {os.path.basename(file_path)}")
    
    all_sheets_data = {}
    
    # 处理每个文件
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"\n处理文件: {file_name}")
        
        try:
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                if df.shape[1] < 2:
                    print(f"  跳过sheet '{sheet_name}' (列数不足)")
                    continue
                
                # 如果是第一次遇到这个sheet，添加第一列
                if sheet_name not in all_sheets_data:
                    # 创建新的DataFrame，并添加第一列
                    first_column = df.iloc[:, 0]
                    all_sheets_data[sheet_name] = pd.DataFrame({f"Image Filename": first_column})
                    print(f"  已添加sheet '{sheet_name}' 的第一列")
                
                # 添加第二列
                second_column = df.iloc[:, 1]
                
                # 生成唯一的列名
                base_column_name = file_name
                column_name = base_column_name
                counter = 1
                while column_name in all_sheets_data[sheet_name].columns:
                    column_name = f"{base_column_name}_{counter}"
                    counter += 1
                
                all_sheets_data[sheet_name][column_name] = second_column
                print(f"  已添加sheet '{sheet_name}' 的第二列")
                
        except Exception as e:
            print(f"  处理失败: {str(e)}")
    
    # 对每个sheet按照最后一行的值进行排序（不包括第一列）
    print("\n正在对每个sheet的列进行排序...")
    for sheet_name, data in all_sheets_data.items():
        if not data.empty:
            # 分离第一列和其他列
            first_column = data.iloc[:, 0]  # 第一列
            other_columns = data.iloc[:, 1:]  # 其他列
            
            # 只对其他列进行排序
            sorted_other_columns = sort_columns_by_last_row(other_columns)
            
            # 重新组合数据
            sorted_data = pd.concat([first_column, sorted_other_columns], axis=1)
            all_sheets_data[sheet_name] = sorted_data
            print(f"  已对sheet '{sheet_name}' 的列进行排序")
    
    # 写入结果文件
    if not all_sheets_data:
        print("没有有效数据可合并")
        return
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, data in all_sheets_data.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"已保存sheet: {sheet_name} ({data.shape[1]}列, {data.shape[0]}行)")
    
    print(f"\n✅ 合并完成！输出文件: {output_file}")

def sort_columns_by_last_row(df, ascending=False):
    """
    根据最后一行的值对DataFrame的列进行排序
    
    参数:
    df: 要排序的DataFrame
    ascending: 排序顺序，True为升序，False为降序
    
    返回:
    排序后的DataFrame
    """
    if df.empty:
        return df
    
    # 获取最后一行的值
    last_row = df.iloc[-1]
    
    # 处理NaN值，将它们放在最后
    last_row_filled = last_row.fillna(np.inf if ascending else -np.inf)
    
    # 获取按照最后一行值排序的列索引
    sorted_columns = last_row_filled.sort_values(ascending=ascending).index
    
    # 按照排序后的列顺序重新排列DataFrame
    return df[sorted_columns]

# 使用示例
if __name__ == "__main__":
    file_path = input('请输入要合并的文件夹')
    dataset_name = input('请输入保存的数据集')
    # 方法2: 扫描文件夹中的所有Excel文件
    merge_excel_files_advanced(
        input_folder=file_path,  # 包含Excel文件的文件夹
        file_pattern="*.xlsx",         # 文件模式
        output_file=f"{dataset_name}.xlsx"  # 输出文件名
    )