import pandas as pd

def process_excel_sheets(input_file, output_file):
    # 读取原始Excel文件
    with pd.ExcelFile(input_file) as xls:
        # 创建一个新的Excel写入器
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 处理每个sheet
            for sheet_name in xls.sheet_names:
                # 读取当前sheet
                df = pd.read_excel(xls, sheet_name=sheet_name)
                
                # 获取第一行（列名）和最后一行（平均值）
                column_names = df.columns.tolist()  # 第一行作为列名
                last_row = df.iloc[-1].tolist()     # 最后一行作为平均值
                
                # 创建新的DataFrame，第一列是方法名，第二列是平均值
                # 注意：第一列是"Image Filename"，我们不需要它
                new_df = pd.DataFrame({
                    'Method': column_names[1:],  # 跳过第一列
                    'Mean_Value': last_row[1:]   # 跳过第一列
                })
                
                # 按照第二列（Mean_Value）进行排序
                new_df = new_df.sort_values(by='Mean_Value', ascending=False)
                
                # 写入新的sheet
                new_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"处理完成: {sheet_name} - 共{len(new_df)}个方法")

# 使用示例
input_filename = input('请输入要翻转的文件名：')
output_filename = './result_sorted.xlsx'

process_excel_sheets(input_filename, output_filename)
print(f"新文件已生成: {output_filename}")