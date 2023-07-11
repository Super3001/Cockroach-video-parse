import pandas as pd
from openpyxl import load_workbook

# 创建DataFrame对象
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# 打开Excel文件
book = load_workbook('example.xlsx')

# 新建工作表并将DataFrame对象写入其中
writer = pd.ExcelWriter('example.xlsx', engine='openpyxl')
writer.book = book
df.to_excel(writer, sheet_name='Sheet2')

# 保存Excel文件
writer.save()

