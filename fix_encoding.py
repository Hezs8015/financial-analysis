import re

# 读取文件
with open('app.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# 查找所有未闭合的字符串（以 � 结尾的字符串）
# 这些通常是中文字符被截断导致的
lines = content.split('\n')
fixed_lines = []

for line in lines:
    # 修复以 � 结尾的字符串
    if '"�' in line and not line.strip().endswith('"'):
        line = line.replace('"�', '"')
    if "'�" in line and not line.strip().endswith("'"):
        line = line.replace("'�", "'")
    fixed_lines.append(line)

# 写回文件
with open('app.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(fixed_lines))

print("修复完成")
