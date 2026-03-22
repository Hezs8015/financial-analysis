import re

# 读取文件
with open('app.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# 修复所有以 � 结尾的字符串
# 匹配模式: "...� 或 '...�
content = re.sub(r'"([^"]*)�', r'"\1"', content)
content = re.sub(r"'([^']*)�", r"'\1'", content)

# 写回文件
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("修复完成")
