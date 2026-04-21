import sys

# Pythonの検索パスを表示
print("\nPython module search paths:")
for path in sys.path:
    print(path)

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Python prefix:", sys.prefix)
