# Python Hello World

当用户要求创建一个输出 Hello World 的 Python 脚本时使用。

1. 若用户没有指定文件名，使用 `hello.py`；若指定了文件名，使用用户指定的工作空间相对路径。
2. 文件内容使用 Python 代码 `print("Hello, world!")`，末尾加换行。
3. 调用 `create_file` 创建文件。若文件已存在，不要直接覆盖；向用户说明，再按用户明确要求使用 `write_file`。
4. 回答实际创建的文件路径。只有确实执行过代码时才声称运行成功。
