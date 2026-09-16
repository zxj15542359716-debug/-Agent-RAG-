#为整个工程提供统一的绝对路径
import os

def get_project_root()->str:
   """获取根目录"""

   #当前文件绝对路径
   current_file = os.path.abspath(__file__)
   #文件所在文件夹
   current_dir = os.path.dirname(current_file)
   project_root = os.path.dirname(current_dir)
   return project_root

def get_abs_path(relative_path:str)->str:
    """入相对路径输出绝对路径（字符串）"""
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)

if __name__ == '__main__':
    print(get_abs_path(__file__))  #最内侧括号里要取得配置文件
