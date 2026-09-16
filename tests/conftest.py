#测试公共夹具
#说明：测试通过 FastAPI TestClient 直接调用应用，不启动真实服务器、不调用外部大模型 API；
#数据库使用临时文件（tmp_path），不触碰项目内 data/database/aftersales.db。
import os
import sys
from pathlib import Path

import pytest

#项目根目录加入 sys.path，保证 `from app import app` 可导入
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#在任何应用模块导入之前设置环境变量：
#auth_service / fectory 在导入期读取这些变量（缺失即 fail-fast），测试需要占位值，
#避免依赖运行机器上是否恰好配置了真实的 Key
os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-0123456789abcdef")
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-placeholder")
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test-placeholder")


@pytest.fixture()
def client_and_db(tmp_path, monkeypatch):
    """临时数据库 + TestClient：每个用例独立数据库，互不干扰。

    import_seed=False 跳过 CSV 初始导入（否则每个用例都要 bcrypt 哈希 170+ 个用户，太慢）。
    """
    from service import database_service as db_module

    db = db_module.DatabaseService(db_path=str(tmp_path / "test.db"), import_seed=False)
    #替换模块级单例：app 内部通过 get_database_service() 取到的就是临时库
    monkeypatch.setattr(db_module, "_instance", db)

    from fastapi.testclient import TestClient

    from app import app

    return TestClient(app), db
