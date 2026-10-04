# ShopCare Backend

`backend/` 是 ShopCare 的 Python/FastAPI 后端工程，包含 API、领域 Agent、意图识别、记忆、知识库、工具、脚本和测试。

## 本地运行

在仓库根目录执行：

```powershell
.venv\\Scripts\\python.exe -m uvicorn api.main:app --app-dir backend --reload
```

或进入后端目录执行：

```powershell
Set-Location backend
..\\.venv\\Scripts\\python.exe -m uvicorn api.main:app --reload
```

## 测试与知识库初始化

根目录的 `pytest.ini` 将测试路径指向 `backend/tests/`，因此可直接执行：

```powershell
.venv\\Scripts\\python.exe -m pytest -q
.venv\\Scripts\\python.exe backend/scripts/init_knowledge_base.py
```

后端本地运行时数据默认位于 `backend/.data/` 或 `backend/data/chroma/`，本地模型位于 `backend/models/`；这些目录均不纳入 Git。
