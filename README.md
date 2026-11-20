# 校园快递驿站智能服务系统

《人工智能》课程综合实践项目源码。四个算法模块 + Flask 单页系统，CPU 即可运行。

| 模块 | 内容 |
|---|---|
| 一 | 工单分类：字符 n-gram TF-IDF + 逻辑回归 / 线性 SVM |
| 二 | 用户分群：KMeans（肘部法 + 轮廓系数）+ PCA |
| 三 | 到件量预测：15 维特征 + 线性回归 / MLP + PCA |
| 四 | 政策问答：TF-IDF 检索条款 + 大模型（零样本 / 少样本 / 思维链） |
| 五 | 系统集成：Flask JSON 接口 + 原生 JS 单页界面 |

## 快速开始

环境：Python 3.10+，无需 GPU。

**方式一：双击 `start.bat`**（Windows）
自动检查 Python 与依赖、模型不匹配时重训、启动服务并打开浏览器；`start.bat stop` 停止。

**方式二：手动**

```bash
pip install -r requirements.txt
python src/run_all.py          # 生成数据 → 清洗 → 训练 → 评估（约 30 秒）
python src/app.py --port 7860  # 浏览器打开 http://127.0.0.1:7860
```

> 换电脑后不要跳过 `run_all.py`：模型文件与训练环境的 scikit-learn 版本绑定，重训一次即可避免加载失败。异常时运行 `python tools/doctor.py` 查看逐模型诊断。

## 目录结构

```
src/       主流程：数据生成、清洗、四个模块、评估、Flask 服务
tools/     启动器、自检、清理、评分汇总、审计脚本
data/      原始数据 / 清洗后数据 / 政策知识库
models/    训练产物（joblib）
outputs/   图表与指标产物（metrics 下的 JSON/CSV）
start.bat  Windows 一键启动
```

## 可选：接入大模型

不配 Key 也能完整运行——问答模块自动降级为知识库检索，界面标注「离线降级」。

```bash
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY（也支持 OPENAI_API_KEY / ZHIPUAI_API_KEY）
```

## 常见问题

- **端口被占用**：`python src/app.py --port 7861`
- **后台运行**：`powershell -File tools/server.ps1 start|stop|status`
- **依赖装不上**：`pip install -r requirements-lock.txt` 使用验证过的精确版本

## 许可证

[MIT License](LICENSE)。数据与知识库均为虚构内容，仅用于课程演示。
