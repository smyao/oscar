# 当前工程约束

先读 `oscar_ascend_agent_start.md`，按其档案引用制度执行。`issue1_full_record.md` 实际为156条（G1–G34、#1–#122），按标题定位；旧标题条数与旧行号存在漂移。

- 只允许参考本项目 `references/` 内指定PR和原生实现；禁止读任何失败OSCAR项目代码。所有参考树只读，不运行格式化或修改。
- 适配代码为外部插件与独立AscendC工程，不能覆盖原生vLLM/Ascend源码，不使用 `cp` 部署。
- 配置以本次 `configs/target.json` 为准；devices尚未确定时不启动NPU。不能从旧项目记忆或已有环境继承其他任务卡号。
- `docs/checklist.md` 是42个稳定checkpoint；源码、CPU、CANN、设备完成、图捕获、图回放、质量、性能分别记录。
- CV/history/window、production RuntimeProvider、物理页精确快照、MTP位置修正和全服务probe已有实现。CANN VM编译与官方CPU-debug通过；真实NPU/图/性能未验收，readiness或编译成功不能冒充这些证据。
- `ops/reference.py` 是独立测试oracle，绝不加入生产路由。精度冻结于 `configs/acceptance.json`，不能事后放宽。
- 凡修改CANN算子，先亲读档案同类条目与D.4完整原始打点，文件头加入档案引用及D.4四问。
- 新真机错误才追加档案；本机CPU测试或理论判断不能伪写为真机记录。
- 用户已授权完成后推送至 https://gitcode.com/jpl123/gpt_new_oscar.git；只推送本工程，保留真实验收边界。

本地验证：`.venv/bin/python -m pytest -q`。NPU probe是 `python -m tools.deploy --only probe-ops`，无NPU时必须失败，不能用CPU替代。
