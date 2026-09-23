# 当前工程约束

先读 `oscar_ascend_agent_start.md`，按其档案引用制度执行。`issue1_full_record.md` 实际为173条（G1–G34、#1–#139），按标题定位；旧标题条数与旧行号存在漂移。

- 只允许参考本项目 `references/` 内指定PR和原生实现；禁止读任何失败OSCAR项目代码。所有参考树只读，不运行格式化或修改。
- 适配代码为外部插件与独立AscendC工程，不能覆盖原生vLLM/Ascend源码，不使用 `cp` 部署。
- 配置以本次 `configs/target.json` 为准：按本次启动文档附录 A/F 默认物理设备 `0,1,2,3`、SOC `ascend910b4`、端口 `8989`。不能从旧项目记忆或已有环境继承其他任务卡号；显式配置非法时仍报错。
- 用户最新要求：保留真实 NPU 算子、CV/旋转和完整服务探针；移除默认部署中的环境清单、原生源码扫描、readiness 前置审计。正常流程为安装/编译/探针/自动旋转文件/服务探针/资源清理/正式服务，任何探针失败不得继续。该要求优先于启动文档旧的环境审计步骤。
- 部署阶段及服务子进程的输出和错误必须实时打印到当前终端，同时保留相位日志、状态和真实退出码；不能只把 traceback 写入文件后让用户手工查找（档案 #94/#95/#125）。
- `docs/checklist.md` 是42个稳定checkpoint；源码、CPU、CANN、设备完成、图捕获、图回放、质量、性能分别记录。
- CV/history/window、production RuntimeProvider、物理页精确快照、MTP位置修正和全服务probe已有实现。CANN VM编译与官方CPU-debug通过；node93已通过75项store/merge原语真NPU探针，后续日志已进入TP4/MTP模型加载，最新已完成编译/初次profiling与KV预算，后续已通过health与128→16短请求；后续128/16K/32K/50K串行请求均已完成；混合并发已现300s请求死线超时失败（心跳证明非死锁、为时间预算问题），性能未验收（#130/#131）；已支持none/align布局并区分物理存储与metadata128虚拟块，完整NPU/图/性能未验收，readiness或编译成功不能冒充这些证据。
- `ops/reference.py` 是独立测试oracle，绝不加入生产路由。精度冻结于 `configs/acceptance.json`，不能事后放宽。
- 凡修改CANN算子，先亲读档案同类条目与D.4完整原始打点，文件头加入档案引用及D.4四问。
- 新真机错误才追加档案；本机CPU测试或理论判断不能伪写为真机记录。
- 用户已授权完成后推送至 https://gitcode.com/jpl123/gpt_new_oscar_kimi.git；只推送本工程，保留真实验收边界。

真机部署只需 `git pull --ff-only && bash scripts/install_probe_serve.sh`，使用当前 Python 环境。`.venv/bin/python -m pytest -q` 仅用于已配置虚拟环境的本地开发机；`bash scripts/validate_vm.sh` 仅用于带 Lima 的 Mac 宿主机，均不是真机前置步骤。单独 NPU probe 是 `python3 -m tools.deploy --only probe-ops`，无 NPU 时必须失败，不能用 CPU 替代。
