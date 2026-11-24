# 交付验证记录

验证时间：2026-08-04 17:33 CST

- 源码测试：`124` 项全部通过。
- bbox 契约：模型 schema、业务对象、裁剪边界与序列化均通过整数 `[0,1000]` 测试。
- 安装验证：wheel 已在隔离 Python 3.12 环境中以 `--no-deps` 安装成功，并从系统环境读取 Pillow。
- CLI 验证：`u1-critic-v41-run`、`u1-critic-v41-report`、`u1-critic-v41-serve`、`u1-critic-v41-compare` 均可执行 `--help`。
- 导入验证：`u1_critic_v41.__version__ == "4.1.0"`，管线构建入口可导入。
- 内容检查：源码交付目录不含 API Key、内部服务地址、图片、运行结果、错误结果或 trace。
- 脚本检查：`examples/run_v41.sh` 通过 Bash 语法检查。
- 包完整性：wheel 可读取且包含 `u1_critic_v41` 与 `critic_workflows` 两个运行包；最终文件校验见 `SHA256SUMS`。

以上验证不调用线上模型，因此不消耗推理服务，也不代表接收方目标模型接口已经兼容；首次接入仍需按 README 进行单图冒烟。
