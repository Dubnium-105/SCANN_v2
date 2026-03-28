# 项目文档索引

`docs/` 目录只保留当前仍在维护、需要持续查阅的设计与实现说明。

## 当前文档

- `architecture.md`
  - 当前代码结构、模块边界、主要业务流与数据层分工
- `dataset_pipeline.md`
  - 数据集预处理流程、任务生成规则与 `scann_dataset.db` 数据库设计
- `native_annotation.md`
  - 原生 FITS 标注平台的后端、前端、数据集、锁与标注链路说明

## 相关目录文档

- `../tests/README.md`
  - 自动化测试目录约定
- `../scripts/README.md`
  - 脚本目录约定
- `../../docker/README.md`
  - Docker 部署、环境变量和 CI/CD

## 维护规则

- `docs/` 中只放长期有效的“当前实现说明”
- 一次性的计划、backlog、阶段总结、临时设计草案不要再放回这里
- 如果某份内容只服务于一次开发任务，应优先放到 issue、PR 或外部任务系统
