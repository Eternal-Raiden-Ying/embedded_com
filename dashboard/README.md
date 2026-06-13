# 智能语音取物机器人控制台 (Robot Dashboard)

本目录包含智能语音取物机器人项目的静态工程 Dashboard，用于将分布于各目录的架构、配置、启动排查和测试说明整合成一个离线可浏览、可点击、可搜索的可视化网页控制台。

---

## 1. 目录结构

```text
docs/dashboard/
├── index.html             # 仪表盘主页面入口 (双击即可在浏览器中本地打开)
├── README.md              # 本说明文档
├── build_dashboard.py     # 自动扫描并提取 Markdown 文档大纲生成索引的 Python 脚本
├── assets/
│   ├── style.css          # 支持深色/浅色自适应的高级工程样式表
│   └── app.js             # 驱动搜索、标签页切换、剪贴板复制和抽屉折叠的核心逻辑
└── data/
    ├── modules.js         # 架构分层和系统域关联数据 (JS 变量封装，用于本地离线加载)
    ├── modules.json       # 同上 (标准 JSON 格式)
    ├── config_matrix.js   # 核心参数配置及危险值校验规约 (JS 变量)
    ├── config_matrix.json # 同上 (标准 JSON 格式)
    ├── troubleshooting.js # 6 大核心故障场景排查排障与恢复 (JS 变量)
    ├── troubleshooting.json# 同上 (标准 JSON 格式)
    ├── docs_manifest.js   # 扫描生成的 Markdown 目录及大纲数据 (JS 变量)
    ├── docs_manifest.json # 同上 (标准 JSON 格式)
    ├── module_pages.js    # 按模块组织编译生成的正文 HTML 集合 (JS 变量)
    ├── module_pages.json  # 同上 (标准 JSON 格式)
    ├── raw_docs.js        # 完整原始 Markdown 转 HTML 的归档数据 (JS 变量)
    └── raw_docs.json      # 同上 (标准 JSON 格式)
```

---

## 2. 离线访问与功能特性

### 2.1 双击直接打开
由于现代浏览器对 `file:///` 协议下 `fetch()` 本地文件的 CORS 跨域限制，本仪表盘采用了**双通道数据加载模式**：
* 网页直接使用标准 `<script>` 注入变量形式加载 `data/*.js` 缓存数据，**不需要搭建任何本地 HTTP 服务器**。
* 双击 `index.html` 即可离线启动，数据渲染与检索响应在微秒级完成。

### 2.2 核心功能
* **左侧导航栏**：可在项目概览、系统架构、启动与 IPC 通信、配置层级矩阵、故障排障、单元测试及参考文档大纲之间平滑切换。
* **快捷按钮与主题**：顶部提供了常用页面的一键跳转，以及 Light (浅色) / Dark (深色，默认) 模式自适应切换（主题选项会自动记录在浏览器本地 `localStorage` 中）。
* **一键复制命令**：所有的排障、启动和测试命令框右侧均提供了“Copy”按钮，一键复制命令至剪贴板。
* **关键词检索**：在顶部搜索框输入文字（如：`UDS`、`dry_run`、`stale` 等），系统会自动高亮并过滤符合条件的配置参数、排障面板和文档大纲卡片。

---

## 3. 重新生成文档索引

当您修改了项目中的 Markdown 文档，或者新增了文档后，可以通过以下命令重新扫描并生成 `data/docs_manifest.json` 与 `data/docs_manifest.js`：

```powershell
# 在项目根目录下执行：
D:\anaconda\Anaconda\envs\embed_sc171\python.exe docs/dashboard/build_dashboard.py
```

该脚本将自动扫描：
1. 根目录下的 `*.md`
2. `docs/` 目录下的所有 `*.md`
3. `orchestrator/` 目录下的所有 `*.md`
4. `VISTA/` 目录下的所有 `*.md`
5. `tools/` 目录下的所有 `*.md`

并自动解析首行标题和各 `##` 与 `###` 小节，重构 Dashboard 导航内的 Reference 菜单。

---

## 4. 重新生成图片资源与清单

若更新了原始图片素材（放置在 `docs/pictures/original/` 目录中），或希望调整图片裁剪、美化增强和手机 mockups 边框参数，可以执行图片资源构建脚本：

```powershell
# 在项目根目录下执行：
D:\anaconda\Anaconda\envs\embed_sc171\python.exe docs/dashboard/build_picture_assets.py
```

该脚本会执行以下处理：
1. 自动校正原始照片的 EXIF 旋转角。
2. 将 AI 架构图等图纸以 16:9 白色画板进行居中补白。
3. 自动对实拍照片进行适度亮度、对比度与锐度美化增强，并裁剪为 16:9 WebP 图像。
4. 将微信小程序截图自动缩放并嵌套至灰色手机物理 Mockup 壳内。
5. 自动为所有图片生成 640x360 WebP 缩略图。
6. 更新 `docs/pictures/meta/pictures_manifest.json` 和 `docs/dashboard/data/pictures_manifest.js` 图片清单，供 Dashboard 进行前端渲染和 fallback 回退展示。
