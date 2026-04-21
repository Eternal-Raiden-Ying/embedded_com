# 开发板环境备份与 Google Drive 使用说明

## 1. 这份说明是给谁看的
这份文档面向后续会继续使用这块 AidLux 开发板的队友，目标是把下面几件事讲清楚：

- 代码应该如何管理
- 环境备份应该如何管理
- 为什么环境备份不能散装上传
- Google Drive 为什么需要代理才能用
- 新队友第一次接手这块板子时，应该先做什么

一句话原则：

`项目代码用 Git 管理，板端环境备份用 Google Drive 管理。`

这两件事情不要混在一起做。

## 2. 代码与环境的分工原则

### 2.1 代码管理
项目代码、脚本、文档、配置模板，统一放在 Git 仓库中管理。

当前主工程目录就是：

`/home/aidlux/embedded_com`

像下面这些内容适合进 Git：

- Python / C++ 源码
- 启动脚本
- 文档
- 配置模板
- 测试脚本
- 备份脚本模板

### 2.2 环境管理
底层环境、运行库、服务配置、授权文件、板端运行时快照，这些不适合用 Git 管理，统一通过本地归档加 Google Drive 备份。

像下面这些内容适合走环境备份：

- `/opt/aidlux/cpf`
- `/usr/local/lib/libaidlite.so`
- `/usr/local/lib/libaidlms.so`
- `/usr/local/lib/libaidrtcm.so*`
- `/usr/local/lib/python3.8/dist-packages/aidlite*`
- `/etc/systemd/system/aidrtcm.service`
- `/etc/systemd/system/aid-lms.service`
- `/usr/local/share/aidlite/examples`
- `/sdcard/Documents/AidLuxLics`

## 3. 为什么环境备份不能散装上传
不要把 `/opt`、`/usr/local/lib`、`/etc/systemd/system` 里的文件一个一个复制到云盘。

原因有三个：

1. 权限容易丢失
2. 软链接容易断裂
3. 目录结构容易被人为破坏

正确做法是：

- 先在板子本地生成一个完整备份目录
- 核心运行时内容必须打成 `tar.gz`
- 同时生成 `sha256` 校验文件
- 然后把整个备份目录上传到 Google Drive

## 4. 推荐的板端目录排布
为了让代码、备份、模型、临时文件更清楚，推荐在板子上长期维持下面这种结构。

```text
/home/aidlux/
  embedded_com/                 # Git 主工程目录
  board_backups/                # 各次环境备份目录
  board_restore_notes/          # 恢复说明、版本对应、人工笔记
  board_tools/                  # 非 Git 的板端辅助工具、临时下载工具
  downloads/                    # 临时下载目录，可定期清理
```

进一步建议：

- `embedded_com/` 只放项目代码和项目文档，不放大体积环境归档
- `board_backups/` 专门放 `board_backup_时间戳/`
- `board_restore_notes/` 可以放人工维护的恢复经验、踩坑记录
- `downloads/` 用来放临时 zip、deb、rclone 下载包，避免污染工程目录

## 5. 推荐的 Google Drive 目录结构
建议在 Google Drive 里固定建立下面几个目录：

```text
AidLuxBoardBackups/
  baseline/
  before_change/
  logs/
  restore_notes/
```

含义如下：

- `baseline/`
  存放当前确认可用、可以作为恢复基线的环境包
- `before_change/`
  每次升级、换库、调整运行环境前的快照
- `logs/`
  存放备份日志、服务状态、测试记录
- `restore_notes/`
  存放恢复说明、版本对应关系、注意事项

## 6. 当前这块板子的已知经验

### 6.1 Git 和 Google Drive 的边界
推荐做法：

- `Git` 只管理工程代码
- `Google Drive` 管理环境备份与运行时快照

不推荐做法：

- 把几十 MB 到几百 MB 的运行时归档直接塞进 Git
- 把系统文件零散上传到云盘

### 6.2 这块板子访问 Google Drive 需要代理
这块板子自己默认不能直接访问 Google API。

已经验证过的可用路径是：

- Windows 电脑开启 SakuraCat
- Windows 本机代理端口为 `7897`
- 开发板通过 Windows 局域网 IP 访问这个代理
- `rclone` 通过代理访问 Google Drive

例如本次验证里：

- Windows IP：`192.168.55.215`
- 代理端口：`7897`

开发板上临时设置代理的方式：

```bash
export HTTP_PROXY=http://192.168.55.215:7897
export HTTPS_PROXY=http://192.168.55.215:7897
export ALL_PROXY=http://192.168.55.215:7897
```

如果 Windows 的局域网 IP 变化了，需要一起改。

### 6.3 VSCode Remote 代理不等于板子全局代理
这个坑非常重要。

即使 VSCode Remote-SSH 本身配置了代理，远端板子里的 `curl`、`rclone`、`wget` 也不一定会自动走代理。

所以凡是板子上主动访问 Google 的命令，都建议先显式设置：

```bash
export HTTP_PROXY=http://<Windows_IP>:7897
export HTTPS_PROXY=http://<Windows_IP>:7897
export ALL_PROXY=http://<Windows_IP>:7897
```

## 7. 第一次接手板子时建议先做的事情

### 7.1 检查代码仓库
确认主工程目录：

```bash
cd /home/aidlux/embedded_com
git status
```

### 7.2 检查是否有历史备份目录
建议统一放在：

```bash
ls /home/aidlux/board_backups
```

如果现在还没有这个目录，建议创建：

```bash
mkdir -p /home/aidlux/board_backups
mkdir -p /home/aidlux/board_restore_notes
mkdir -p /home/aidlux/board_tools
mkdir -p /home/aidlux/downloads
```

### 7.3 检查代理是否可用
如果要使用 Google Drive，先确认 Windows 端代理可用。

在板子上测试：

```bash
curl --proxy http://<Windows_IP>:7897 -I https://www.googleapis.com
```

如果能返回 HTTP 响应头，说明代理链路是通的。

## 8. rclone 的推荐使用方式

### 8.1 推荐远端名
Google Drive 的 `rclone` remote 建议统一叫：

```text
gdrive
```

这样队友之间命令统一，后续文档也统一。

### 8.2 检查当前配置

```bash
rclone config show gdrive
```

### 8.3 每次使用前的推荐动作
先设置代理：

```bash
export HTTP_PROXY=http://<Windows_IP>:7897
export HTTPS_PROXY=http://<Windows_IP>:7897
export ALL_PROXY=http://<Windows_IP>:7897
```

再测试：

```bash
rclone lsd gdrive:
```

如果这一步能列目录，再开始上传。

## 9. 推荐的环境备份工作流

### 9.1 什么时候备份
建议在这些时间点做备份：

- 首次把板子调通后
- 升级 AidLux 相关组件前
- 更换底层运行库前
- 调整授权、service、模型运行环境前
- 任何“不确定会不会把环境搞坏”的操作前

### 9.2 备份命名
统一使用时间戳目录，例如：

```text
/home/aidlux/board_backups/board_backup_2026-04-20_190811
```

### 9.3 备份脚本
当前工程里已经有一份可以参考和复用的脚本：

[`backup_board_env.sh`](/home/aidlux/embedded_com/backup_board_env.sh)

还有一份更偏“云端归档流程化”的模板脚本：

[`backup_board_env_cloud_ready.sh`](/home/aidlux/board_backup_2026-04-20_190811/scripts/backup_board_env_cloud_ready.sh)

### 9.4 备份完成后怎么上传
如果上传整个目录：

```bash
rclone copy /home/aidlux/board_backup_YYYY-MM-DD_HHMMSS gdrive:AidLuxBoardBackups/before_change/board_backup_YYYY-MM-DD_HHMMSS -P
```

如果上传总压缩包和校验值：

```bash
rclone copy /home/aidlux/board_backup_YYYY-MM-DD_HHMMSS/board_backup_YYYY-MM-DD_HHMMSS_upload.tar.gz gdrive:AidLuxBoardBackups/before_change -P
rclone copy /home/aidlux/board_backup_YYYY-MM-DD_HHMMSS/board_backup_YYYY-MM-DD_HHMMSS_upload.tar.gz.sha256 gdrive:AidLuxBoardBackups/before_change -P
```

推荐做法：

- Google Drive 上传“整个目录”
- 同时再上传“总压缩包 + 校验文件”

这样信息最完整，恢复也最方便。

## 10. 推荐的项目内文档排布
建议把“代码文档”和“环境文档”稍微分开。

当前仓库根目录推荐长期保留这些文件：

```text
README.md
BOARD_ENV_BACKUP_README_CN.md
SYSTEM_UPGRADE_DELIVERY.md
LOG_STANDARD.md
backup_board_env.sh
```

如果后续继续扩展，建议再增加一个文档目录：

```text
docs/
  board_env/
    backup_strategy.md
    restore_strategy.md
    google_drive_usage.md
```

如果你不想新建太多目录，至少保留当前这份中文说明即可。

## 11. 不建议做的事情

- 不要把环境备份目录直接提交到 Git
- 不要把 `/opt`、`/usr/local/lib` 的文件逐个上传
- 不要只传压缩包、不传校验文件
- 不要默认认为 VSCode 能连上板子，就代表板子自己能访问 Google
- 不要在没有备份的情况下直接升级底层库

## 12. 推荐的长期操作习惯

### 代码变更
走 Git：

```bash
git status
git add
git commit
git push
```

### 环境变更
走本地备份 + Google Drive：

```bash
1. 本地生成备份目录
2. 校验压缩包
3. 上传 Google Drive
4. 在 restore_notes 里补一句这次变更目的
```

### 交接给队友
建议直接告诉对方：

1. 代码看 Git 仓库
2. 环境看 Google Drive 里的 `AidLuxBoardBackups`
3. 板子访问 Google 要先配代理
4. 先看这份文件，再操作环境

## 13. 一套推荐的日常命令模板

### 13.1 设置代理
```bash
export HTTP_PROXY=http://<Windows_IP>:7897
export HTTPS_PROXY=http://<Windows_IP>:7897
export ALL_PROXY=http://<Windows_IP>:7897
```

### 13.2 测试 Google API
```bash
curl --proxy http://<Windows_IP>:7897 -I https://www.googleapis.com
```

### 13.3 测试 Google Drive
```bash
rclone lsd gdrive:
```

### 13.4 上传一个备份目录
```bash
rclone copy /home/aidlux/board_backup_YYYY-MM-DD_HHMMSS gdrive:AidLuxBoardBackups/before_change/board_backup_YYYY-MM-DD_HHMMSS -P
```

### 13.5 上传一个基线包
```bash
rclone copy /home/aidlux/board_backup_YYYY-MM-DD_HHMMSS gdrive:AidLuxBoardBackups/baseline/board_backup_YYYY-MM-DD_HHMMSS -P
```

## 14. 最后的建议
如果是项目代码出问题，优先看 Git。
如果是板端运行环境出问题，优先看最近一次环境备份。

把“代码版本”和“环境版本”分开管理，是这块板子长期稳定维护的关键。
