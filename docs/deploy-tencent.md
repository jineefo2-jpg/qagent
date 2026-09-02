# 腾讯云部署手册（单人自用，SSH 隧道访问，无需备案）

## 0. 前提（Mac 上，一次性）
- [ ] **轮换凭据**（上云硬前提）：DeepSeek key / Google & GitHub OAuth secret / SMTP 密码 /
      `APP_SECRET_KEY`。写进本地 `.env`（`TUSHARE_TOKEN`、`AGENT_ACCESS_KEY` 不用换）。
- [ ] `git push`（代码走 git；数据走 rsync，镜像不含数据与密钥 —— 见 .dockerignore）

## 1. 买服务器（腾讯云控制台）
- **轻量应用服务器**即可：**4 核 8G、100GB SSD**（预留数据增长；2 核 4G 能跑但重建/回测吃力）
- 地域选国内（上海/广州，Tushare 直连快）；镜像选 **Ubuntu 24.04**
- 防火墙（安全组）：**只放行 22 端口**。5000 不对公网开 —— 用 SSH 隧道访问，
  这正是 compose 里端口只绑 127.0.0.1 的原因，也因此**不涉及域名备案**

## 2. 服务器初始化（ssh 进去后，逐条跑）
```bash
# 装 docker（官方脚本，国内机器会自动走镜像源）
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER && exit
# 重新 ssh 进来使 docker 组生效，然后：
git clone git@github.com:jineefo666/quantagent.git ~/quantagent   # 或 https + token
cd ~/quantagent && mkdir -p data out logs rag_db
```
（git clone 需要在服务器上配 deploy key，或用 https 私有仓库 token —— 控制台里生成）

## 3. 传数据 + 密钥（Mac 上）
```bash
scripts/sync_data_to_cloud.sh ubuntu@<服务器IP>     # ~4.3GB，家宽上行约 10-30 分钟
scp .env ubuntu@<服务器IP>:quantagent/.env          # 密钥单独传，不进任何脚本/镜像
```

## 4. 构建并启动（服务器上）
```bash
cd ~/quantagent && docker compose up -d --build     # 首次构建 10-20 分钟（torch CPU 版）
docker compose ps                                    # 三个服务 Up 即成
docker compose logs app --tail 20                    # 看到 Uvicorn running 即成
```

## 5. 访问（Mac 上，日常就这一条）
```bash
ssh -N -L 8080:127.0.0.1:5000 ubuntu@<服务器IP>
# 浏览器开 http://localhost:8080 —— 登录方式不变（访问密钥 + 邮箱）
```

## 6. 验证每日更新（部署当晚）
- 容器内 cron 工作日 21:30（镜像已钉 Asia/Shanghai）自动跑，**云上无 TCC、IP 固定，
  本地折磨我们的两大问题（权限 + IP 轮换）在云上天然不存在**
- 次日早：`docker compose exec cron tail -5 /app/logs/ashare_daily.log` 看到 `daily OK` 即闭环
- 周五晚会自动多一步 nightly-signal：出下周清单、落 ledger、导出 out/signals/

## 7. 割接后本地怎么办
- 本地 cron 定时器建议停掉（`crontab -e` 注释那两行），避免两边同时更新各自的库产生两套事实
- 本地库保留当备份；重大操作前云上 `data/` 打包下载一份即可

## 已知注意项
- **两边不要同时跑每日更新**：数据是两份独立副本，会各自漂移。割接 = 云上跑通当晚就停本地的
- Docker 镜像 ~4-5GB（torch），首次构建慢是正常的
- 服务器重启后 compose 会自动拉起（restart: unless-stopped）；@reboot 补漏由容器 cron 的
  RunAtLoad 等价物（catch-up pass 任意时刻安全）保证
