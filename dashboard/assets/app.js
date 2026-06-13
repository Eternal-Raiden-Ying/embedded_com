document.addEventListener('DOMContentLoaded', () => {
  // 1. Theme Management (Light Mode Default for V2 slate theme)
  const themeToggleBtn = document.getElementById('themeToggle');
  const body = document.body;
  
  const currentTheme = localStorage.getItem('theme') || 'light';
  body.setAttribute('data-theme', currentTheme);
  updateThemeIcon(currentTheme);

  themeToggleBtn.addEventListener('click', () => {
    const activeTheme = body.getAttribute('data-theme');
    const newTheme = activeTheme === 'dark' ? 'light' : 'dark';
    body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
  });

  function updateThemeIcon(theme) {
    if (theme === 'light') {
      themeToggleBtn.innerHTML = `
        <svg viewBox="0 0 24 24">
          <path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707m0-12.728l.707.707m11.314 11.314l.707.707M12 8a4 4 0 100 8 4 4 0 000-8z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
      `;
    } else {
      themeToggleBtn.innerHTML = `
        <svg viewBox="0 0 24 24">
          <path d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/>
        </svg>
      `;
    }
  }

  // 2. Navigation System & Nested Submenus
  const navItems = document.querySelectorAll('.nav-item');
  const submenuItems = document.querySelectorAll('.submenu-item');
  const panels = document.querySelectorAll('.section-panel');

  function switchTab(targetId) {
    // Reset selections on sidebar main items
    navItems.forEach(item => {
      const itemTarget = item.getAttribute('data-target');
      if (itemTarget === targetId) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });

    // Reset selections on submenu items
    submenuItems.forEach(subItem => {
      const subTarget = subItem.getAttribute('data-target');
      if (subTarget === targetId) {
        subItem.classList.add('active');
        // Auto-highlight parent menu
        const parentId = subItem.getAttribute('data-parent');
        const parentNav = document.querySelector(`.nav-item[data-target="${parentId}"]`);
        if (parentNav) parentNav.classList.add('active');
      } else {
        subItem.classList.remove('active');
      }
    });

    // Toggle panels visibility
    panels.forEach(panel => {
      if (panel.id === targetId) {
        panel.classList.add('active');
      } else {
        panel.classList.remove('active');
      }
    });

    // Update context rail info
    updateContextRail(targetId);
    
    // Smooth scroll main layout to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // Bind Sidebar main tabs
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const target = item.getAttribute('data-target');
      
      // If item has a submenu, toggle visibility but also enter it
      const hasSubmenu = item.nextElementSibling && item.nextElementSibling.classList.contains('sidebar-submenu');
      if (hasSubmenu) {
        // Toggle display of next submenu
        const submenu = item.nextElementSibling;
        const isCollapsed = window.getComputedStyle(submenu).display === 'none';
        submenu.style.display = isCollapsed ? 'flex' : 'none';
      }
      
      switchTab(target);
    });
  });

  // Bind Sidebar subpage items
  submenuItems.forEach(subItem => {
    subItem.addEventListener('click', (e) => {
      e.stopPropagation();
      const target = subItem.getAttribute('data-target');
      switchTab(target);
    });
  });

  // Bind Quick Access shortcut buttons on Overview
  const shortcutMappings = {
    'btn-qa-sc171': 'sc171',
    'btn-qa-vista': 'vista',
    'btn-qa-orch': 'orchestrator',
    'btn-qa-chassis': 'chassis',
    'btn-qa-arm': 'arm',
    'btn-qa-grasp': 'cloud_grasp',
    'btn-qa-startup': 'startup',
    'btn-qa-config': 'config'
  };

  Object.entries(shortcutMappings).forEach(([btnId, targetTab]) => {
    const btn = document.getElementById(btnId);
    if (btn) {
      btn.addEventListener('click', () => switchTab(targetTab));
    }
  });

  // Bind Domain quick access cards on Overview
  const domainMappings = {
    'dom-card-user': 'mobile_gateway',
    'dom-card-edge': 'sc171',
    'dom-card-execution': 'chassis',
    'dom-card-cloud': 'cloud_grasp'
  };

  Object.entries(domainMappings).forEach(([cardId, targetTab]) => {
    const card = document.getElementById(cardId);
    if (card) {
      card.addEventListener('click', () => switchTab(targetTab));
      card.style.cursor = 'pointer';
    }
  });

  // 3. Render Dashboard Data elements
  renderOverviewStatus();
  renderArchitectureMatrix();
  renderModuleEmbeddedHTML();
  renderConfigMatrix();
  renderTroubleshooting();
  renderTesting();
  renderReferenceDocs();
  renderDashboardPictures();
  initLightbox();

  // Draw Dynamic SVG line graphs
  drawSVGFrequencyGraph();

  // Load default context rail
  updateContextRail('overview');

  // 4. One-Click Copy Action for code-blocks
  document.addEventListener('click', (e) => {
    const copyBtn = e.target.closest('.copy-btn');
    if (copyBtn) {
      const codeEl = copyBtn.nextElementSibling.querySelector('code');
      const textToCopy = codeEl ? codeEl.innerText.trim() : copyBtn.nextElementSibling.innerText.trim();
      
      navigator.clipboard.writeText(textToCopy).then(() => {
        copyBtn.innerText = 'Copied!';
        copyBtn.style.backgroundColor = 'var(--success)';
        copyBtn.style.color = '#ffffff';
        
        setTimeout(() => {
          copyBtn.innerText = 'Copy';
          copyBtn.style.backgroundColor = '';
          copyBtn.style.color = '';
        }, 2000);
      }).catch(err => {
        console.error('Failed to copy text: ', err);
      });
    }
  });

  // 5. Context Rail dynamic update
  function updateContextRail(sectionId) {
    const railSummary = document.getElementById('railSummary');
    const railInfoList = document.getElementById('railInfoList');
    const railModuleBadge = document.getElementById('railModuleBadge');
    
    if (!railSummary || !railInfoList || !railModuleBadge) return;
    railInfoList.innerHTML = '';
    
    const contextMap = {
      'overview': {
        tag: 'Overview',
        summary: '机器人控制中心。支持查看设备自检链路、状态流图占比、系统故障风险评估和核心自检数据。',
        details: [
          ['关键目录', 'docs/'],
          ['配置来源', 'docs/config.md'],
          ['通信接口', 'MQTT, UDS (Unix Domain Sockets), UART'],
          ['相关模块', 'SC171, Chassis Execution, Cloud Grasp'],
          ['关联图片', '系统工作闭环图, 系统组成总图'],
          ['推荐查看', 'docs/system_runbook.md']
        ]
      },
      'architecture': {
        tag: 'Architecture',
        summary: '系统二维网络架构矩阵。纵向五层与横向四个功能域在这里紧密协同。点击对应模块即可切入配置正文说明。',
        details: [
          ['关键目录', 'docs/architecture.md'],
          ['配置来源', 'common/config/'],
          ['通信接口', 'UDS (AF_UNIX) / TCP / UART'],
          ['相关模块', '用户交互/SC171/STM32/云端'],
          ['关联图片', '系统组成总图'],
          ['推荐查看', 'ROBOT_MOTION_CONTRACT.md']
        ]
      },
      'sc171': {
        tag: 'SC171 / Edge',
        summary: 'SC171 边缘控制计算中心。包含状态决策、进程守护、图像模式切换等高负荷处理流。',
        details: [
          ['关键目录', 'VISTA/, orchestrator/'],
          ['配置来源', 'common/config/schema.py'],
          ['通信接口', 'Qualcomm Edge SDK / UDS Sockets'],
          ['相关模块', 'VISTA, Orchestrator, Mobile Gateway'],
          ['关联图片', 'SC171 边缘计算封面'],
          ['推荐查看', 'docs/architecture.md']
        ]
      },
      'orchestrator': {
        tag: 'SC171 / Runtime',
        summary: 'Orchestrator 任务编排状态机详情。管理车辆搜索、停靠、接近、微调和刹车策略切换。',
        details: [
          ['关键目录', 'orchestrator/orchestrator_service/'],
          ['配置来源', 'orchestrator/configs/stage_params.yaml'],
          ['通信接口', 'task_cmd.sock / vision_obs.sock / UART'],
          ['相关模块', 'VISTA, Mobile Gateway, STM32 Chassis'],
          ['关联图片', 'Orchestrator 编排封面'],
          ['推荐查看', 'orchestrator/README.md']
        ]
      },
      'vista': {
        tag: 'SC171 / Vision',
        summary: 'VISTA 视觉算法感知服务。管理 RealSense 相机硬件与 Yolov7 模型的全周期运行。',
        details: [
          ['关键目录', 'VISTA/vision_module/'],
          ['配置来源', 'VISTA/vision_module/config/'],
          ['通信接口', 'vision_req.sock / vision_obs.sock'],
          ['相关模块', 'Orchestrator, RealSense Camera'],
          ['关联图片', 'VISTA 视觉模块封面, camera_view, preview_debug'],
          ['推荐查看', 'VISTA/ReadMe.md']
        ]
      },
      'mobile_gateway': {
        tag: 'SC171 / Gateway',
        summary: 'Mobile Gateway 小程序网关桥接服务。作为用户命令的前置准入门限，负责北向指令的鉴权解析。',
        details: [
          ['关键目录', 'docs/mobile_gateway_runbook.md'],
          ['配置来源', 'configs/gateway_config.json'],
          ['通信接口', 'MQTT Topic / task_cmd.sock / task_ack.sock'],
          ['相关模块', 'Orchestrator, Wechat Miniapp'],
          ['关联图片', 'Mobile Gateway 封面, miniapp_home, miniapp_voice'],
          ['推荐查看', 'docs/mobile_gateway_runbook.md']
        ]
      },
      'chassis': {
        tag: 'Chassis / STM32',
        summary: 'STM32 底盘运动解算及几何对齐控制。基于麦克纳姆轮三轴速度协议实现物理驱动。',
        details: [
          ['关键目录', 'ROBOT_MOTION_CONTRACT.md'],
          ['配置来源', 'configs/chassis_params.yaml'],
          ['通信接口', '/dev/ttyHS1 (115200 UART) / Heartbeat'],
          ['相关模块', 'Orchestrator, STM32F407 Board'],
          ['关联图片', '麦克纳姆底盘封面, docking_approach'],
          ['推荐查看', 'ROBOT_MOTION_CONTRACT.md']
        ]
      },
      'arm': {
        tag: 'Chassis / Arm',
        summary: '机械臂控制器。接收状态机转换到达 AT_TABLE_EDGE 后的取物指令，完成近地对准与云台抓取动作。',
        details: [
          ['关键目录', 'orchestrator/orchestrator_service/control/'],
          ['配置来源', 'configs/arm_params.yaml'],
          ['通信接口', 'RS485 Modbus RTU / AT_TABLE_EDGE signal'],
          ['相关模块', 'Orchestrator, Cloud Grasping'],
          ['关联图片', '远程抓取实验场景, 抓取流程展示'],
          ['推荐查看', 'docs/ipc_refactor_notes.md']
        ]
      },
      'cloud_grasp': {
        tag: 'Cloud / Grasp',
        summary: '云端 3D 点云处理与高维抓取算法规约。利用云端 GPU 计算高维姿态解算。',
        details: [
          ['关键目录', 'VISTA/vision_module/backend/'],
          ['配置来源', 'configs/cloud_grasp_params.json'],
          ['通信接口', 'HTTP / WebSocket JSON exchange'],
          ['相关模块', 'Orchestrator, Arm Controller'],
          ['关联图片', '抓取细节图, 抓取香蕉实验, 实验场地俯视图'],
          ['推荐查看', 'docs/system_runbook.md']
        ]
      },
      'startup': {
        tag: 'SC171 / Debug',
        summary: '系统软件栈的进程拉起顺序、UDS 套接字建立、和端口就绪的健康自检。',
        details: [
          ['关键目录', 'scripts/'],
          ['配置来源', 'start_robot_stack.sh'],
          ['通信接口', 'UDS /tmp/robot_stack/ socket checks'],
          ['相关模块', 'VISTA, Orchestrator, Mobile Gateway'],
          ['关联图片', '闭环系统工作图'],
          ['推荐查看', 'docs/system_runbook.md']
        ]
      },
      'config': {
        tag: 'SC171 / Config',
        summary: '配置的多层覆盖关系与调参矩阵。提示可能导致小车失控的危险参数限幅。',
        details: [
          ['关键目录', 'configs/'],
          ['配置来源', 'common/config/schema.py, YAML parameters'],
          ['通信接口', 'System environment override check'],
          ['相关模块', 'Orchestrator, Chassis, VISTA'],
          ['关联图片', 'SC171 封面, 底盘模块图'],
          ['推荐查看', 'docs/config.md']
        ]
      },
      'testing': {
        tag: 'Host / Testing',
        summary: '单元测试与契约测试防护库。在开发机上进行无硬件环境的安全模拟逻辑仿真。',
        details: [
          ['关键目录', 'tests/'],
          ['配置来源', 'pytest.ini, conftest.py'],
          ['通信接口', 'Mock Serial / UDS loopback logic'],
          ['相关模块', 'Developer Host Simulation / Anaconda'],
          ['关联图片', '视觉调试效果预览'],
          ['推荐查看', 'docs/testing.md']
        ]
      },
      'reference': {
        tag: 'Archive / Docs',
        summary: '项目内全量 Markdown 原始文档归档阅读中心。主要用于历史协议和大段说明的查阅。',
        details: [
          ['关键目录', 'docs/'],
          ['配置来源', 'build_dashboard.py'],
          ['通信接口', 'Browser local javascript variables'],
          ['相关模块', 'All modules documentation'],
          ['关联图片', 'All catalog images'],
          ['推荐查看', 'dashboard/README.md']
        ]
      }
    };
    
    const info = contextMap[sectionId] || contextMap['overview'];
    railSummary.innerText = info.summary;
    railModuleBadge.innerText = info.tag;
    
    railInfoList.innerHTML = info.details.map(d => `
      <div class="rail-info-row">
        <span class="rail-info-label">${d[0]}</span>
        <span class="rail-info-value">${d[1]}</span>
      </div>
    `).join('');
  }

  // Helper for image asset path conversion for local double-click CORS bypass
  function getRelativePath(path) {
    if (!path) return '';
    if (path.startsWith('pictures/')) {
      return '../' + path;
    }
    return path.replace(/^docs\//, '../');
  }

  // Global Image Error Handler to prevent broken images and render styled inline SVG fallbacks
  window.handleImgError = function(img) {
    img.onerror = null;
    const div = document.createElement('div');
    div.className = 'fallback-placeholder-svg';
    div.innerHTML = `
      <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/>
      </svg>
    `;
    if (img.parentNode) {
      img.parentNode.replaceChild(div, img);
    }
  };

  // Render processed images, banners, and screenshots under mock phone chassis
  function renderDashboardPictures() {
    const imageSlotConfig = {
      overviewHero: {
        aspectRatio: '21/9',
        maxHeight: 320,
        fit: 'cover',
        className: 'image-slot-overview-hero'
      },
      systemBanner: {
        aspectRatio: '21/9',
        maxHeight: 280,
        fit: 'contain',
        className: 'image-slot-system-banner'
      },
      moduleCover: {
        aspectRatio: '21/9',
        maxHeight: 280,
        fit: 'cover',
        className: 'image-slot-module-cover'
      },
      galleryImage: {
        aspectRatio: '4/3',
        maxHeight: 260,
        fit: 'contain',
        className: 'image-slot-gallery'
      },
      debugScene: {
        aspectRatio: '4/3',
        maxHeight: 280,
        fit: 'contain',
        className: 'image-slot-debug-scene'
      },
      phoneMockup: {
        aspectRatio: '9/19',
        maxHeight: 420,
        fit: 'contain',
        className: 'image-slot-phone'
      },
      thumb: {
        aspectRatio: '16/9',
        maxHeight: 160,
        fit: 'cover',
        className: 'image-slot-thumb'
      }
    };

    // 1. Overview Hero Banner
    const overviewHero = document.getElementById('overviewHeroContainer');
    if (overviewHero && window.PICTURES_MANIFEST) {
      const systemGlobalItem = window.PICTURES_MANIFEST.find(item => item.id === 'system_global');
      if (systemGlobalItem) {
        const heroUrl = getRelativePath(systemGlobalItem.hero || systemGlobalItem.cover);
        overviewHero.innerHTML = `
          <div class="module-hero-banner media-card" data-lightbox="true" data-title="${systemGlobalItem.title}" data-caption="${systemGlobalItem.caption || ''}" data-tag="系统全局图" style="cursor: pointer;">
            <div class="module-hero-frame" style="aspect-ratio: 21 / 9; width: 100%;">
              <img class="module-hero-img image-slot-overview-hero" src="${heroUrl}" onerror="window.handleImgError(this)" style="object-fit: contain; height: 100%; width: 100%;" alt="System Global">
            </div>
            <div class="module-hero-caption">
              <div class="module-hero-caption-title">面向视障服务的智能语音取物机器人</div>
              <div class="module-hero-desc">手机小程序唤醒词下发 ➜ MQTT网关桥接 ➜ Orchestrator状态编排与安全守护 ➜ VISTA端侧视觉定位 ➜ STM32底盘与机械臂协同执行</div>
            </div>
          </div>
        `;
      }
    }

    // 1b. Overview Secondary Diagram Card (Removed system-level diagram, but keep container rendering if needed)
    const secondaryContainer = document.getElementById('overviewSecondaryContainer');
    if (secondaryContainer) {
      secondaryContainer.style.display = 'none'; // hide it since system loop moved to architecture
    }

    // 1c. Architecture Hero Banner
    const architectureHero = document.getElementById('architectureHeroContainer');
    if (architectureHero && window.PICTURES_MANIFEST) {
      const systemOverviewItem = window.PICTURES_MANIFEST.find(item => item.id === 'overview');
      if (systemOverviewItem) {
        const heroUrl = getRelativePath(systemOverviewItem.hero || systemOverviewItem.cover);
        architectureHero.innerHTML = `
          <div class="module-hero-banner media-card" data-lightbox="true" data-title="${systemOverviewItem.title}" data-caption="${systemOverviewItem.caption || ''}" data-tag="系统组成总图" style="cursor: pointer;">
            <div class="module-hero-frame" style="aspect-ratio: 21 / 9; width: 100%;">
              <img class="module-hero-img image-slot-overview-hero" src="${heroUrl}" onerror="window.handleImgError(this)" style="object-fit: contain; height: 100%; width: 100%;" alt="System Overview">
            </div>
            <div class="module-hero-caption">
              <div class="module-hero-caption-title">系统组成总图</div>
              <div class="module-hero-caption-desc">${systemOverviewItem.caption || ''}</div>
            </div>
          </div>
        `;
      }
    }

    // 2. Module Headers
    const headerMounts = document.querySelectorAll('.module-header-mount');
    headerMounts.forEach(element => {
      const modId = element.getAttribute('data-module');
      if (!modId || !window.ROBOT_MODULES) return;

      const moduleDetailsMap = {
        'orchestrator': {
          title: 'Orchestrator 任务状态机',
          imgId: 'system_loop',
          responsibility: window.ROBOT_MODULES.layers[1].responsibility,
          inputs: window.ROBOT_MODULES.layers[1].inputs,
          outputs: window.ROBOT_MODULES.layers[1].outputs,
          directories: window.ROBOT_MODULES.layers[1].directories
        },
        'vista': {
          title: 'VISTA 视觉算法服务',
          imgId: 'vista',
          responsibility: window.ROBOT_MODULES.layers[2].responsibility,
          inputs: window.ROBOT_MODULES.layers[2].inputs,
          outputs: window.ROBOT_MODULES.layers[2].outputs,
          directories: window.ROBOT_MODULES.layers[2].directories
        },
        'mobile_gateway': {
          title: 'Mobile Gateway 网关服务',
          imgId: 'mobile_gateway',
          responsibility: window.ROBOT_MODULES.layers[0].responsibility,
          inputs: window.ROBOT_MODULES.layers[0].inputs,
          outputs: window.ROBOT_MODULES.layers[0].outputs,
          directories: window.ROBOT_MODULES.layers[0].directories
        },
        'chassis': {
          title: 'STM32 麦克纳姆底盘驱动',
          imgId: 'chassis',
          responsibility: '物理速度指令三轴解算与执行反馈，配合物理急停与离散微占空比插帧降低最低平均速。',
          inputs: window.ROBOT_MODULES.layers[4].inputs,
          outputs: window.ROBOT_MODULES.layers[4].outputs,
          directories: window.ROBOT_MODULES.layers[4].directories
        },
        'arm': {
          title: '机械臂控制模块',
          imgId: 'stm32_execution',
          responsibility: '接收状态机转换到达 AT_TABLE_EDGE 后的取物指令，完成多自由度机械臂接近并完成取物精准抓取动作。',
          inputs: '物理状态机到达临边信号 (AT_TABLE_EDGE)',
          outputs: 'Modbus 关节动作指令 & 夹爪抓取控制字节',
          directories: ['orchestrator/orchestrator_service/control/', 'runtime/states/grasp_flow.py']
        },
        'cloud_grasp': {
          title: '云端 3D 抓取 (Cloud Grasp)',
          imgId: 'sc171',
          responsibility: window.ROBOT_MODULES.domains[3].responsibility,
          inputs: 'SC171 (RealSense RGB-D 帧上传)',
          outputs: '云端 3D 抓取规避碰撞姿态解算 (GR-ConvNet / AnyGrasp 3D)',
          directories: ['docs/cloud_grasp.md', 'VISTA/vision_module/backend/']
        }
      };

      const details = moduleDetailsMap[modId];
      if (!details) return;

      const imgItem = (window.PICTURES_MANIFEST || []).find(item => item.id === details.imgId);
      const coverUrl = imgItem ? getRelativePath(imgItem.cover) : '';

      element.innerHTML = `
        <div class="module-hero-banner media-card" data-lightbox="true" data-title="${details.title}" data-caption="${details.responsibility}" data-tag="模块封面" style="cursor: pointer;">
          <div class="module-hero-frame" style="aspect-ratio: 21 / 9; width: 100%;">
            <img class="module-hero-img image-slot-module-cover" src="${coverUrl}" onerror="window.handleImgError(this)" style="object-fit: contain; height: 100%; width: 100%;" alt="${details.title}">
          </div>
          <div class="module-hero-caption">
            <div class="module-hero-caption-title">${details.title}</div>
            <div class="module-hero-caption-desc">${details.responsibility}</div>
          </div>
        </div>
        
        <div class="grid-3" style="margin-bottom: 24px;">
          <div class="card">
            <div class="card-title" style="color: var(--accent); font-weight: 700; font-size: 0.88rem;">输入通道 & 感知 (Input)</div>
            <p class="card-desc" style="font-family: var(--font-mono); font-size: 0.74rem; line-height: 1.5;">${details.inputs}</p>
          </div>
          <div class="card">
            <div class="card-title" style="color: var(--accent); font-weight: 700; font-size: 0.88rem;">输出通道 & 控制 (Output)</div>
            <p class="card-desc" style="font-family: var(--font-mono); font-size: 0.74rem; line-height: 1.5;">${details.outputs}</p>
          </div>
          <div class="card">
            <div class="card-title" style="color: var(--accent); font-weight: 700; font-size: 0.88rem;">核心关联目录 (Directories)</div>
            <ul style="padding-left: 16px; font-size: 0.7rem; color: var(--text-secondary); line-height: 1.5; font-family: var(--font-mono); margin-top: 4px; word-break: break-all;">
              ${details.directories.map(d => `<li>${d}</li>`).join('')}
            </ul>
          </div>
        </div>
      `;
    });

    // 3. Module Galleries
    const galleryMounts = document.querySelectorAll('.module-gallery-mount');
    galleryMounts.forEach(element => {
      const modId = element.getAttribute('data-module');
      if (!modId || !window.PICTURES_MANIFEST) return;

      const galleryItemIds = {
        'vista': ['camera_view', 'preview_debug'],
        'mobile_gateway': ['miniapp_home', 'miniapp_voice'],
        'chassis': ['docking_approach'],
        'cloud_grasp': ['grasp_scene', 'grasp_demo', 'grasp_detail', 'banana_grasp', 'experiment_overview'],
        'arm': ['grasp_scene', 'grasp_demo', 'grasp_detail', 'banana_grasp', 'experiment_overview']
      };

      const ids = galleryItemIds[modId];
      if (!ids || ids.length === 0) return;

      const items = (window.PICTURES_MANIFEST || []).filter(item => ids.includes(item.id));
      if (items.length === 0) return;

      if (modId === 'mobile_gateway') {
        element.innerHTML = `
          <h4 style="font-size: 0.9rem; font-weight: 800; margin-top: 24px; margin-bottom: 16px; text-transform: uppercase; color: var(--accent); letter-spacing: 0.05em;">小程序微信端控制屏 (UI Mockups)</h4>
          <div class="image-gallery-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); justify-content: center; gap: 30px; max-width: 580px; margin: 0 auto 24px auto;">
            ${items.map(item => `
              <div class="phone-mockup-container" style="max-width: 240px; margin: 0 auto; width: 100%;">
                <figure class="media-card" data-lightbox="true" data-title="${item.title}" data-caption="${item.caption || ''}" data-tag="界面截图">
                  <div class="phone-mockup-frame" style="aspect-ratio: 9 / 19.3; width: 100%; height: auto;">
                    <img class="image-slot-phone" src="${getRelativePath(item.src)}" onerror="window.handleImgError(this)" style="width: 100%; height: 100%; object-fit: contain; display: block;" alt="${item.title}">
                  </div>
                  <figcaption>
                    <span class="media-title">${item.title}</span>
                    <span class="media-caption">${item.caption || ''}</span>
                  </figcaption>
                </figure>
              </div>
            `).join('')}
          </div>
        `;
      } else {
        element.innerHTML = `
          <h4 style="font-size: 0.9rem; font-weight: 800; margin-top: 24px; margin-bottom: 16px; text-transform: uppercase; color: var(--accent); letter-spacing: 0.05em;">关联场景与算法调试实图 (Gallery)</h4>
          <div class="image-gallery-grid">
            ${items.map(item => {
              const slot = item.slot || 'galleryImage';
              const slotCfg = imageSlotConfig[slot] || imageSlotConfig.galleryImage;
              const tag = item.sourceType === 'screenshot' ? '调试视图' : '现场实图';
              return `
                <figure class="media-card" data-lightbox="true" data-title="${item.title}" data-caption="${item.caption || ''}" data-tag="${tag}">
                  <div class="media-frame" style="aspect-ratio: ${slotCfg.aspectRatio}; width: 100%; height: auto;">
                    <img class="image-card-img ${slotCfg.className}" src="${getRelativePath(item.src)}" onerror="window.handleImgError(this)" style="object-fit: contain; height: 100%; width: 100%;" alt="${item.title}">
                  </div>
                  <figcaption>
                    <span class="media-title">${item.title}</span>
                    <span class="media-caption">${item.caption || ''}</span>
                  </figcaption>
                </figure>
              `;
            }).join('')}
          </div>
        `;
      }
    });
  }

  // 6. Dynamic Renders
  function renderOverviewStatus() {
    const statusContainer = document.getElementById('overviewStatusGrid');
    if (!statusContainer) return;
    
    const statuses = [
      { name: "VISTA UDS", status: "OK", desc: "/tmp/robot_stack/vision_req.sock", class: "status-ok" },
      { name: "Orchestrator UDS", status: "OK", desc: "/tmp/robot_stack/task_cmd.sock", class: "status-ok" },
      { name: "Mobile Gateway", status: "ONLINE", desc: "MQTT port 1883 connected", class: "status-ok" },
      { name: "UART Keepalive", status: "50ms", desc: " Ch. serial ticks normal", class: "status-ok" },
      { name: "Config Profile", status: "BOARD", desc: "sc171_board active", class: "status-warn" },
      { name: "STOP/SSTOP", status: "ACTIVE", desc: "Chassis safety gated", class: "status-ok" }
    ];
    
    statusContainer.innerHTML = statuses.map(s => `
      <div class="card" style="padding: 12px 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
          <strong style="font-size: 0.8rem; color: var(--text-primary);">${s.name}</strong>
          <span class="status-badge ${s.class}" style="font-size: 0.65rem; padding: 1px 6px;">${s.status}</span>
        </div>
        <div style="font-size: 0.72rem; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${s.desc}</div>
      </div>
    `).join('');
  }

  function renderArchitectureMatrix() {
    const matrixContainer = document.getElementById('archMatrixBody');
    if (!matrixContainer) return;
    
    // Matrix data structure corresponding to the 5x4 layout grid
    const layers = [
      {
        name: "应用交互层 (Application Interaction)",
        cells: [
          { text: "小程序 / 语音指令", target: "overview" },
          { text: "mobile_gateway", target: "mobile_gateway" },
          { text: "-", target: "" },
          { text: "-", target: "" }
        ]
      },
      {
        name: "任务编排层 (Task Orchestration)",
        cells: [
          { text: "-", target: "" },
          { text: "Orchestrator 编排服务", target: "orchestrator" },
          { text: "-", target: "" },
          { text: "-", target: "" }
        ]
      },
      {
        name: "感知算法层 (Perception Algorithm)",
        cells: [
          { text: "-", target: "" },
          { text: "VISTA / YOLO / RGB-D 对齐", target: "vista" },
          { text: "-", target: "" },
          { text: "Grasp Planning API 规划", target: "cloud_grasp" }
        ]
      },
      {
        name: "数据通信层 (Data Communication)",
        cells: [
          { text: "MQTT 协议", target: "mobile_gateway" },
          { text: "UDS / TCP / UART", target: "startup" },
          { text: "UART 底盘接口", target: "chassis" },
          { text: "HTTP / WebSocket", target: "cloud_grasp" }
        ]
      },
      {
        name: "物理执行层 (Physical Execution)",
        cells: [
          { text: "手机移动端", target: "overview" },
          { text: "SC171 芯片 / RealSense", target: "sc171" },
          { text: "麦克纳姆底盘 / 机械臂", target: "chassis" },
          { text: "GPU Server 运算集群", target: "cloud_grasp" }
        ]
      }
    ];
    
    matrixContainer.innerHTML = layers.map(l => `
      <tr>
        <td class="matrix-row-header">${l.name}</td>
        ${l.cells.map(c => {
          if (c.text === "-") {
            return `<td style="color: var(--text-secondary); text-align: center; font-style: italic;">无模块配置</td>`;
          } else {
            return `
              <td>
                <span class="matrix-cell-module" onclick="window.switchTabGlobal('${c.target}')">
                  ${c.text}
                </span>
              </td>
            `;
          }
        }).join('')}
      </tr>
    `).join('');
  }

  // Exposed globally to allow quick jump from onclick attributes
  window.switchTabGlobal = function(targetId) {
    if (targetId) switchTab(targetId);
  };

  function renderModuleEmbeddedHTML() {
    // Inject parsed HTML content compiled by build_dashboard.py
    if (!window.ROBOT_MODULE_PAGES) return;
    
    const modulesToInject = [
      'orchestrator', 'vista', 'mobile_gateway',
      'chassis', 'arm', 'cloud_grasp',
      'startup', 'config', 'testing'
    ];

    const moduleSummaryTexts = {
      'orchestrator': 'Orchestrator 是机器人软件系统的核心控制器。负责解析外部 MQTT 取物指令，协调 VISTA 感知状态切换，并生成串口下发的目标速度与急停帧，是状态安全转移的终极屏障。',
      'vista': 'VISTA 服务负责板端相机的周期采集以及深度图像对齐。内部集成了 QNN int8 YOLO 边缘算力，对场景内的取物目标和桌边距离进行周期预测并回传状态机。',
      'mobile_gateway': '网关服务作为北向接口接入 MQTT 服务器，支持视障用户的语音/按钮取物操作下发。服务包含消息准入过滤，且双向桥接 Unix Domain Socket 通道。',
      'chassis': '底盘运动系统基于 STM32F407，支持离散占空比微调插帧以突破电机低速死区。串口通信协议定义了 MODE、VEL、STOP 和 BRAKE 命令，并支持硬件级直接急停线抢占。',
      'arm': '机械臂模块在小车通过 VISTA 和底盘微调至 AT_TABLE_EDGE 状态后，接收抓取指令并与云端 GR-ConvNet / AnyGrasp 推理配合，通过 Modbus 控制执行 6 自由度闭环抓取动作。',
      'cloud_grasp': '云端抓取通过 SC171 边缘端上传 RGB-D 点云数据，利用云端大算力运行 3D Grasp 规划。云端解析三维空间内的障碍规避并下发关节控制补偿包给底盘和机械臂。',
      'startup': '本节说明了多进程启动校验时序与进程就绪自检指标。UDS 套接字（/tmp/robot_stack/*.sock）在启动前将被彻底清理，启动自检超时设定为 35 秒。',
      'config': '系统采用多层配置覆盖逻辑，提供 car_cmd_params.yaml 与 stage_params.yaml 精调参数。任何横移与速度阈值均有安全越界校验以防止底盘在无摩擦面上漂移失控。',
      'testing': '系统配备以 pytest 驱动的测试防护体系。仿真环境通过 mock 硬件接口以避免在主机上运行真实串口读写，在每次代码变更时提供核心契约拦截与行为回归防御。'
    };
    
    modulesToInject.forEach(modId => {
      const container = document.getElementById(`doc-content-${modId}`);
      if (!container) return;
      
      const fragments = window.ROBOT_MODULE_PAGES[modId] || [];
      if (fragments.length === 0) {
        container.innerHTML = `
          <div class="card" style="border-left: 4px solid var(--warning);">
            <p style="font-size: 0.82rem; font-style: italic;">此模块暂无关联的 Markdown 正文文档。</p>
          </div>
        `;
        return;
      }

      const summaryHtml = `
        <div class="card" style="border-left: 4px solid var(--accent); margin-bottom: 24px; background-color: var(--accent-light); padding: 16px;">
          <strong style="font-size: 0.82rem; color: var(--accent); display: block; margin-bottom: 6px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;">文档摘要与运行安全规约 (Summary)</strong>
          <p class="card-desc" style="font-size: 0.8rem; line-height: 1.5; color: var(--text-primary); margin: 0;">${moduleSummaryTexts[modId] || '此部分包含该模块的详细工程说明及核心接口协议文档。'}</p>
        </div>
      `;
      
      container.innerHTML = summaryHtml + fragments.map(frag => `
        <div style="margin-bottom: 30px; border-bottom: 1px dashed var(--border-color); padding-bottom: 24px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <span style="font-size: 0.75rem; font-weight: 700; color: var(--accent);">文档源自：<code style="font-family: var(--font-mono); font-size: 0.72rem;">${frag.source_path}</code></span>
            <span class="source-path-badge" style="font-size: 0.65rem;">COMPILED HTML</span>
          </div>
          <div class="markdown-html-content">
            ${frag.html}
          </div>
        </div>
      `).join('');
    });
  }

  function renderConfigMatrix() {
    const paramsContainer = document.getElementById('configParamsBody');
    if (paramsContainer && window.ROBOT_CONFIG_MATRIX && window.ROBOT_CONFIG_MATRIX.parameters) {
      paramsContainer.innerHTML = window.ROBOT_CONFIG_MATRIX.parameters.map(p => {
        const isDangerous = p.dangerous_threshold !== "N/A";
        return `
          <tr>
            <td style="font-weight: 600; font-family: var(--font-mono); color: var(--accent);">${p.name}</td>
            <td style="color: var(--text-secondary); font-size: 0.8rem;">${p.description}</td>
            <td style="font-family: var(--font-mono); font-size: 0.8rem; text-align: center;">${p.board_val}</td>
            <td style="font-family: var(--font-mono); font-size: 0.8rem; text-align: center;">${p.dev_val}</td>
            <td>
              ${isDangerous ? `<span class="status-badge status-danger" style="margin-bottom: 4px;">阈值: ${p.dangerous_threshold}</span>` : `<span class="status-badge status-ok">安全默认</span>`}
              <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 2px;">${p.danger_reason}</div>
            </td>
          </tr>
        `;
      }).join('');
    }
  }

  function renderTroubleshooting() {
    const listContainer = document.getElementById('troubleAccordionList');
    if (!listContainer || !window.ROBOT_TROUBLESHOOTING || !window.ROBOT_TROUBLESHOOTING.troubleshootings) return;
    
    listContainer.innerHTML = window.ROBOT_TROUBLESHOOTING.troubleshootings.map((item, idx) => `
      <div class="accordion-item" id="acc-${item.id}">
        <div class="accordion-header" onclick="toggleAccordion('acc-${item.id}')">
          <span style="display: flex; align-items: center; gap: 8px;">
            <span style="background-color: var(--danger-light); color: var(--danger); width: 20px; height: 20px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: bold;">!</span>
            <span>${item.title}</span>
          </span>
          <svg class="accordion-icon" viewBox="0 0 24 24">
            <path d="M7 10l5 5 5-5z"/>
          </svg>
        </div>
        <div class="accordion-content">
          <div class="accordion-body">
            <div style="margin-bottom: 12px;"><strong style="color: var(--text-primary);">故障现象：</strong><span style="color: var(--danger);">${item.symptom}</span></div>
            <div style="margin-bottom: 12px;"><strong style="color: var(--text-primary);">可能原因：</strong><p style="margin-top: 4px; white-space: pre-wrap; font-size: 0.8rem; line-height: 1.5;">${item.cause}</p></div>
            
            <div style="margin-bottom: 12px;">
              <strong style="color: var(--text-primary);">排查检查命令：</strong>
              <div class="code-block-container" style="margin-top: 6px;">
                <button class="copy-btn">Copy</button>
                <pre class="code-block"><code id="code-check-${item.id}">${item.check_command}</code></pre>
              </div>
            </div>
            
            <div style="margin-bottom: 12px;"><strong style="color: var(--text-primary);">修复方向：</strong><p style="margin-top: 4px; white-space: pre-wrap; font-size: 0.8rem; line-height: 1.5; color: var(--success);">${item.fix}</p></div>
            <div><strong style="color: var(--text-primary);">日志过滤关键词：</strong><code style="background: rgba(239, 68, 68, 0.1); color: var(--danger); padding: 2px 6px; border-radius: 4px; font-family: var(--font-mono); font-size: 0.75rem; border: 1px solid rgba(239, 68, 68, 0.2);">${item.keywords}</code></div>
          </div>
        </div>
      </div>
    `).join('');
  }

  function renderTesting() {
    // Manual testing items list
    const testCases = [
      { name: "test_config_loader.py", protect: "配置系统 Schema 正确加载，防缺失爆雷", command: "python -m pytest tests/common/test_config_loader.py -q" },
      { name: "test_config_override.py", protect: "安全校验横移速度 (edge_slide_vy_mps) 溢出拦截规约", command: "python -m pytest tests/orchestrator/test_config_override.py -q" },
      { name: "test_safety_gating.py", protect: "底盘移动阻挡、丢帧保护及物理跌落阻断", command: "python -m pytest tests/orchestrator/test_safety_gating.py -q" },
      { name: "test_emergency_stop.py", protect: "物理急停指令优先级插队抢占，防止底层速度包覆盖", command: "python -m pytest tests/orchestrator/test_emergency_stop.py -q" },
      { name: "test_simple_car_protocol.py", protect: "底盘协议 (MODE/VEL/STOP/BRAKE) 串口打包编解码稳定", command: "python -m pytest tests/orchestrator/test_simple_car_protocol.py -q" },
      { name: "test_vision_state_sync.py", protect: "VISTA Perception Stage 与状态机同步失步规避", command: "python -m pytest tests/orchestrator/test_vision_state_sync.py -q" },
      { name: "test_grasp_reposition.py", protect: "接近桌边和微调对齐超调限制逻辑", command: "python -m pytest tests/orchestrator/test_grasp_reposition.py -q" },
      { name: "test_observation_router.py", protect: "控制级观测与诊断大包日志彻底物理独立信道隔离", command: "python -m pytest VISTA/vision_module/test/test_observation_router.py -q" },
      { name: "test_stage_contract.py", protect: "VISTA 内置感知状态机协议生命周期和输入校验稳定", command: "python -m pytest VISTA/vision_module/test/test_stage_contract.py -q" }
    ];
    
    const tBody = document.getElementById('testingListBody');
    if (!tBody) return;
    
    tBody.innerHTML = testCases.map((tc, idx) => `
      <tr>
        <td style="font-family: var(--font-mono); font-weight: 700; color: var(--accent);">${tc.name}</td>
        <td>${tc.protect}</td>
        <td>
          <div class="code-block-container" style="margin: 0;">
            <button class="copy-btn">Copy</button>
            <pre class="code-block" style="padding: 6px 12px;"><code id="code-test-${idx}">${tc.command}</code></pre>
          </div>
        </td>
      </tr>
    `).join('');
  }

  function renderReferenceDocs() {
    const listContainer = document.getElementById('referenceDocsList');
    const docModal = document.getElementById('rawDocViewModal');
    const docModalTitle = document.getElementById('rawDocModalTitle');
    const docModalBody = document.getElementById('rawDocModalBody');
    const docModalClose = document.getElementById('rawDocModalClose');
    
    if (!listContainer || !window.ROBOT_DOCS_MANIFEST || !window.RAW_DOCS) return;
    
    // Group references
    const groups = {};
    window.ROBOT_DOCS_MANIFEST.forEach(doc => {
      if (!groups[doc.group]) groups[doc.group] = [];
      groups[doc.group].push(doc);
    });
    
    listContainer.innerHTML = Object.keys(groups).map(gName => `
      <div style="margin-bottom: 24px;">
        <h4 style="color: var(--accent); font-size: 0.95rem; border-bottom: 2px solid var(--border-color); padding-bottom: 4px; margin-bottom: 12px; font-weight: 800;">
          ${gName}
        </h4>
        <div class="grid-3">
          ${groups[gName].map(doc => `
            <div class="card" style="cursor: pointer;" onclick="window.openRawDocModal('${doc.id}')">
              <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">
                <div>
                  <h5 style="font-size: 0.85rem; margin-bottom: 6px; color: var(--text-primary); font-weight: 700;">${doc.title}</h5>
                  <p class="card-desc" style="font-size: 0.75rem; margin-bottom: 10px;">${doc.summary}</p>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.65rem; color: var(--text-secondary);">
                  <span class="source-path-badge">${doc.source_path}</span>
                  <span style="color: var(--accent); font-weight: bold;">阅读原文 &rarr;</span>
                </div>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `).join('');

    // Bind modal controls
    window.openRawDocModal = function(docId) {
      const doc = window.RAW_DOCS[docId];
      if (!doc) return;
      
      docModalTitle.innerText = doc.title;
      docModalBody.innerHTML = `
        <div style="font-size: 0.72rem; color: var(--text-secondary); margin-bottom: 14px; border-bottom: 1px solid var(--card-border); padding-bottom: 6px;">
          源文件路径：<code style="font-family: var(--font-mono); font-size: 0.72rem; background: var(--bg-color); padding: 1px 4px; border-radius: 3px;">${doc.source_path}</code>
        </div>
        <div class="markdown-html-content">
          ${doc.html}
        </div>
      `;
      docModal.style.display = 'block';
    };

    if (docModalClose) {
      docModalClose.addEventListener('click', () => {
        docModal.style.display = 'none';
      });
    }

    // Close on overlay click
    window.addEventListener('click', (e) => {
      if (e.target === docModal) {
        docModal.style.display = 'none';
      }
    });
  }

  // 7. Dynamic SVG polylines frequency chart rendering
  function drawSVGFrequencyGraph() {
    const svgEl = document.getElementById('svgLineChart');
    if (!svgEl) return;
    
    // Create random fluctuating ticks around 20Hz (Ticker loop time 50ms)
    let points = "";
    const width = 280;
    const height = 120;
    const padding = 10;
    
    const ticksCount = 20;
    const step = (width - padding * 2) / (ticksCount - 1);
    
    for (let i = 0; i < ticksCount; i++) {
      // 20Hz +- 1.5Hz fluctuation
      const hzVal = 20 + (Math.random() * 3.0 - 1.5);
      // Map HZ (range 15 - 25) to Y height (120 - 10)
      const y = height - padding - ((hzVal - 15) / 10) * (height - padding * 2);
      const x = padding + i * step;
      points += `${x},${y} `;
    }
    
    const polyline = svgEl.querySelector('polyline');
    if (polyline) {
      polyline.setAttribute('points', points.trim());
    }
  }

  // 8. Core Client-Side Global Searching (can redirect to modules details)
  const searchInput = document.getElementById('searchInput');
  searchInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase().trim();
    filterContent(query);
  });

  function filterContent(query) {
    if (!query) {
      document.querySelectorAll('.card, .accordion-item, tr').forEach(el => {
        el.style.display = '';
        removeHighlights(el);
      });
      
      const notify = document.getElementById('searchNotification');
      if (notify) notify.style.display = 'none';
      return;
    }
    
    // Search within active tab panel
    const activePanel = document.querySelector('.section-panel.active');
    if (!activePanel) return;
    
    const cards = activePanel.querySelectorAll('.card');
    const accordions = activePanel.querySelectorAll('.accordion-item');
    const tableRows = activePanel.querySelectorAll('tbody tr');
    
    let activeMatchCount = 0;
    
    cards.forEach(card => {
      removeHighlights(card);
      const text = card.innerText.toLowerCase();
      if (text.includes(query)) {
        card.style.display = '';
        highlightText(card, query);
        activeMatchCount++;
      } else {
        card.style.display = 'none';
      }
    });

    accordions.forEach(acc => {
      removeHighlights(acc);
      const text = acc.innerText.toLowerCase();
      if (text.includes(query)) {
        acc.style.display = '';
        acc.classList.add('active');
        highlightText(acc, query);
        activeMatchCount++;
      } else {
        acc.style.display = 'none';
        acc.classList.remove('active');
      }
    });
    
    tableRows.forEach(row => {
      removeHighlights(row);
      const text = row.innerText.toLowerCase();
      if (text.includes(query)) {
        row.style.display = '';
        highlightText(row, query);
        activeMatchCount++;
      } else {
        row.style.display = 'none';
      }
    });

    // Check if matching module pages exist globally
    // If no matches in active page, notify user of other matches
    const globalMatches = [];
    const moduleTitles = {
      'orchestrator': 'SC171 Edge Center -> Orchestrator',
      'vista': 'SC171 Edge Center -> VISTA',
      'mobile_gateway': 'SC171 Edge Center -> Mobile Gateway',
      'chassis': 'Execution System -> STM32 Chassis',
      'arm': 'Execution System -> Arm Controller',
      'cloud_grasp': 'Cloud Grasping',
      'startup': 'Runtime & Startup',
      'config': 'Configuration',
      'testing': 'Testing & Debug'
    };

    if (window.ROBOT_MODULE_PAGES) {
      Object.entries(window.ROBOT_MODULE_PAGES).forEach(([modId, frags]) => {
        frags.forEach(frag => {
          if (frag.html.toLowerCase().includes(query) || frag.title.toLowerCase().includes(query)) {
            if (!globalMatches.includes(modId)) {
              globalMatches.push(modId);
            }
          }
        });
      });
    }

    const notify = document.getElementById('searchNotification');
    if (notify) {
      if (globalMatches.length > 0) {
        notify.style.display = 'block';
        notify.innerHTML = `
          <span>在其他模块中发现匹配项：</span>
          ${globalMatches.map(mId => `
            <button class="shortcut-btn" style="padding: 2px 6px; font-size: 0.65rem;" onclick="window.switchTabGlobal('${mId}')">
              ${moduleTitles[mId] || mId}
            </button>
          `).join(' ')}
        `;
      } else {
        notify.style.display = 'none';
      }
    }
  }

  function highlightText(element, query) {
    const walk = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null, false);
    let node;
    const textNodes = [];
    while (node = walk.nextNode()) {
      textNodes.push(node);
    }
    
    textNodes.forEach(node => {
      const parent = node.parentNode;
      if (parent.tagName === 'SCRIPT' || parent.tagName === 'STYLE' || parent.classList.contains('copy-btn') || parent.tagName === 'CODE') return;
      
      const val = node.nodeValue;
      const index = val.toLowerCase().indexOf(query);
      if (index >= 0) {
        const span = document.createElement('span');
        span.className = 'temp-highlight-container';
        
        let remaining = val;
        let safeCounter = 0;
        while (remaining.toLowerCase().indexOf(query) >= 0 && safeCounter < 20) {
          safeCounter++;
          const idx = remaining.toLowerCase().indexOf(query);
          const prefix = remaining.substring(0, idx);
          const match = remaining.substring(idx, idx + query.length);
          
          if (prefix) span.appendChild(document.createTextNode(prefix));
          
          const highlightSpan = document.createElement('span');
          highlightSpan.className = 'highlight';
          highlightSpan.appendChild(document.createTextNode(match));
          span.appendChild(highlightSpan);
          
          remaining = remaining.substring(idx + query.length);
        }
        if (remaining) span.appendChild(document.createTextNode(remaining));
        
        parent.replaceChild(span, node);
      }
    });
  }

  // --- V3 Lightbox Viewer Implementation ---
  let currentPlaylist = [];
  let currentPlaylistIndex = 0;

  function initLightbox() {
    if (document.getElementById('lightboxOverlay')) return;
    
    const overlay = document.createElement('div');
    overlay.id = 'lightboxOverlay';
    overlay.className = 'lightbox-overlay';
    overlay.innerHTML = `
      <button class="lightbox-close" id="lightboxCloseBtn" title="关闭 (Esc)">&times;</button>
      <button class="lightbox-nav lightbox-prev" id="lightboxPrevBtn" title="上一张 (←)">&#10094;</button>
      <button class="lightbox-nav lightbox-next" id="lightboxNextBtn" title="下一张 (→)">&#10095;</button>
      <div class="lightbox-content">
        <div class="lightbox-image-container">
          <img id="lightboxImage" src="" alt="">
        </div>
        <div class="lightbox-caption-container">
          <span class="lightbox-tag" id="lightboxTag">Tag</span>
          <h4 class="lightbox-title" id="lightboxTitle">Title</h4>
          <p class="lightbox-caption" id="lightboxCaption">Caption</p>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    const closeBtn = document.getElementById('lightboxCloseBtn');
    const prevBtn = document.getElementById('lightboxPrevBtn');
    const nextBtn = document.getElementById('lightboxNextBtn');

    closeBtn.addEventListener('click', closeLightbox);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay || e.target.closest('.lightbox-image-container')) {
        if (e.target !== document.getElementById('lightboxImage')) {
          closeLightbox();
        }
      }
    });

    prevBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      navigateLightbox(-1);
    });
    nextBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      navigateLightbox(1);
    });

    document.addEventListener('keydown', (e) => {
      if (!overlay.classList.contains('active')) return;
      if (e.key === 'Escape') closeLightbox();
      if (e.key === 'ArrowLeft') navigateLightbox(-1);
      if (e.key === 'ArrowRight') navigateLightbox(1);
    });
  }

  function openLightbox(imgUrl, title, caption, tag, playlist, index) {
    const overlay = document.getElementById('lightboxOverlay');
    const img = document.getElementById('lightboxImage');
    const titleEl = document.getElementById('lightboxTitle');
    const captionEl = document.getElementById('lightboxCaption');
    const tagEl = document.getElementById('lightboxTag');

    if (!overlay || !img || !titleEl || !captionEl || !tagEl) return;

    img.src = imgUrl;
    titleEl.innerText = title || '';
    captionEl.innerText = caption || '';
    tagEl.innerText = tag || 'IMAGE';
    tagEl.style.display = tag ? 'inline-block' : 'none';

    currentPlaylist = playlist || [];
    currentPlaylistIndex = index || 0;

    const prevBtn = document.getElementById('lightboxPrevBtn');
    const nextBtn = document.getElementById('lightboxNextBtn');
    if (currentPlaylist.length > 1) {
      prevBtn.style.display = 'flex';
      nextBtn.style.display = 'flex';
    } else {
      prevBtn.style.display = 'none';
      nextBtn.style.display = 'none';
    }

    overlay.style.display = 'flex';
    overlay.offsetWidth; // force reflow
    overlay.classList.add('active');
  }

  function closeLightbox() {
    const overlay = document.getElementById('lightboxOverlay');
    if (overlay) {
      overlay.classList.remove('active');
      setTimeout(() => {
        overlay.style.display = 'none';
      }, 300);
    }
  }

  function navigateLightbox(direction) {
    if (currentPlaylist.length <= 1) return;
    currentPlaylistIndex = (currentPlaylistIndex + direction + currentPlaylist.length) % currentPlaylist.length;
    const nextItem = currentPlaylist[currentPlaylistIndex];
    
    const img = document.getElementById('lightboxImage');
    const titleEl = document.getElementById('lightboxTitle');
    const captionEl = document.getElementById('lightboxCaption');
    const tagEl = document.getElementById('lightboxTag');

    if (!img || !titleEl || !captionEl || !tagEl) return;

    img.src = nextItem.src;
    titleEl.innerText = nextItem.title || '';
    captionEl.innerText = nextItem.caption || '';
    tagEl.innerText = nextItem.tag || 'IMAGE';
    tagEl.style.display = nextItem.tag ? 'inline-block' : 'none';
  }

  // Event Delegation for Lightbox trigger clicks
  document.addEventListener('click', (e) => {
    const mediaCard = e.target.closest('[data-lightbox="true"]');
    if (mediaCard) {
      const img = mediaCard.querySelector('img');
      const src = img ? img.getAttribute('src') : '';
      const title = mediaCard.getAttribute('data-title') || '';
      const caption = mediaCard.getAttribute('data-caption') || '';
      const tag = mediaCard.getAttribute('data-tag') || 'IMAGE';

      let siblingCards = [];
      if (mediaCard.classList.contains('module-hero-banner')) {
        siblingCards = [mediaCard];
      } else {
        const gallery = mediaCard.closest('.image-gallery-grid') || mediaCard.closest('#overview') || mediaCard.closest('.main-content');
        siblingCards = gallery ? Array.from(gallery.querySelectorAll('[data-lightbox="true"]')).filter(card => !card.classList.contains('module-hero-banner')) : [mediaCard];
      }
      
      const playlist = siblingCards.map(card => {
        const cardImg = card.querySelector('img');
        return {
          src: cardImg ? cardImg.getAttribute('src') : '',
          title: card.getAttribute('data-title') || '',
          caption: card.getAttribute('data-caption') || '',
          tag: card.getAttribute('data-tag') || 'IMAGE'
        };
      });

      const index = siblingCards.indexOf(mediaCard);
      openLightbox(src, title, caption, tag, playlist, index);
    }
  });

  function removeHighlights(element) {
    const containers = element.querySelectorAll('.temp-highlight-container');
    containers.forEach(container => {
      const parent = container.parentNode;
      const text = container.innerText;
      const textNode = document.createTextNode(text);
      parent.replaceChild(textNode, container);
    });
    
    const highlights = element.querySelectorAll('.highlight');
    highlights.forEach(h => {
      const parent = h.parentNode;
      const text = h.innerText;
      const textNode = document.createTextNode(text);
      parent.replaceChild(textNode, h);
    });
  }
});
