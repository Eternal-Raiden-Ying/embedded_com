#!/usr/bin/env python3
import os
import shutil
import json
from PIL import Image, ImageOps, ImageEnhance, ImageDraw

def pad_to_ratio(img, target_w, target_h, bg_color=(255, 255, 255)):
    """
    Fits image within target size and pads the background to avoid stretching.
    """
    img_w, img_h = img.size
    target_aspect = target_w / target_h
    img_aspect = img_w / img_h
    
    if img_aspect > target_aspect:
        # Image is wider: fit width, pad height
        new_w = img_w
        new_h = int(img_w / target_aspect)
        offset_x = 0
        offset_y = (new_h - img_h) // 2
    else:
        # Image is taller: fit height, pad width
        new_h = img_h
        new_w = int(img_h * target_aspect)
        offset_x = (new_w - img_w) // 2
        offset_y = 0
        
    canvas = Image.new('RGB', (new_w, new_h), bg_color)
    canvas.paste(img, (offset_x, offset_y))
    return canvas.resize((target_w, target_h), Image.Resampling.LANCZOS)

def crop_to_ratio(img, target_w, target_h):
    """
    Crops image from center to match target aspect ratio.
    """
    img_w, img_h = img.size
    target_aspect = target_w / target_h
    img_aspect = img_w / img_h
    
    if img_aspect > target_aspect:
        new_w = int(img_h * target_aspect)
        offset_x = (img_w - new_w) // 2
        img = img.crop((offset_x, 0, offset_x + new_w, img_h))
    else:
        new_h = int(img_w / target_aspect)
        offset_y = (img_h - new_h) // 2
        img = img.crop((0, offset_y, img_w, offset_y + new_h))
        
    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

def main():
    # Setup base directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    
    pictures_dir = os.path.join(root_dir, 'pictures')
    original_dir = os.path.join(pictures_dir, 'original')
    processed_dir = os.path.join(pictures_dir, 'processed')
    meta_dir = os.path.join(pictures_dir, 'meta')
    
    hero_dir = os.path.join(processed_dir, 'hero')
    modules_dir = os.path.join(processed_dir, 'modules')
    scenes_dir = os.path.join(processed_dir, 'scenes')
    phone_dir = os.path.join(processed_dir, 'phone')
    thumbs_dir = os.path.join(processed_dir, 'thumbs')
    
    # Create necessary folders
    for d in [original_dir, processed_dir, meta_dir, hero_dir, modules_dir, scenes_dir, phone_dir, thumbs_dir]:
        os.makedirs(d, exist_ok=True)
        
    print("Moving original images from root pictures/ to original/...")
    # Move original images if they are in root docs/pictures
    for name in os.listdir(pictures_dir):
        path = os.path.join(pictures_dir, name)
        if os.path.isfile(path) and name.lower().endswith(('.png', '.jpg', '.jpeg')):
            dest = os.path.join(original_dir, name)
            shutil.move(path, dest)
            print(f"  Moved: {name} -> original/{name}")
            
    # Classify images
    images_catalog = [
        # 1. AI Diagrams / Module Covers
        {"file": "VISTA.png", "id": "vista", "type": "module_cover", "module": "VISTA", "title": "VISTA 视觉模块", "notes": "视觉推理服务，包含摄像头管理与 Yolov7 检测", "slot": "moduleCover", "fit": "cover", "lightbox": False, "sourceType": "diagram"},
        {"file": "gateway.png", "id": "mobile_gateway", "type": "module_cover", "module": "Mobile Gateway", "title": "Mobile Gateway 网关", "notes": "小程序MQTT网关服务说明与指令桥接", "slot": "moduleCover", "fit": "cover", "lightbox": False, "sourceType": "diagram"},
        {"file": "sc171计算.png", "id": "sc171", "type": "module_cover", "module": "SC171", "title": "SC171 边缘计算", "notes": "边缘端侧核心处理器资源和接口拓扑", "slot": "moduleCover", "fit": "cover", "lightbox": False, "sourceType": "diagram"},
        {"file": "stm32执行.png", "id": "stm32_execution", "type": "module_cover", "module": "Chassis Execution", "title": "STM32底盘执行", "notes": "底盘麦克纳姆轮低速解算以及总线控制", "slot": "moduleCover", "fit": "cover", "lightbox": False, "sourceType": "diagram"},
        {"file": "底盘封面.png", "id": "chassis", "type": "module_cover", "module": "Chassis", "title": "麦克纳姆底盘", "notes": "小车底盘结构与底盘三轴解算协议", "slot": "moduleCover", "fit": "cover", "lightbox": False, "sourceType": "diagram"},
        {"file": "总图.png", "id": "overview", "type": "overview_hero", "module": "Overview", "title": "系统组成总图", "notes": "系统核心组件连接总图", "slot": "systemBanner", "fit": "contain", "lightbox": True, "sourceType": "diagram"},
        {"file": "系统工作闭环.png", "id": "system_loop", "type": "overview_hero", "module": "Overview", "title": "系统工作闭环图", "notes": "手机端到边缘端再到底盘端的闭环路径", "slot": "overviewHero", "fit": "cover", "lightbox": False, "sourceType": "diagram"},
        {"file": "系统全局图比例优化版.png", "id": "system_global", "type": "overview_hero", "module": "Overview", "title": "系统全局总览图", "notes": "面向视障服务的智能语音取物机器人系统全局关系图", "slot": "overviewHero", "fit": "cover", "lightbox": True, "sourceType": "diagram"},
        {"file": "preview效果图.png", "id": "preview_debug", "type": "scene_photo", "module": "VISTA", "title": "视觉调试效果预览", "notes": "视觉端计算对齐的BBox及调试实时框", "slot": "debugScene", "fit": "contain", "lightbox": True, "sourceType": "screenshot"},
        
        # 2. Real Photos / Scenarios
        {"file": "俯视图.jpg", "id": "experiment_overview", "type": "scene_photo", "module": "Overview", "title": "实验场地俯视图", "notes": "实验室内跑车测试场景的顶空俯瞰图", "slot": "galleryImage", "fit": "contain", "lightbox": True, "sourceType": "real_photo"},
        {"file": "抓取场景图.jpg", "id": "grasp_scene", "type": "scene_photo", "module": "Cloud Grasp", "title": "远程抓取实验场景", "notes": "小车停靠在桌边，机械臂在取物动作中的场景", "slot": "galleryImage", "fit": "contain", "lightbox": True, "sourceType": "real_photo"},
        {"file": "抓取展示图.jpg", "id": "grasp_demo", "type": "scene_photo", "module": "Cloud Grasp", "title": "抓取流程展示", "notes": "机械臂抓取目标物体的演示效果", "slot": "galleryImage", "fit": "contain", "lightbox": True, "sourceType": "real_photo"},
        {"file": "抓取细节图.jpg", "id": "grasp_detail", "type": "scene_photo", "module": "Cloud Grasp", "title": "爪部抓取细节", "notes": "夹爪接触物体的瞬间对齐特写", "slot": "galleryImage", "fit": "contain", "lightbox": True, "sourceType": "real_photo"},
        {"file": "抓取香蕉.jpg", "id": "banana_grasp", "type": "scene_photo", "module": "Cloud Grasp", "title": "抓取香蕉实验", "notes": "针对异形水果（香蕉）的高灵敏度抓取测试", "slot": "galleryImage", "fit": "contain", "lightbox": True, "sourceType": "real_photo"},
        {"file": "摄像机视角.jpg", "id": "camera_view", "type": "scene_photo", "module": "VISTA", "title": "摄像机原始视场", "notes": "板端 RealSense RGB 相机的实际视场测试图", "slot": "galleryImage", "fit": "contain", "lightbox": True, "sourceType": "real_photo"},
        {"file": "靠近.jpg", "id": "docking_approach", "type": "scene_photo", "module": "Chassis", "title": "小车停靠接近", "notes": "小车前端接近桌角锁边的过程特写", "slot": "galleryImage", "fit": "contain", "lightbox": True, "sourceType": "real_photo"},
        
        # 3. Mobile App Screenshots
        {"file": "小程序截图.jpg", "id": "miniapp_home", "type": "phone_screenshot", "module": "Mobile Gateway", "title": "小程序主界面", "notes": "视障语音取物微信小程序主功能控制屏截图", "slot": "phoneMockup", "fit": "contain", "lightbox": True, "sourceType": "screenshot"},
        {"file": "小程序语音识别图.jpg", "id": "miniapp_voice", "type": "phone_screenshot", "module": "Mobile Gateway", "title": "小程序语音唤醒", "notes": "小程序在识别视障用户语音命令时的状态页截图", "slot": "phoneMockup", "fit": "contain", "lightbox": True, "sourceType": "screenshot"}
    ]
    
    manifest = []
    
    ratio_optimized_dir = os.path.join(pictures_dir, '比例优化')
    optimized_source_files = {
        "vista": "视觉服务示意图比例优化后.png",
        "mobile_gateway": "网关图比例优化后.png",
        "sc171": "SC 171图比例优化后.png",
        "stm32_execution": "STM32执行控制系统图比例优化后.png",
        "chassis": "底盘模块图比例优化后.png",
        "overview": "全局总览图比例优化后.png",
        "system_loop": "闭环系统图比例调整版.png",
        "system_global": "系统全局图比例优化版.png"
    }

    print("Processing images with Pillow...")
    for item in images_catalog:
        opt_file = optimized_source_files.get(item["id"])
        src_path = ""
        is_ratio_optimized = False
        if opt_file:
            opt_path = os.path.join(ratio_optimized_dir, opt_file)
            if os.path.exists(opt_path):
                src_path = opt_path
                is_ratio_optimized = True
                print(f"  Found ratio-optimized source for {item['id']}: {opt_file}")
                
        if not src_path:
            src_path = os.path.join(original_dir, item["file"])
            
        if not os.path.exists(src_path):
            print(f"  Warning: Image not found: {item['file']} (checked original and optimized)")
            continue
            
        try:
            img = Image.open(src_path)
            
            # Auto-orient based on Exif
            img = ImageOps.exif_transpose(img)
            
            # Setup destination names (kebab-case)
            kebab_name = item["id"].replace('_', '-')
            
            cover_rel = ""
            hero_rel = ""
            scene_rel = ""
            phone_rel = ""
            thumb_rel = f"pictures/processed/thumbs/{kebab_name}-thumb.webp"
            thumb_abs = os.path.join(thumbs_dir, f"{kebab_name}-thumb.webp")
            
            # 1. AI Diagrams / Module Covers
            if item["type"] in ("module_cover", "overview_hero"):
                # Convert to RGB if PNG to add white background
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Dynamic sizes based on aspect ratio
                if is_ratio_optimized:
                    cov_w, cov_h = 1200, 514
                    her_w, her_h = 1600, 686
                    thm_w, thm_h = 640, 274
                else:
                    cov_w, cov_h = 1200, 675
                    her_w, her_h = 1600, 900
                    thm_w, thm_h = 640, 360
                
                # Make Cover
                cover_img = pad_to_ratio(img, cov_w, cov_h, (255, 255, 255))
                cover_abs = os.path.join(modules_dir, f"{kebab_name}-cover.webp")
                cover_img.save(cover_abs, 'WEBP', quality=88)
                cover_rel = f"pictures/processed/modules/{kebab_name}-cover.webp"
                
                # Make Hero
                hero_img = pad_to_ratio(img, her_w, her_h, (255, 255, 255))
                hero_abs = os.path.join(hero_dir, f"{kebab_name}-hero.webp")
                hero_img.save(hero_abs, 'WEBP', quality=88)
                hero_rel = f"pictures/processed/hero/{kebab_name}-hero.webp"
                
                # Thumbnail
                thumb_img = pad_to_ratio(img, thm_w, thm_h, (255, 255, 255))
                thumb_img.save(thumb_abs, 'WEBP', quality=88)
                
            # 2. Real Photos / Scenes
            elif item["type"] == "scene_photo":
                # Enhance image slightly
                img = ImageEnhance.Brightness(img).enhance(1.05)
                img = ImageEnhance.Contrast(img).enhance(1.10)
                img = ImageEnhance.Sharpness(img).enhance(1.15)
                
                # Crop to 16:9
                scene_img = crop_to_ratio(img, 1200, 675)
                scene_abs = os.path.join(scenes_dir, f"{kebab_name}-scene.webp")
                scene_img.save(scene_abs, 'WEBP', quality=88)
                scene_rel = f"pictures/processed/scenes/{kebab_name}-scene.webp"
                
                # Thumbnail
                thumb_img = crop_to_ratio(img, 640, 360)
                thumb_img.save(thumb_abs, 'WEBP', quality=88)
                
            # 3. Phone Screenshots
            elif item["type"] == "phone_screenshot":
                # Enhance screenshot slightly
                img = ImageEnhance.Contrast(img).enhance(1.05)
                
                # Fit vertical inside canvas
                mockup_w, mockup_h = 420, 900
                screen_w, screen_h = 360, 780
                
                # Canvas
                canvas = Image.new('RGB', (mockup_w, mockup_h), (241, 245, 249)) # Slate light gray background
                
                # Resize screenshot to screen bezel dimensions
                resized_screen = img.resize((screen_w, screen_h), Image.Resampling.LANCZOS)
                
                # Center screen on mockup canvas
                offset_x = (mockup_w - screen_w) // 2
                offset_y = (mockup_h - screen_h) // 2
                canvas.paste(resized_screen, (offset_x, offset_y))
                
                # Draw dark bezel frame
                draw = ImageDraw.Draw(canvas)
                bezel_border = [offset_x - 3, offset_y - 3, offset_x + screen_w + 3, offset_y + screen_h + 3]
                draw.rectangle(bezel_border, outline=(15, 23, 42), width=4) # dark slate bezel
                
                # Save phone mockup
                phone_abs = os.path.join(phone_dir, f"{kebab_name}-mockup.webp")
                canvas.save(phone_abs, 'WEBP', quality=88)
                phone_rel = f"pictures/processed/phone/{kebab_name}-mockup.webp"
                
                # Thumbnail (make it 16:9 padded)
                thumb_img = pad_to_ratio(canvas, 640, 360, (241, 245, 249))
                thumb_img.save(thumb_abs, 'WEBP', quality=88)
                
            # Populate manifest item
            manifest_item = {
                "id": item["id"],
                "src": cover_rel or scene_rel or phone_rel,
                "cover": cover_rel or scene_rel or phone_rel,
                "hero": hero_rel or scene_rel,
                "thumb": thumb_rel,
                "slot": item.get("slot", "galleryImage"),
                "fit": item.get("fit", "contain"),
                "title": item["title"],
                "caption": item["notes"],
                "notes": item["notes"],
                "module": item["module"],
                "lightbox": item.get("lightbox", True),
                "sourceType": "optimized_image" if is_ratio_optimized else item.get("sourceType", "real_photo"),
                "type": item["type"]
            }
            manifest.append(manifest_item)
            print(f"  Processed: {item['file']} -> {manifest_item['cover']}")
            
        except Exception as e:
            print(f"  Error processing image {item['file']}: {e}")
            
    # Write to pictures/meta/pictures_manifest.json
    manifest_json_path = os.path.join(meta_dir, 'pictures_manifest.json')
    with open(manifest_json_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        
    # Write to dashboard/data/pictures_manifest.js
    dashboard_data_dir = os.path.join(root_dir, 'dashboard', 'data')
    os.makedirs(dashboard_data_dir, exist_ok=True)
    manifest_js_path = os.path.join(dashboard_data_dir, 'pictures_manifest.js')
    with open(manifest_js_path, 'w', encoding='utf-8') as f:
        f.write("window.PICTURES_MANIFEST = " + json.dumps(manifest, ensure_ascii=False, indent=2) + ";")
        
    print(f"\nSuccessfully generated pictures manifest:")
    print(f"  - JSON: {manifest_json_path}")
    print(f"  - JS: {manifest_js_path}")

if __name__ == '__main__':
    main()
