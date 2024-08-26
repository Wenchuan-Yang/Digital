# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: run.py

"""
 python run.py \
 --src_image assets/examples/source/s12.jpg \
 --dri_video assets/examples/driving/d0.mp4 \
 --cfg configs/trt_infer.yaml
"""
import os
import argparse
import pdb
import subprocess
import ffmpeg
import cv2
import time
import numpy as np
import datetime
import platform
from omegaconf import OmegaConf

os.environ["DISPLAY"] = ":0"  # 设置DISPLAY变量
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # 禁用Qt的显示功能


from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import video_has_audio

if platform.system().lower() == 'windows':
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster Live Portrait Pipeline')
    parser.add_argument('--src_image', required=False, type=str, default="/home/ubuntu/digital/video/WeChat1.mp4",
                        help='source image')
    parser.add_argument('--dri_video', required=False, type=str, default="/home/ubuntu/digital/video/d9_0-2_demo_feman2+_Easy-Wav2Lip.mp4",
                        help='driving video')
    parser.add_argument('--cfg', required=False, type=str, default="configs/onnx_infer.yaml", help='inference config')
    parser.add_argument('--realtime', action='store_true', help='realtime inference', default=False)
    parser.add_argument('--animal', action='store_true', help='use animal model', default=False)
    
    # 解析命令行参数
    args, unknown = parser.parse_known_args()

    # 加载配置文件
    infer_cfg = OmegaConf.load(args.cfg)

    # 初始化FasterLivePortraitPipeline类
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    
    # 准备源图像（如进行人脸检测）
    ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
    if not ret:
        print(f"no face in {args.src_image}! exit!") # 如果没有检测到人脸，退出程序
        exit(1)
        
    # 检查是否提供了驱动视频路径
    if not args.dri_video or not os.path.exists(args.dri_video):
        # read frame from camera if no driving video input
        # 如果没有提供驱动视频，从摄像头读取帧
        vcap = cv2.VideoCapture(0)
        if not vcap.isOpened():
            print("no camera found! exit!")
            exit(1)
    else:
        # 使用提供的驱动视频
        vcap = cv2.VideoCapture(args.dri_video)
        
    # 获取视频的帧率
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    print(f" ==== 驱动视频的帧数为：{fps}")
    
    # 获取源图像的高度和宽度
    h, w = pipe.src_imgs[0].shape[:2]
    
    # 创建结果保存目录，使用当前日期时间命名
    save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # render output video
    # 如果不是实时模式，初始化视频写入器
    if not args.realtime:
        # 设置编码格式为mp4v
        vout_crop = fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 定义裁剪视频的保存路径
        vsave_crop_path = os.path.join(save_dir,
                                       f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-crop.mp4")
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512 * 2, 512))
        
        # 定义原始尺寸视频的保存路径
        vsave_org_path = os.path.join(save_dir,
                                      f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-org.mp4")
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

    infer_times = []  # 用于记录每帧推理的时间
    
    # 逐帧读取驱动视频
    index = 0
    while vcap.isOpened():
        ret, frame = vcap.read()
        if not ret:
            break # 如果没有更多帧，退出循环
        
        # 记录推理的开始时间
        t0 = time.time()
        
        # 进行人像处理，返回裁剪后的视频帧和原始尺寸的视频帧
        index = index % len(pipe.src_imgs)
        dri_crop, out_crop, out_org = pipe.run(frame, pipe.src_imgs[index], pipe.src_infos[index])
        index += 1
        
        # 记录推理耗时
        infer_times.append(time.time() - t0)
        print(time.time() - t0)
        
        # 调整裁剪后的视频帧尺寸并合并驱动视频和输出视频
        dri_crop = cv2.resize(dri_crop, (512, 512))
        out_crop = np.concatenate([dri_crop, out_crop], axis=1)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        
        
        if not args.realtime:
            # 如果不是实时模式，将处理后的帧写入视频文件
            vout_crop.write(out_crop)
            out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
            vout_org.write(out_org)
        else:
            # 实时模式下，显示处理后的帧
            if infer_cfg.infer_params.flag_pasteback:
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                cv2.imshow('Render', out_org)
            else:
                # image show in realtime mode
                # 直接显示裁剪后的输出
                cv2.imshow('Render', out_crop)
            # 按下'q'键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    # 释放视频捕获资源  
    vcap.release()
    if not args.realtime:
        # 如果不是实时模式，释放视频写入资源
        vout_crop.release()
        vout_org.release()
        
        # 检查驱动视频是否包含音频
        if video_has_audio(args.dri_video):
            # 将音频合并到处理后的视频中
            vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_crop_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_crop_path_new, "-y", "-shortest"])
            vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_org_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_org_path_new, "-y", "-shortest"])

            # 输出最终视频文件路径
            print(vsave_crop_path_new)
            print(vsave_org_path_new)
        else:
            # 如果驱动视频没有音频，直接输出处理后的视频路径
            print(vsave_crop_path)
            print(vsave_org_path)
    else:
        cv2.destroyAllWindows()

    print(
        "inference median time: {} ms/frame, mean time: {} ms/frame".format(np.median(infer_times) * 1000,
                                                                            np.mean(infer_times) * 1000))
