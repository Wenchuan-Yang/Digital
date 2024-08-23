import os
import sys
import re
import argparse
from easy_functions import (format_time,
                            get_input_length,
                            get_video_details,
                            show_video,
                            g_colab)
import contextlib
import shutil
import subprocess
import time
from IPython.display import Audio, Image, clear_output, display
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import configparser
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"


# 创建ArgumentParser对象，用于从命令行解析参数
parser = argparse.ArgumentParser(description='Easy-Wav2Lip main run file')

parser.add_argument('-video_file', type=str, 
                    help='Input video file path', required=False, default="/home/ubuntu/digital/video/demo_feman1.mp4")
parser.add_argument('-vocal_file', type=str, 
                    help='Input audio file path', required=False, default="/home/ubuntu/digital/vocal/demo_feman2+.mp3")
parser.add_argument('-output_file', type=str, 
                    help='Output video file path', required=False, default="output")
args = parser.parse_args()

# retrieve variables from config.ini
# 创建ConfigParser对象，用于读取配置文件中的配置项
config = configparser.ConfigParser()

# 读取配置文件config.ini中的配置项
config.read('config.ini')

# 根据命令行参数或配置文件获取视频文件路径
if args.video_file:
    video_file = args.video_file
else:
    video_file = config['OPTIONS']['video_file']
if args.vocal_file:
    vocal_file = args.vocal_file
else:
    vocal_file = config['OPTIONS']['vocal_file']
    
# 从配置文件中获取其他配置项
quality = config['OPTIONS']['quality']
output_height = config['OPTIONS']['output_height']
wav2lip_version = config['OPTIONS']['wav2lip_version']
use_previous_tracking_data = config['OPTIONS']['use_previous_tracking_data']
nosmooth = config.getboolean('OPTIONS', 'nosmooth') # 将字符串转换为布尔值
U = config.getint('PADDING', 'U') # 获取整数值
D = config.getint('PADDING', 'D')
L = config.getint('PADDING', 'L')
R = config.getint('PADDING', 'R')
size = config.getfloat('MASK', 'size') # 获取浮点数值
feathering = config.getint('MASK', 'feathering')
mouth_tracking = config.getboolean('MASK', 'mouth_tracking')
debug_mask = config.getboolean('MASK', 'debug_mask')
batch_process = config.getboolean('OTHER', 'batch_process')
output_suffix = config['OTHER']['output_suffix']
include_settings_in_suffix = config.getboolean('OTHER', 'include_settings_in_suffix')

# 检查是否在Google Colab环境中运行，并相应设置预览选项
if g_colab():
    preview_input = config.getboolean("OTHER", "preview_input")
else:
    preview_input = False
preview_settings = config.getboolean("OTHER", "preview_settings")
frame_to_preview = config.getint("OTHER", "frame_to_preview")

# 获取当前工作目录
working_directory = os.getcwd()

# 记录开始时间，用于计算程序执行时间
start_time = time.time()

# 移除视频和音频文件路径中可能存在的引号
video_file = video_file.strip('"')
vocal_file = vocal_file.strip('"')

# check video_file exists
if video_file == "":
    sys.exit(f"video_file cannot be blank")

# 检查视频文件路径是否为空
if os.path.isdir(video_file):
    sys.exit(f"{video_file} is a directory, you need to point to a file")

if not os.path.exists(video_file):
    sys.exit(f"Could not find file: {video_file}")

# 根据Wav2Lip版本选择对应的检查点文件路径
if wav2lip_version == "Wav2Lip_GAN":
    checkpoint_path = os.path.join(working_directory, "checkpoints", "Wav2Lip_GAN.pth")
else:
    checkpoint_path = os.path.join(working_directory, "checkpoints", "Wav2Lip.pth")

# 根据 feathering 的值调整羽化参数
if feathering == 3:
    feathering = 5
if feathering == 2:
    feathering = 3

# 根据输出分辨率设置分辨率缩放比例
resolution_scale = 1
res_custom = False
if output_height == "half resolution":
    resolution_scale = 2
elif output_height == "full resolution":
    resolution_scale = 1
else:
    res_custom = True
    resolution_scale = 3

# 获取输入视频的宽度、高度、帧率和时长
in_width, in_height, in_fps, in_length = get_video_details(video_file)

# 根据分辨率缩放比例计算输出视频的高度
out_height = round(in_height / resolution_scale)

# 如果用户指定了自定义输出高度，则使用该值
if res_custom:
    out_height = int(output_height)
    
# 设置静态图片的帧率
fps_for_static_image = 30

# 检查输出文件的后缀设置，防止覆盖输入视频文件
if output_suffix == "" and not include_settings_in_suffix:
    sys.exit(
        "Current suffix settings will overwrite your input video! Please add a suffix or tick include_settings_in_suffix"
    )

# 调整要预览的帧数（确保不小于0）
frame_to_preview = max(frame_to_preview - 1, 0)

# 如果需要将配置设置包含在输出文件的后缀中，则生成后缀字符串
if include_settings_in_suffix:
    # 如果使用的是Wav2Lip_GAN版本，将后缀添加_GAN
    if wav2lip_version == "Wav2Lip_GAN":
        output_suffix = f"{output_suffix}_GAN"
    # 添加质量设置到后缀中
    output_suffix = f"{output_suffix}_{quality}"
    if output_height != "full resolution":
        output_suffix = f"{output_suffix}_{out_height}"
    # 根据nosmooth选项，将是否平滑的设置添加到后缀中
    if nosmooth:
        output_suffix = f"{output_suffix}_nosmooth1"
    else:
        output_suffix = f"{output_suffix}_nosmooth0"  
    # 如果任何边缘的填充值不为0，将填充值添加到后缀中
    if U != 0 or D != 0 or L != 0 or R != 0:
        output_suffix = f"{output_suffix}_pads-"
        if U != 0:
            output_suffix = f"{output_suffix}U{U}"
        if D != 0:
            output_suffix = f"{output_suffix}D{D}"
        if L != 0:
            output_suffix = f"{output_suffix}L{L}"
        if R != 0:
            output_suffix = f"{output_suffix}R{R}"
    # 如果质量设置不是“fast”，则将mask的大小和羽化设置添加到后缀中
    if quality != "fast":
        output_suffix = f"{output_suffix}_mask-S{size}F{feathering}"
        # 如果启用了嘴唇跟踪，则将其添加到后缀中
        if mouth_tracking:
            output_suffix = f"{output_suffix}_mt"
        # 如果启用了调试mask模式，则将其添加到后缀中
        if debug_mask:
            output_suffix = f"{output_suffix}_debug"
            
# 如果启用了预览设置选项，将预览设置添加到后缀中
if preview_settings:
    output_suffix = f"{output_suffix}_preview"


# 计算分辨率缩放比例和边距填充值，准备用于处理
rescaleFactor = str(round(1 // resolution_scale))
pad_up = str(round(U * resolution_scale))
pad_down = str(round(D * resolution_scale))
pad_left = str(round(L * resolution_scale))
pad_right = str(round(R * resolution_scale))
################################################################################


######################### 重构输入路径部分 ###################################
# 分解视频文件路径，获取文件夹路径和文件名（带扩展名）
folder, filename_with_extension = os.path.split(video_file)
# 获取文件名和文件类型（扩展名）
filename, file_type = os.path.splitext(filename_with_extension)

# Extract filenumber if it exists
# 如果文件名中存在数字后缀，提取文件号
filenumber_match = re.search(r"\d+$", filename)
if filenumber_match:  # if there is a filenumber - extract it
    filenumber = str(filenumber_match.group())
    filenamenonumber = re.sub(r"\d+$", "", filename)
else:  # if there is no filenumber - make it blank
    filenumber = ""
    filenamenonumber = filename

# if vocal_file is blank - use the video as audio
if vocal_file == "":
    vocal_file = video_file
# if not, check that the vocal_file file exists
else:
    if not os.path.exists(vocal_file):
        sys.exit(f"Could not find file: {vocal_file}")
    if os.path.isdir(vocal_file):
        sys.exit(f"{vocal_file} is a directory, you need to point to a file")

# Extract each part of the path
audio_folder, audio_filename_with_extension = os.path.split(vocal_file)
audio_filename, audio_file_type = os.path.splitext(audio_filename_with_extension)

# Extract filenumber if it exists
audio_filenumber_match = re.search(r"\d+$", audio_filename)
if audio_filenumber_match:  # if there is a filenumber - extract it
    audio_filenumber = str(audio_filenumber_match.group())
    audio_filenamenonumber = re.sub(r"\d+$", "", audio_filename)
else:  # if there is no filenumber - make it blank
    audio_filenumber = ""
    audio_filenamenonumber = audio_filename
################################################################################

# set process_failed to False so that it may be set to True if one or more processings fail
# 初始化一个布尔变量process_failed，用于标记处理过程是否失败
process_failed = False

# 设置临时输出文件和临时文件夹路径，用于中间处理结果的存储
temp_output = os.path.join(working_directory, "temp", "output.mp4")
temp_folder = os.path.join(working_directory, "temp")

# 初始化用于存储上一次输入视频和音频文件的变量
last_input_video = None
last_input_audio = None

# --------------------------Batch processing loop-------------------------------!
while True:

    # construct input_video
    # 构造输入视频路径
    input_video = os.path.join(folder, filenamenonumber + str(filenumber) + file_type)
    input_videofile = os.path.basename(input_video) # 获取视频文件名

    # construct input_audio
    # 构造输入音频路径
    input_audio = os.path.join(
        audio_folder, audio_filenamenonumber + str(audio_filenumber) + audio_file_type
    )
    input_audiofile = os.path.basename(input_audio) # 获取音频文件名

    # see if filenames are different:
    # 检查视频和音频的文件名是否不同
    if filenamenonumber + str(filenumber) != audio_filenamenonumber + str(
        audio_filenumber
    ):
        # 如果不同，则构造输出文件名为"视频文件名_音频文件名"
        output_filename = (
            filenamenonumber
            + str(filenumber)
            + "_"
            + audio_filenamenonumber
            + str(audio_filenumber)
        )
    else:
        # 如果相同，则直接使用视频文件名
        output_filename = filenamenonumber + str(filenumber)

    # construct output_video
    # 构造输出视频路径
    output_video = os.path.join(folder, output_filename + output_suffix + ".mp4")
    output_video = os.path.normpath(output_video) # 规范化路径
    output_videofile = os.path.basename(output_video) # 获取输出视频文件名

    # remove last outputs
    # 删除上次处理的输出文件
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp", exist_ok=True) # 创建临时文件夹用于存储中间结果

    # preview inputs (if enabled)
    # 如果启用了输入预览，则显示输入视频和音频（预览功能目前被禁用）
    if preview_input and 0:
        print("input video:")
        show_video(input_video)
        if vocal_file != "":
            print("input audio:")
            display(Audio(input_audio))
        else:
            print("using", input_videofile, "for audio")
        print("You may want to check now that they're the correct files!")

    # 记录上一次输入的视频和音频文件
    last_input_video = input_video
    last_input_audio = input_audio
    shutil.copy(input_video, temp_folder) # 将输入视频复制到临时文件夹
    shutil.copy(input_audio, temp_folder) # 将输入音频复制到临时文件夹

    # rename temp file to include padding or else changing padding does nothing
    # 重命名临时文件以包含填充值，否则填充设置不会生效
    temp_input_video = os.path.join(temp_folder, input_videofile)
    renamed_temp_input_video = os.path.join(
        temp_folder, str(U) + str(D) + str(L) + str(R) + input_videofile
    )
    shutil.copy(temp_input_video, renamed_temp_input_video)
    temp_input_video = renamed_temp_input_video
    temp_input_videofile = os.path.basename(renamed_temp_input_video)
    temp_input_audio = os.path.join(temp_folder, input_audiofile)

    # trim video if it's longer than the audio
    # 如果视频长度大于音频长度，剪辑视频以匹配音频长度
    video_length = get_input_length(temp_input_video)
    audio_length = get_input_length(temp_input_audio)

    # 如果启用了预览设置，则处理预览视频和音频
    if preview_settings:
        batch_process = False # 关闭批量处理模式

        preview_length_seconds = 1 # 预览时长设置为1秒
        converted_preview_frame = frame_to_preview / in_fps # 将预览帧转换为时间
        preview_start_time = min(
            converted_preview_frame, video_length - preview_length_seconds
        )

        # 构造预览视频和音频路径
        preview_video_path = os.path.join(
            temp_folder,
            "preview_"
            + str(preview_start_time)
            + "_"
            + str(U)
            + str(D)
            + str(L)
            + str(R)
            + input_videofile,
        )
        preview_audio_path = os.path.join(temp_folder, "preview_" + input_audiofile)

        # 使用ffmpeg剪辑预览视频
        subprocess.call(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                temp_input_video,
                "-ss",
                str(preview_start_time),
                "-to",
                str(preview_start_time + preview_length_seconds),
                "-c",
                "copy",
                preview_video_path,
            ]
        )
        # 使用ffmpeg剪辑预览音频
        print("   使用ffmpeg剪辑预览音频")
        subprocess.call(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                temp_input_audio,
                "-ss",
                str(preview_start_time),
                "-to",
                str(preview_start_time + 1),
                "-c",
                "copy",
                preview_audio_path,
            ]
        )
        temp_input_video = preview_video_path # 更新临时输入视频为预览视频
        temp_input_audio = preview_audio_path # 更新临时输入音频为预览音频

    # 如果视频长度大于音频长度，则剪辑视频
    if video_length > audio_length:
        trimmed_video_path = os.path.join(
            temp_folder, "trimmed_" + temp_input_videofile
        )
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
                devnull
            ):
                ffmpeg_extract_subclip(
                    temp_input_video, 0, audio_length, targetname=trimmed_video_path
                )
        temp_input_video = trimmed_video_path
    # check if face detection has already happened on this clip
    # 检查是否已经在此剪辑上进行过人脸检测
    last_detected_face = os.path.join(working_directory, "last_detected_face.pkl")
    if os.path.isfile("last_file.txt"):
        with open("last_file.txt", "r") as file:
            last_file = file.readline()
        if last_file != temp_input_video or use_previous_tracking_data == "False":
            if os.path.isfile(last_detected_face):
                os.remove(last_detected_face)

    # ----------------------------Process the inputs!-----------------------------!
    # ----------------------------处理输入视频和音频！-----------------------------!
    print(
        f"Processing{' preview of' if preview_settings else ''} "
        f"{input_videofile} using {input_audiofile} for audio"
    )

    # execute Wav2Lip & upscaler
    # 执行Wav2Lip和放大操作
    print(" ===== 开始执行Wav2Lip和放大操作 =====")
    cmd = [
        sys.executable,
        "inference.py",
        "--face",
        temp_input_video,
        "--audio",
        temp_input_audio,
        "--outfile",
        temp_output,
        "--pads",
        str(pad_up),
        str(pad_down),
        str(pad_left),
        str(pad_right),
        "--checkpoint_path",
        checkpoint_path,
        "--out_height",
        str(out_height),
        "--fullres",
        str(resolution_scale),
        "--quality",
        quality,
        "--mask_dilation",
        str(size),
        "--mask_feathering",
        str(feathering),
        "--nosmooth",
        str(nosmooth),
        "--debug_mask",
        str(debug_mask),
        "--preview_settings",
        str(preview_settings),
        "--mouth_tracking",
        str(mouth_tracking),
    ]

    print("======================================")
    print(cmd)
    print("======================================")
    # Run the command
    # 运行命令
    subprocess.run(cmd)

    # 如果启用了预览设置，检查预览是否成功
    if preview_settings:
        if os.path.isfile(os.path.join(temp_folder, "preview.jpg")):
            print(f"preview successful! Check out temp/preview.jpg")
            with open("last_file.txt", "w") as f:
                f.write(temp_input_video)
            # end processing timer and format the time it took
            # 结束处理计时并格式化时间
            end_time = time.time()
            elapsed_time = end_time - start_time
            formatted_setup_time = format_time(elapsed_time)
            print(f"Execution time: {formatted_setup_time}")
            break

        else:
            print(f"Processing failed! :( see line above 👆")
            print("Consider searching the issues tab on the github:")
            print("https://github.com/anothermartz/Easy-Wav2Lip/issues")
            exit()

    # rename temp file and move to correct directory
    # 重命名临时文件并移动到正确的目录
    if os.path.isfile(temp_output):
        if os.path.isfile(output_video):
            os.remove(output_video)
        shutil.copy(temp_output, output_video)
        # show output video
        # 显示输出视频的路径
        with open("last_file.txt", "w") as f:
            f.write(temp_input_video)
        print(f"{output_filename} successfully lip synced! It will be found here:")
        print(output_video)

        # end processing timer and format the time it took
        # 结束处理计时并格式化时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_setup_time = format_time(elapsed_time)
        print(f"Execution time: {formatted_setup_time}")

    else:
        print(f"Processing failed! :( see line above 👆")
        print("Consider searching the issues tab on the github:")
        print("https://github.com/anothermartz/Easy-Wav2Lip/issues")
        process_failed = True

    # 如果不是批量处理模式，则退出循环
    if batch_process == False:
        if process_failed:
            exit()
        else:
            break

    elif filenumber == "" and audio_filenumber == "":
        print("Files not set for batch processing")
        break

    # -----------------------------Batch Processing!------------------------------!
    # -----------------------------批量处理！------------------------------!
    # 如果视频文件有文件号，增加文件号以处理下一个视频文件
    if filenumber != "":  # if video has a filenumber
        match = re.search(r"\d+", filenumber)
        # add 1 to video filenumber
        filenumber = (
            f"{filenumber[:match.start()]}{int(match.group())+1:0{len(match.group())}d}"
        )
        
    # 如果音频文件有文件号，增加文件号以处理下一个音频文件
    if audio_filenumber != "":  # if audio has a filenumber
        match = re.search(r"\d+", audio_filenumber)
        # add 1 to audio filenumber
        audio_filenumber = f"{audio_filenumber[:match.start()]}{int(match.group())+1:0{len(match.group())}d}"

    # 构造下一个输入视频的路径
    input_video = os.path.join(folder, filenamenonumber + str(filenumber) + file_type)
    input_videofile = os.path.basename(input_video)

    # 构造下一个输入音频的路径
    input_audio = os.path.join(
        audio_folder, audio_filenamenonumber + str(audio_filenumber) + audio_file_type
    )
    input_audiofile = os.path.basename(input_audio)

    # now check which input files exist and what to do for each scenario
    # 检查下一个输入文件是否存在，并决定如何继续处理

    # both +1 files exist - continue processing
    # 如果视频和音频的+1文件都存在，继续处理
    if os.path.exists(input_video) and os.path.exists(input_audio):
        continue

    # video +1 only - continue with last audio file
    # 如果只有视频+1文件存在，继续处理最后一个音频文件
    if os.path.exists(input_video) and input_video != last_input_video:
        if audio_filenumber != "":  # if audio has a filenumber
            match = re.search(r"\d+", audio_filenumber)
            # take 1 from audio filenumber
            audio_filenumber = f"{audio_filenumber[:match.start()]}{int(match.group())-1:0{len(match.group())}d}"
        continue

    # audio +1 only - continue with last video file
    # 如果只有音频+1文件存在，继续处理最后一个视频文件
    if os.path.exists(input_audio) and input_audio != last_input_audio:
        if filenumber != "":  # if video has a filenumber
            match = re.search(r"\d+", filenumber)
            # take 1 from video filenumber
            filenumber = f"{filenumber[:match.start()]}{int(match.group())-1:0{len(match.group())}d}"
        continue

    # neither +1 files exist or current files already processed - finish processing
    # 如果没有找到下一个文件或当前文件已处理完毕，结束处理
    print("Finished all sequentially numbered files")
    if process_failed:
        sys.exit("Processing failed on at least one video")
    else:
        break
