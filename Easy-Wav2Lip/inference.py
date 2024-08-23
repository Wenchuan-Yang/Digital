import torch
import numpy as np
from PIL import Image
import argparse
import configparser
import math
import os
import subprocess
import pickle
import cv2
import audio
from batch_face import RetinaFace
import re
from functools import partial
from tqdm import tqdm
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.transforms.functional_tensor"
)
from enhance import upscale
from enhance import load_sr
from easy_functions import load_model, g_colab

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
gpu_id = 0 if torch.cuda.is_available() else -1

if device == 'cpu':
    print('Warning: No GPU detected so inference will be done on the CPU which is VERY SLOW!')
parser = argparse.ArgumentParser(
    description="Inference code to lip-sync videos in the wild using Wav2Lip models"
)

args = None
parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Name of saved checkpoint to load weights from",
    # required=True,
    default="/home/ubuntu/digital/Easy-Wav2Lip/checkpoints/Wav2Lip.pth"
)

parser.add_argument(
    "--segmentation_path",
    type=str,
    default="checkpoints/face_segmentation.pth",
    help="Name of saved checkpoint of segmentation network",
    required=False,
)

parser.add_argument(
    "--face",
    type=str,
    help="Filepath of video/image that contains faces to use",
    # required=True,
    default="/home/ubuntu/digital/video/demo_feman1.mp4",
)
parser.add_argument(
    "--audio",
    type=str,
    help="Filepath of video/audio file to use as raw audio source",
    # required=True,
    default="/home/ubuntu/digital/vocal/s3_1.mp3",
)
parser.add_argument(
    "--outfile",
    type=str,
    help="Video path to save result. See default for an e.g.",
    default="results/result_voice.mp4",
)

parser.add_argument(
    "--static",
    type=bool,
    help="If True, then use only first video frame for inference",
    default=False,
)
parser.add_argument(
    "--fps",
    type=float,
    help="Can be specified only if input is a static image (default: 25)",
    default=25.0,
    required=False,
)

parser.add_argument(
    "--pads",
    nargs="+",
    type=int,
    default=[0, 10, 0, 0],
    help="Padding (top, bottom, left, right). Please adjust to include chin at least",
)

parser.add_argument(
    "--wav2lip_batch_size", type=int, help="Batch size for Wav2Lip model(s)", default=1
)

parser.add_argument(
    "--out_height",
    default=480,
    type=int,
    help="Output video height. Best results are obtained at 480 or 720",
)

parser.add_argument(
    "--crop",
    nargs="+",
    type=int,
    default=[0, -1, 0, -1],
    help="Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. "
    "Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width",
)

parser.add_argument(
    "--box",
    nargs="+",
    type=int,
    default=[-1, -1, -1, -1],
    help="Specify a constant bounding box for the face. Use only as a last resort if the face is not detected."
    "Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).",
)

parser.add_argument(
    "--rotate",
    default=False,
    action="store_true",
    help="Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg."
    "Use if you get a flipped result, despite feeding a normal looking video",
)

parser.add_argument(
    "--nosmooth",
    type=str,
    default=True,
    help="Prevent smoothing face detections over a short temporal window",
)

parser.add_argument(
    "--no_seg",
    default=False,
    action="store_true",
    help="Prevent using face segmentation",
)

parser.add_argument(
    "--no_sr", default=False, action="store_true", help="Prevent using super resolution"
)

parser.add_argument(
    "--sr_model",
    type=str,
    default="gfpgan",
    help="Name of upscaler - gfpgan or RestoreFormer",
    required=False,
)

parser.add_argument(
    "--fullres",
    default=1,
    type=int,
    help="used only to determine if full res is used so that no resizing needs to be done if so",
)

parser.add_argument(
    "--debug_mask",
    type=str,
    default=False,
    help="Makes background grayscale to see the mask better",
)

parser.add_argument(
    "--preview_settings", type=str, default=False, help="Processes only one frame"
)

parser.add_argument(
    "--mouth_tracking",
    type=str,
    default=False,
    help="Tracks the mouth in every frame for the mask",
)

parser.add_argument(
    "--mask_dilation",
    default=2.5,
    type=float,
    help="size of mask around mouth",
    required=False,
)

parser.add_argument(
    "--mask_feathering",
    default=3,
    type=int,
    help="amount of feathering of mask around mouth",
    required=False,
)

parser.add_argument(
    "--quality",
    type=str,
    help="Choose between Fast, Improved and Enhanced",
    default="Fast",
)

with open(os.path.join("checkpoints", "predictor.pkl"), "rb") as f:
    predictor = pickle.load(f)

with open(os.path.join("checkpoints", "mouth_detector.pkl"), "rb") as f:
    mouth_detector = pickle.load(f)

# creating variables to prevent failing when a face isn't detected
kernel = last_mask = x = y = w = h = None

g_colab = g_colab()

print(f" g_colab = {g_colab}")
if not g_colab:
  # Load the config file
  config = configparser.ConfigParser()
  config.read('config.ini')

  # Get the value of the "preview_window" variable
  preview_window = config.get('OPTIONS', 'preview_window')

all_mouth_landmarks = []

model = detector = detector_model = None

def do_load(checkpoint_path):
    global model, detector, detector_model
    model = load_model(checkpoint_path)
    detector = RetinaFace(
        gpu_id=gpu_id, model_path="checkpoints/mobilenet.pth", network="mobilenet"
    )
    detector_model = detector.model

def face_rect(images):
    # 定义每批次处理的图像数量，设置为8
    face_batch_size = 8
    # 计算总批次数量，使用math.ceil确保有余数时多处理一批
    num_batches = math.ceil(len(images) / face_batch_size)
    
    # 初始化一个变量来保存上一次检测到的脸部框的返回值
    prev_ret = None
    
    # 逐批处理图像
    for i in range(num_batches):
        # 从图像列表中提取当前批次的图像
        batch = images[i * face_batch_size : (i + 1) * face_batch_size]
        # 使用预定义的`detector`函数对当前批次的图像进行人脸检测
        # `detector`函数返回一个列表，其中每个元素是对应图像的人脸检测结果
        all_faces = detector(batch)  # 返回所有图像的人脸列表
        
        # 遍历当前批次中所有图像的人脸检测结果
        for faces in all_faces:
            # 如果在图像中检测到人脸
            if faces:
                # 提取检测到的第一个人脸的边框、关键点和置信度分数
                box, landmarks, score = faces[0]
                # 将边框的坐标转换为整数，并将其保存到prev_ret变量中
                prev_ret = tuple(map(int, box))
                
            # 使用yield语句返回当前图像的人脸边框，如果没有检测到人脸则返回上一次的检测结果
            yield prev_ret

def create_tracked_mask(img, original_img):
    '''
    这个函数的主要作用是在图像中检测嘴部区域，并基于检测结果生成一个平滑的mask，
    用于将处理后的图像粘贴到原始图像上，从而实现图像融合效果。
    '''
    
    # 声明全局变量，供函数内部使用和更新
    global kernel, last_mask, x, y, w, h  # last_mask保存上一次成功生成的mask

    # Convert color space from BGR to RGB if necessary
    # 将图像从BGR色彩空间转换为RGB色彩空间
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

    # Detect face
    # 检测图像中的人脸
    faces = mouth_detector(img)
    if len(faces) == 0: # 如果未检测到人脸
        if last_mask is not None: # 如果之前有成功生成的mask
            last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0])) # 调整mask大小以适应当前图像
            mask = last_mask  # 使用上一次的mask
        else:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img) # 将图像转换回BGR色彩空间
            return img, None # 返回原始图像和None作为mask
    # 如果检测到人脸
    else:
        face = faces[0] # 获取检测到的第一张人脸
        shape = predictor(img, face) # 获取面部特征

        # Get points for mouth
        # 提取嘴部的特征点（68点中的48到68点）
        mouth_points = np.array(
            [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
        )

        # Calculate bounding box dimensions
        # 计算嘴部的边界框维度（x, y为左上角坐标，w, h为宽度和高度）
        x, y, w, h = cv2.boundingRect(mouth_points)

        # Set kernel size as a fraction of bounding box size
        # 设置kernel的大小，作为边界框大小的一部分
        kernel_size = int(max(w, h) * args.mask_dilation)
        # if kernel_size % 2 == 0:  # Ensure kernel size is odd # 确保kernel的大小为奇数
        # kernel_size += 1

        # Create kernel
        # 创建kernel（用于膨胀操作）
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Create binary mask for mouth
        # 为嘴部创建二值mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, mouth_points, 255)

        last_mask = mask  # 更新last_mask为当前生成的mask

    # Dilate the mask
    # 膨胀mask，使嘴部区域稍微扩大
    dilated_mask = cv2.dilate(mask, kernel)

    # Calculate distance transform of dilated mask
    # 计算膨胀后的mask的距离变换
    dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

    # Normalize distance transform
    # 归一化距离变换的结果，使其范围在0到255之间
    cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

    # Convert normalized distance transform to binary mask and convert it to uint8
    # 将归一化后的距离变换转换为二值mask，并转换为uint8类型
    _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
    masked_diff = masked_diff.astype(np.uint8)

    # make sure blur is an odd number
    blur = args.mask_feathering
    if blur % 2 == 0:
        blur += 1
    # Set blur size as a fraction of bounding box size
    # 设置模糊大小，作为边界框大小的一部分
    blur = int(max(w, h) * blur)  # 10% of bounding box size
    if blur % 2 == 0:  # Ensure blur size is odd
        blur += 1
    masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

    # Convert numpy arrays to PIL Images
    # 将numpy数组转换为PIL图像
    input1 = Image.fromarray(img)
    input2 = Image.fromarray(original_img)

    # Convert mask to single channel where pixel values are from the alpha channel of the current mask
    # 将mask转换为单通道图像，其中像素值来自当前mask的alpha通道
    mask = Image.fromarray(masked_diff)

    # Ensure images are the same size
    # 确保输入图像和mask的尺寸一致
    assert input1.size == input2.size == mask.size

    # Paste input1 onto input2 using the mask
    # 使用mask将input1（处理后的图像）粘贴到input2（原始图像）上
    input2.paste(input1, (0, 0), mask)

    # Convert the final PIL Image back to a numpy array
    
    input2 = np.array(input2)

    # input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
    cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)

    return input2, mask


def create_mask(img, original_img):
    '''
    该函数的主要目的是在输入图像中检测嘴部区域，并根据检测结果生成一个平滑的mask，
    用于将处理后的图像粘贴到原始图像上，从而实现图像融合效果。如果有可用的last_mask，它会被重用以节省计算资源。
    '''
    global kernel, last_mask, x, y, w, h # Add last_mask to global variables

    # Convert color space from BGR to RGB if necessary
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

    if last_mask is not None:
        last_mask = np.array(last_mask)  # Convert PIL Image to numpy array
        last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
        mask = last_mask  # use the last successful mask
        mask = Image.fromarray(mask)

    else:
        # Detect face
        faces = mouth_detector(img)
        if len(faces) == 0:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img, None
        else:
            face = faces[0]
            shape = predictor(img, face)

            # Get points for mouth
            mouth_points = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
            )

            # Calculate bounding box dimensions
            x, y, w, h = cv2.boundingRect(mouth_points)

            # Set kernel size as a fraction of bounding box size
            kernel_size = int(max(w, h) * args.mask_dilation)
            # if kernel_size % 2 == 0:  # Ensure kernel size is odd
            # kernel_size += 1

            # Create kernel
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Create binary mask for mouth
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, mouth_points, 255)

            # Dilate the mask
            dilated_mask = cv2.dilate(mask, kernel)

            # Calculate distance transform of dilated mask
            dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

            # Normalize distance transform
            cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

            # Convert normalized distance transform to binary mask and convert it to uint8
            _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
            masked_diff = masked_diff.astype(np.uint8)

            if not args.mask_feathering == 0:
                blur = args.mask_feathering
                # Set blur size as a fraction of bounding box size
                blur = int(max(w, h) * blur)  # 10% of bounding box size
                if blur % 2 == 0:  # Ensure blur size is odd
                    blur += 1
                masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

            # Convert mask to single channel where pixel values are from the alpha channel of the current mask
            mask = Image.fromarray(masked_diff)

            last_mask = mask  # Update last_mask with the final mask after dilation and feathering

    # Convert numpy arrays to PIL Images
    input1 = Image.fromarray(img)
    input2 = Image.fromarray(original_img)

    # Resize mask to match image size
    # mask = Image.fromarray(mask)
    mask = mask.resize(input1.size)

    # Ensure images are the same size
    assert input1.size == input2.size == mask.size

    # Paste input1 onto input2 using the mask
    input2.paste(input1, (0, 0), mask)

    # Convert the final PIL Image back to a numpy array
    input2 = np.array(input2)

    # input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
    cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)

    return input2, mask


def get_smoothened_boxes(boxes, T):
    '''
    这个函数的主要用途是对一系列检测到的边界框进行平滑处理，以减少抖动或噪声，
    使边界框在连续帧中移动更加平滑和稳定。常用于视频处理、对象跟踪等任务中。
    
    参数:
    - boxes：表示多个边界框的数组，通常是一个二维数组，其中每一行是一个边界框的坐标。
    - T：平滑窗口的大小，即在计算平滑值时考虑的边界框数量。

    '''
    # 遍历每个box（边界框）并应用平滑处理
    for i in range(len(boxes)):
        # 检查当前索引加上窗口大小T是否超出了boxes的长度
        if i + T > len(boxes):
            # 如果超出长度，则取从当前索引到最后T个box作为窗口
            window = boxes[len(boxes) - T :]
        else:
            # 否则，取从当前索引开始的T个box作为窗口
            window = boxes[i : i + T]
        
        # 对窗口内的boxes计算平均值，并将该平均值替换当前box
        boxes[i] = np.mean(window, axis=0)
    
    # 返回经过平滑处理的boxes
    return boxes

            
def face_detect(images, results_file="last_detected_face.pkl"):
    # 如果结果文件存在，则加载并返回结果，避免重新进行人脸检测
    if os.path.exists(results_file):
        print("Using face detection data from last input")
        with open(results_file, "rb") as f:
            return pickle.load(f)

    results = []  # 用于存储每个图像的人脸检测结果
    pady1, pady2, padx1, padx2 = args.pads  # 从全局变量args中获取用于扩展边界框的padding值
    
    # 使用tqdm显示进度条并部分应用于for循环
    tqdm_partial = partial(tqdm, position=0, leave=True)
    
    # 遍历图像和检测到的人脸边界框
    for image, (rect) in tqdm_partial(
        zip(images, face_rect(images)),  # 将图像和对应的边界框配对
        total=len(images),  # 总共处理的图像数量，用于tqdm显示进度
        desc="detecting face in every frame",  # 进度条描述
        ncols=100,  # 进度条的宽度
    ):
        if rect is None:  # 如果没有检测到人脸
            # 将没有检测到人脸的帧保存下来，便于后续检查
            cv2.imwrite("temp/faulty_frame.jpg", image)
            # 抛出异常并停止处理，提醒用户视频的所有帧中都应包含人脸
            raise ValueError(
                "Face not detected! Ensure the video contains a face in all the frames."
            )

        # 根据padding值调整边界框的坐标，确保边界框不会超出图像边界
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        # 将调整后的边界框坐标添加到结果列表中
        results.append([x1, y1, x2, y2])

    # 将结果转换为numpy数组，以便后续操作
    boxes = np.array(results)
    
    # 如果全局变量`args.nosmooth`为`False`，则对边界框进行平滑处理
    if str(args.nosmooth) == "False":
        boxes = get_smoothened_boxes(boxes, T=5)  # 平滑窗口大小为5
    
    # 根据平滑后的边界框从每个图像中裁剪出人脸区域，并将其与对应的边界框坐标一起存储
    results = [
        [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
        for image, (x1, y1, x2, y2) in zip(images, boxes)
    ]

    # 将结果保存到文件中，以便下次使用时直接加载
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    return results  # 返回裁剪后的人脸图像及其对应的边界框坐标


def datagen(frames, mels):
    '''
    一个生成器函数，用于在批处理中生成模型输入所需的数据。该函数接收视频帧(frames)和对应的梅尔频谱图(mels),
    然后进行人脸检测或使用指定的边界框来提取脸部区域，将图像和音频数据进行预处理并组织成批次，最终作为生成器输出
    '''
    # 初始化用于存储图像、梅尔频谱图、帧和坐标的批次列表
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    # 清空上一行的输出（打印进度用）
    print("\r" + " " * 100, end="\r")

    # 如果未指定检测框（即args.box[0] == -1），则需要进行人脸检测
    if args.box[0] == -1:
        if not args.static:  # 如果视频是动态的（非静态）
            face_det_results = face_detect(frames)  # 对每一帧进行人脸检测（BGR转RGB）
        else:  # 如果是静态视频
            face_det_results = face_detect([frames[0]])  # 只检测第一帧的人脸
    else:
        # 如果指定了边界框，直接使用指定的框来提取脸部区域
        print("Using the specified bounding box instead of face detection...")
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    # 遍历所有梅尔频谱图
    for i, m in enumerate(mels):
        # 如果是静态视频，始终使用第一帧；否则，使用循环的方式选择帧
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()  # 复制当前帧以保存

        # 获取当前帧中检测到的人脸和相应的坐标
        face, coords = face_det_results[idx].copy()

        # 将检测到的人脸调整为指定的图像大小
        face = cv2.resize(face, (args.img_size, args.img_size))

        # 将人脸图像、梅尔频谱图、帧和坐标添加到各自的批次列表中
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        # 当批次大小达到指定的batch size时，准备批次数据进行生成
        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            # 创建遮挡部分人脸的掩码
            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2 :] = 0  # 将人脸的一半遮挡

            # 将原始人脸图像与遮挡后的图像进行拼接，并将像素值归一化到[0,1]范围
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0

            # 将梅尔频谱图调整为4维数组，增加一个通道维度
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )

            # 生成当前批次的图像、梅尔频谱图、帧和坐标
            yield img_batch, mel_batch, frame_batch, coords_batch

            # 清空批次列表，为下一个批次做准备
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    # 处理剩余的样本（最后一个不满批次的样本）
    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        # 创建遮挡部分人脸的掩码
        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2 :] = 0  # 将人脸的一半遮挡

        # 将原始人脸图像与遮挡后的图像进行拼接，并将像素值归一化到[0,1]范围
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0

        # 将梅尔频谱图调整为4维数组，增加一个通道维度
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )

        # 生成最后一个批次的图像、梅尔频谱图、帧和坐标
        yield img_batch, mel_batch, frame_batch, coords_batch


mel_step_size = 16  # 定义梅尔频谱图步长，可能用于控制音频处理的步进大小

def _load(checkpoint_path):
    # 检查当前使用的设备（CPU或其他设备，如GPU）
    if device != "cpu":
        # 如果设备不是CPU（例如是GPU），直接加载检查点
        checkpoint = torch.load(checkpoint_path)
    else:
        # 如果设备是CPU，则指定map_location来将检查点加载到CPU上
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
    return checkpoint  # 返回加载的检查点


def main(arg, temp_audio_file_path):
    global args
    args = arg
    args.audio = temp_audio_file_path
    print(f"===dd11= args.audio = {args.audio}")
    args.img_size = 96 # 设置图像大小为96*96像素
    frame_number = 11 # 定义帧号为11, 可能在后续处理中使用
    

    # 检查提供的`--face`参数是否是一个图像文件（jpg, png, jpeg格式）
    if os.path.isfile(args.face) and args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
        args.static = True

    # 如果提供的`--face`参数不是有效的文件路径，则抛出错误
    if not os.path.isfile(args.face):
        raise ValueError("--face argument must be a valid path to video/image file")

    # 如果`--face`参数是一个图像文件
    elif args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
        full_frames = [cv2.imread(args.face)] # 读取图像并存储在`full_frames`列表中
        fps = args.fps # 将帧率设置为用户指定的`args.fps`

    # 如果`--face`参数是一个视频文件
    else:
        if args.fullres != 1:
            print("Resizing video...") # 如果未使用全分辨率，则打印提示信息
            
        video_stream = cv2.VideoCapture(args.face) # 打开视频文件
        fps = video_stream.get(cv2.CAP_PROP_FPS) # 获取视频文件

        full_frames = [] # 初始化用于存储视频帧的列表
        while 1:
            still_reading, frame = video_stream.read() # 读取视频的每一帧
            if not still_reading:
                video_stream.release() # 如果没有更多帧可读取，则释放视频资源
                break

            if args.fullres != 1: # 如果未使用全分辨率
                aspect_ratio = frame.shape[1] / frame.shape[0] # 计算帧的宽高比
                frame = cv2.resize(
                    frame, (int(args.out_height * aspect_ratio), args.out_height)
                ) # 调整帧的大小以适应指定的输出高度

            if args.rotate: # 如果需要旋转帧
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            # 裁剪帧到指定区域
            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1] # 如果x2为-1，则设置为帧的宽度
            if y2 == -1:
                y2 = frame.shape[0] # 如果y2为-1，则设置为帧的高度

            frame = frame[y1:y2, x1:x2] # 裁剪帧到指定的矩形区域

            full_frames.append(frame) # 将处理后的帧添加到`full_frames`列表中

    print(f"===11= args.audio = {args.audio}")
    if not args.audio.endswith(".wav"):
        # 如果输入的音频文件不是.wav格式，转换为.wav格式
        print("Converting audio to .wav")
        subprocess.check_call(
            [
                "ffmpeg",
                "-y", # 覆盖输出文件（如果存在）
                "-loglevel", # 将日志级别设置为错误，只显示错误信息
                "error",
                "-i", # 输入音频文件
                args.audio,
                "temp/temp.wav", # 输出为临时的.wav文件
            ]
        )
        args.audio = "temp/temp.wav"

    print("analysing audio...")
    print(f"==== args.audio = {args.audio}")
    wav = audio.load_wav(args.audio, 16000)  # 加载音频文件并将采样率调整为16000Hz
    mel = audio.melspectrogram(wav) # 生成音频的梅尔频谱图

    if np.isnan(mel.reshape(-1)).sum() > 0:
        # 如果梅尔频谱图中包含NaN值，抛出错误并提示用户可能需要添加噪声处理
        raise ValueError(
            "Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again"
        )

    mel_chunks = [] # 初始化梅尔频谱图块的列表

    mel_idx_multiplier = 80.0 / fps # 计算梅尔频谱图与视频帧率的比例
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier) # 计算当前帧对应的梅尔频谱图起始索引
        if start_idx + mel_step_size > len(mel[0]):
            # 如果超过了梅尔频谱图的长度，添加最后一块并退出循环
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
            break
        # 添加当前块的梅尔频谱图
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
        
    # 将视频帧数裁剪到与梅尔频谱图块数相同
    full_frames = full_frames[: len(mel_chunks)]
    
    # 如果启用了预览模式，只处理第一帧和第一个梅尔频谱图块
    if str(args.preview_settings) == "True":
        full_frames = [full_frames[0]]
        mel_chunks = [mel_chunks[0]]
        
    print(str(len(full_frames)) + " frames to process") # 输出处理的帧数
    batch_size = args.wav2lip_batch_size # 设置批处理大小
    
    if str(args.preview_settings) == "True":
        gen = datagen(full_frames, mel_chunks) # 生成用于预览的数据
    else:
        gen = datagen(full_frames.copy(), mel_chunks) # 生成用于处理的数据

    # 遍历生成的数据批次，并逐批处理
    for i, (img_batch, mel_batch, frames, coords) in enumerate(
        tqdm(
            gen,
            total=int(np.ceil(float(len(mel_chunks)) / batch_size)),
            desc="Processing Wav2Lip", # 进度条描述
            ncols=100, # 设置进度条宽度
        )
    ):
        if i == 0:
            if not args.quality == "Fast":
                # 如果质量不是“Fast”，输出mask大小和羽化设置
                print(
                    f"mask size: {args.mask_dilation}, feathering: {args.mask_feathering}"
                )
                if not args.quality == "Improved":
                    # 如果质量不是"Improved"，加载超分辨率模型
                    print("Loading", args.sr_model)
                    run_params = load_sr()

            print("Starting...") # 开始处理
            frame_h, frame_w = full_frames[0].shape[:-1] # 获取帧的高度和宽度
            fourcc = cv2.VideoWriter_fourcc(*"mp4v") # 设置视频编码格式
            out = cv2.VideoWriter("temp/result.mp4", fourcc, fps, (frame_w, frame_h)) # 创建视频输出文件

        # 将图像批次从NHWC格式转换为NCHW格式，并转换为张量
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        # 使用模型进行推理
        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        # 将预测结果从张量转换为numpy数组，并调整格式为NHWC
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for p, f, c in zip(pred, frames, coords):
            # cv2.imwrite('temp/f.jpg', f)
            # 遍历每个预测结果和对应的帧、坐标
            # cv2.imwrite('temp/f.jpg', f)  # 可选：保存帧以检查

            y1, y2, x1, x2 = c # 获取坐标

            # 如果启用了调试mask，转换背景为黑白
            if (str(args.debug_mask) == "True"):  # makes the background black & white so you can see the mask better
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))  # 调整预测结果的大小以匹配帧
            cf = f[y1:y2, x1:x2] # 裁剪原始帧的脸部区域

            if args.quality == "Enhanced":
                p = upscale(p, run_params) # 如果质量为"Enhanced"，进行超分辨率处理

            if args.quality in ["Enhanced", "Improved"]:
                if str(args.mouth_tracking) == "True":
                    p, last_mask = create_tracked_mask(p, cf) # 使用追踪的mask创建脸部区域
                else:
                    p, last_mask = create_mask(p, cf) # 将处理后的脸部区域放回原始帧中

            f[y1:y2, x1:x2] = p

            if not g_colab and 0:
                # Display the frame
                # 如果不是在Colab中运行并且0为False，显示预览窗口
                if preview_window == "Face":
                    cv2.imshow("face preview - press Q to abort", p)
                elif preview_window == "Full":
                    cv2.imshow("full preview - press Q to abort", f)
                elif preview_window == "Both":
                    cv2.imshow("face preview - press Q to abort", p)
                    cv2.imshow("full preview - press Q to abort", f)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    exit()  # Exit the loop when 'Q' is pressed

            if str(args.preview_settings) == "True":
                # 如果启用了预览设置，保存并显示处理结果
                cv2.imwrite("temp/preview.jpg", f)
                if not g_colab:
                    cv2.imshow("preview - press Q to close", f)
                    if cv2.waitKey(-1) & 0xFF == ord('q'):
                        exit()  # Exit the loop when 'Q' is pressed

            else:
                out.write(f)

    # # Close the window(s) when done
    # cv2.destroyAllWindows()

    out.release()
    # 释放VideoWriter对象，完成视频文件的写入操作

    if str(args.preview_settings) == "False":
        print("converting to final video")
        # 如果未启用预览模式，则继续处理最终的视频生成
        '''将处理后的视频文件和音频文件合成为一个最终输出的视频文件'''
        subprocess.check_call([
            "ffmpeg",
            "-y", # 覆盖输出文件（如果已存在）
            "-loglevel",  # 设置日志级别为错误，隐藏所有非错误信息
            "error",
            "-i",  
            "temp/result.mp4", # 输入处理后的无声视频文件
            "-i",
            args.audio,  # 输入原始音频文件
            "-c:v",
            "libx264",
            args.outfile  # 输出最终的视频文件，路径由args.outfile指定
        ])
        print(f"输出视频的路径：{args.outfile}")
        
    return args.outfile

if __name__ == "__main__":
    import time
    args = parser.parse_args()
    time1 = time.time()
    do_load(args.checkpoint_path)
    main()
    
    # time2 = time.time()
    # print(f"        do_load()耗时：{time.time() - time1}")
    
    # args.audio = "/home/ubuntu/digital/vocal/hello.mp3"
    # args.outfile = "/home/ubuntu/digital/Easy-Wav2Lip/results/result_voice1.mp4"
    # main()
    # print(f"        main()耗时：{time.time() - time2}")
    # args.audio = "/home/ubuntu/digital/vocal/s3_1.mp3"
    # args.outfile = "/home/ubuntu/digital/Easy-Wav2Lip/results/result_voice2.mp4"
    # time2 = time.time()
    # main()
    # print(f"        main()耗时：{time.time() - time2}")
