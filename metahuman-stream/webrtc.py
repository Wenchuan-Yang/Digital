import asyncio  # 导入 asyncio 库，用于编写异步 I/O 操作
import json  # 导入 json 库，用于处理 JSON 数据
import logging  # 导入 logging 库，用于记录日志信息
import threading  # 导入 threading 库，用于多线程操作
import time  # 导入 time 库，用于时间相关操作
from typing import Tuple, Dict, Optional, Set, Union  # 导入类型提示相关的模块
from av.frame import Frame  # 从 av 库导入 Frame 类，用于处理音视频帧
from av.packet import Packet  # 从 av 库导入 Packet 类，用于处理音视频数据包
from av import AudioFrame  # 从 av 库导入 AudioFrame 类，用于处理音频帧
import fractions  # 导入 fractions 库，用于处理分数（如时间基）
import numpy as np  # 导入 numpy 库，用于数值计算

AUDIO_PTIME = 0.020  # 20ms 音频打包时间，用于定义音频帧的时间间隔
VIDEO_CLOCK_RATE = 90000  # 视频时钟速率，用于定义视频时间戳的基准单位
VIDEO_PTIME = 1 / 25  # 视频帧的时间间隔，这里设定为 25fps（每秒25帧）
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)  # 定义视频的时间基数，用于计算时间戳
SAMPLE_RATE = 16000  # 音频采样率，定义每秒钟音频采样点数
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)  # 定义音频的时间基数，用于计算时间戳

#from aiortc.contrib.media import MediaPlayer, MediaRelay  # 注释掉的代码，可能是用于处理媒体流的工具
#from aiortc.rtcrtpsender import RTCRtpSender  # 注释掉的代码，可能是用于发送 RTP 数据的工具
from aiortc import (  # 从 aiortc 导入 MediaStreamTrack 类，用于定义 WebRTC 的媒体流
    MediaStreamTrack,
)

logging.basicConfig()  # 配置基本的日志记录设置
logger = logging.getLogger(__name__)  # 创建一个日志记录器对象，记录本模块的日志信息

class PlayerStreamTrack(MediaStreamTrack):
    """
    一个视频轨道类，用于返回一个动画的帧。
    """

    def __init__(self, player, kind):
        '''
        player 是关联的播放器对象，kind 是媒体流的类型（audio 或 video）。
        '''
        super().__init__()  # 调用父类的构造函数，初始化 MediaStreamTrack
        self.kind = kind  # 存储媒体流的类型（音频或视频）
        self._player = player  # 关联一个播放器对象
        self._queue = asyncio.Queue()  # 创建一个异步队列，用于存储帧数据
        self.timelist = []  # 记录最近的包的时间戳
        if self.kind == 'video':
            self.framecount = 0  # 初始化视频帧计数器
            self.lasttime = time.perf_counter()  # 记录上一次获取帧的时间
            self.totaltime = 0  # 累积总时间

    _start: float  # 定义一个浮点型的开始时间（时间戳）
    _timestamp: int  # 定义一个整数型的时间戳

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        '''
        计算并返回当前帧的时间戳和时间基数。
        对于视频流，根据设定的帧率（如 25fps），逐帧计算时间戳，并根据需要异步等待，以保证帧之间的时间间隔一致。
        对于音频流，类似地计算时间戳，并根据音频打包时间（20ms）进行时间同步。
        该方法确保音视频帧的播放时间保持一致，从而使媒体播放流畅。
        '''
        if self.readyState != "live":
            raise Exception  # 如果媒体流不是 "live" 状态，抛出异常

        if self.kind == 'video':  # 如果是视频流
            if hasattr(self, "_timestamp"):
                #self._timestamp = (time.time()-self._start) * VIDEO_CLOCK_RATE
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)  # 计算下一个时间戳
                wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()  # 计算需要等待的时间
                if wait > 0:
                    await asyncio.sleep(wait)  # 如果需要等待，异步休眠等待
            else:
                self._start = time.time()  # 设置开始时间为当前时间
                self._timestamp = 0  # 初始化时间戳为 0
                self.timelist.append(self._start)  # 将开始时间加入时间列表
                print('video start:', self._start)  # 输出视频开始的时间
            return self._timestamp, VIDEO_TIME_BASE  # 返回计算的时间戳和视频时间基数
        else:  # 如果是音频流
            if hasattr(self, "_timestamp"):
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)  # 计算下一个时间戳
                wait = self._start + (self._timestamp / SAMPLE_RATE) - time.time()  # 计算需要等待的时间
                if wait > 0:
                    await asyncio.sleep(wait)  # 如果需要等待，异步休眠等待
            else:
                self._start = time.time()  # 设置开始时间为当前时间
                self._timestamp = 0  # 初始化时间戳为 0
                self.timelist.append(self._start)  # 将开始时间加入时间列表
                print('audio start:', self._start)  # 输出音频开始的时间
            return self._timestamp, AUDIO_TIME_BASE  # 返回计算的时间戳和音频时间基数

    async def recv(self) -> Union[Frame, Packet]:
        '''
        从异步队列 _queue 中获取下一帧的音视频数据。
        调用 next_timestamp 方法计算该帧的时间戳，并将其设置到帧的 pts 属性中。
        对视频帧进行计数，并计算帧率，以监控和输出平均帧率。
        如果队列中没有数据或帧为空，则停止播放并抛出异常。
        返回处理后的音视频帧，用于后续播放或传输。
        '''
        # frame = self.frames[self.counter % 30]
        self._player._start(self)  # 启动播放器，关联当前的媒体流轨道
        frame = await self._queue.get()  # 从队列中获取下一帧数据
        pts, time_base = await self.next_timestamp()  # 计算下一帧的时间戳和时间基数
        frame.pts = pts  # 设置帧的时间戳
        frame.time_base = time_base  # 设置帧的时间基数
        if frame is None:
            self.stop()  # 如果帧为空，停止媒体流
            raise Exception  # 抛出异常
        if self.kind == 'video':  # 如果是视频流
            self.totaltime += (time.perf_counter() - self.lasttime)  # 计算总的处理时间
            self.framecount += 1  # 视频帧计数器加 1
            self.lasttime = time.perf_counter()  # 更新最后一帧的处理时间
            if self.framecount == 100:  # 每 100 帧输出一次平均帧率
                print(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0  # 重置帧计数器
                self.totaltime = 0  # 重置总时间
        return frame  # 返回当前帧

    def stop(self):
        super().stop()  # 调用父类的 stop 方法，停止媒体流
        if self._player is not None:
            self._player._stop(self)  # 停止关联的播放器
            self._player = None  # 解除对播放器的引用

def player_worker_thread(
    quit_event,
    loop,
    container,
    audio_track,
    video_track
):
    '''
    在一个单独的线程中运行，用于处理音视频数据的渲染或传输。
    接收 quit_event 用于通知线程何时退出，loop 是 asyncio 的事件循环，container 是用于处理音视频流的容器，audio_track 和 video_track 是音频和视频轨道。
    通过调用 container.render 方法，处理传入的音视频轨道，并将其发送或播放。
    '''
    container.render(quit_event, loop, audio_track, video_track)  # 在单独的线程中渲染媒体内容

class HumanPlayer:
    def __init__(
        self, nerfreal, format=None, options=None, timeout=None, loop=False, decode=True):
        '''
        接收多个可选参数，如格式、选项、超时时间、是否循环播放、是否解码等。
        初始化实例的多个属性，包括线程对象、退出事件、音视频轨道、容器对象等。
        创建并初始化 PlayerStreamTrack 类的音频轨道 __audio 和视频轨道 __video。
        '''
        self.__thread: Optional[threading.Thread] = None  # 定义一个可选的线程对象，初始化为空
        self.__thread_quit: Optional[threading.Event] = None  # 定义一个可选的线程退出事件，初始化为空

        # examine streams
        self.__started: Set[PlayerStreamTrack] = set()  # 定义一个集合，用于存储已启动的媒体轨道
        self.__audio: Optional[PlayerStreamTrack] = None  # 定义音频轨道，初始化为空
        self.__video: Optional[PlayerStreamTrack] = None  # 定义视频轨道，初始化为空

        self.__audio = PlayerStreamTrack(self, kind="audio")  # 创建音频轨道
        self.__video = PlayerStreamTrack(self, kind="video")  # 创建视频轨道

        self.__container = nerfreal  # 关联一个容器对象

    @property
    def audio(self) -> MediaStreamTrack:
        """
        返回音频轨道对象。
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        返回视频轨道对象。
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        '''
        启动音视频流的播放或传输。
        将传入的轨道 track 添加到已启动轨道的集合 __started 中。
        如果线程未启动，则创建并启动一个新的线程，该线程用于处理音视频数据的渲染或传输。
        该方法确保音视频流能够在独立的线程中正常启动并播放。
        '''
        self.__started.add(track)  # 将轨道加入已启动的轨道集合
        if self.__thread is None:  # 如果线程未启动
            self.__log_debug("Starting worker thread")  # 记录日志，表示开始启动线程
            self.__thread_quit = threading.Event()  # 创建一个线程退出事件
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(
                    self.__thread_quit,
                    asyncio.get_event_loop(),
                    self.__container,
                    self.__audio,
                    self.__video                   
                ),
            )
            self.__thread.start()  # 启动线程

    def _stop(self, track: PlayerStreamTrack) -> None:
        '''
        停止音视频流的播放或传输。
        将传入的轨道 track 从已启动轨道的集合 __started 中移除。
        如果所有轨道都已停止，并且线程存在，则触发退出事件并等待线程结束，随后清除线程对象。
        该方法确保音视频流在停止时能够正确清理资源，并结束相关线程。
        '''
        self.__started.discard(track)  # 从已启动轨道集合中移除轨道

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")  # 记录日志，表示停止线程
            self.__thread_quit.set()  # 触发退出事件，通知线程退出
            self.__thread.join()  # 等待线程结束
            self.__thread = None  # 清除线程对象

        if not self.__started and self.__container is not None:
            #self.__container.close()
            self.__container = None  # 清除容器对象

    def __log_debug(self, msg: str, *args) -> None:
        '''
        记录调试日志信息。
        该方法格式化并输出 HumanPlayer 实例的调试信息，以便开发者进行调试和问题排查。
        接收一个字符串 msg 和可变参数 *args，用于组合成完整的日志信息。
        '''
        logger.debug(f"HumanPlayer {msg}", *args)  # 记录调试日志信息
