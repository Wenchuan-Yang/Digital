var pc = null;  // 全局变量 `pc`，用于存储 RTCPeerConnection 对象的实例

// 函数 negotiate() 用于创建 WebRTC offer ，发送到服务器，并处理服务器返回的 answer
function negotiate() {
    // 向 PeerConnection 添加视频和音频的 transceiver（发送接收器），设置为只接收模式
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });

    // 创建 offer，并将其设置为本地描述
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // 等待 ICE gathering（ICE 候选者收集）过程完成
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                // 如果 ICE 收集已经完成，直接解析 Promise
                resolve();
            } else {
                // 否则监听 ICE gathering 的状态变化
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        // 当 ICE gathering 完成时，移除事件监听器并解析 Promise
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                // 添加 ICE gathering 状态变化的事件监听器
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        // 获取本地描述（offer），并将其发送到服务器
        var offer = pc.localDescription;
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,  // 发送 SDP（Session Description Protocol）数据
                type: offer.type,  // 发送描述类型（通常为 'offer'）
            }),
            headers: {
                'Content-Type': 'application/json'  // 设置请求头，指定内容类型为 JSON
            },
            method: 'POST'  // 发送 POST 请求
        });
    }).then((response) => {
        // 从服务器响应中获取 JSON 数据
        return response.json();
    }).then((answer) => {
        // 从服务器的响应中提取 sessionid 并设置到页面的隐藏字段中
        document.getElementById('sessionid').value = answer.sessionid;
        // 设置远程描述（服务器返回的 answer）
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        // 如果在流程中发生错误，则显示警告弹窗
        alert(e);
    });
}

// 函数 start() 用于初始化 RTCPeerConnection，处理媒体流，并开始协商
function start() {
    var config = {
        sdpSemantics: 'unified-plan'  // 设置 SDP 语义为 unified-plan，这是 WebRTC 的默认配置
    };

    // 如果用户选择使用 STUN 服务器，则将其配置到 ICE 服务器列表中
    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];  // Google 的公开 STUN 服务器
    }

    // 创建一个新的 RTCPeerConnection 实例，使用指定的配置
    pc = new RTCPeerConnection(config);

    // 监听 `track` 事件，当收到远程媒体流时，将其绑定到相应的音视频元素上
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video') {
            // 如果是视频流，将其绑定到页面中的视频元素上
            document.getElementById('video').srcObject = evt.streams[0];
        } else {
            // 如果是音频流，将其绑定到页面中的音频元素上
            document.getElementById('audio').srcObject = evt.streams[0];
        }
    });

    // 隐藏 “Start” 按钮，因为连接过程已经开始
    document.getElementById('start').style.display = 'none';
    // 调用 negotiate() 函数开始协商 WebRTC 连接
    negotiate();
    // 显示 “Stop” 按钮，以便用户可以终止连接
    document.getElementById('stop').style.display = 'inline-block';
}

// 函数 stop() 用于停止 WebRTC 连接
function stop() {
    // 隐藏 “Stop” 按钮，因为连接即将关闭
    document.getElementById('stop').style.display = 'none';

    // 延迟 500 毫秒后关闭 PeerConnection，这样可以确保所有的清理工作都顺利完成
    setTimeout(() => {
        pc.close();  // 关闭 RTCPeerConnection 实例，断开所有连接
    }, 500);
}
