FFmpeg 可以用一行命令实现视频和图像的编辑，特别是制作 video demo 非常方便。


## 安装

```console
$ sudo apt install ffmpeg
```


## 常用命令

**! 使用 ffmpeg 前最好退出 conda**

### 0. 转码
* 可以嵌入到任何命令中，确保输出的视频可以在 Windows 播放
```console
-c:v libx264 -pix_fmt yuv420p
```

### 1. 空间拼接视频（图片同理）
* 左右拼接
```console
$ ffmpeg -i video1.mp4 -i video2.mp4 -filter_complex hstack output.mp4
```
* 上下拼接
```console
$ ffmpeg -i video1.mp4 -i video2.mp4 -filter_complex vstack output.mp4
```
* 多于2个视频
```console
$ ffmpeg -i video1.mp4 -i video2.mp4 -i video3.mp4 -filter_complex hstack=inputs=3 output.mp4
```

### 2. 调整分辨率
```console
$ ffmpeg -i input.png -vf scale=320:240 output.png
```
* 指定宽度，高度基于保持宽高比进行自适应调整
```console
$ ffmpeg -i input.png -vf scale=320:-1 output.png
```

### 3. 单张图片重复一段时间形成视频（用于 demo 中的标题页）
```console
$ ffmpeg -loop 1 -i input.png -t 15 output.mp4  （t表示延续时长，单位：秒）
```

### 4. 时序拼接视频
* 推荐先将待拼接的视频路径都整理到一个.txt内，随后用简短的命令实现
```console
$ cat inputlist.txt
file /path/to/file1
file /path/to/file2
file /path/to/file3   
$ ffmpeg -f concat -safe 0 -i inputlist.txt -c copy output.mp4
```

### 5. 截取视频片段
* 以时/分/秒(hh:mm:ss)为单位指定片段开始和结束的时间点（**! 不要用-c copy，虽然处理速度快，但会出现截取的片段开始播放时有一段黑屏的bug**）
```console
$ ffmpeg -ss 00:00:10 -to 00:00:20 -i input.mp4 output.mp4
```

### 6. Change video speed
* Speed up
```console
$ ffmpeg -i input.mp4 -filter:v "setpts=0.5*PTS" output.mp4
```
* Slow down
```console
$ ffmpeg -i input.mp4 -filter:v "setpts=2.0*PTS" output.mp4
```
