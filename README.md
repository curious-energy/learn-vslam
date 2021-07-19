# learn-vslam
Learn slam by hand using python

learn from [twitchslam](https://github.com/geohot/twitchslam)

下面是环境配置（因为一些文件和之前配置的系统环境有冲突，为了防止损坏环境，使用docker）：

- 使用docker配环境比较好，基础环境使用`opencvcourses/opencv-docker:4.4.0`, 运行如下命令：
`xhost +local:root`

`docker run --name learn-vslam --runtime=nvidia -ti -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/learn-pyhton/slam-visual:/slam-visual opencvcourses/opencv-docker:4.4.0`

- 文件比较大，耐心等待下载

到这里可以进行刚g2o和panlion的python版本编译
具体参考`https://github.com/uoip/g2opy`、`https://github.com/uoip/pangolin`

作者在这里也折腾了很长时间，要有耐心！