# learn-vslam
Learn slam by hand using python

learn from [twitchslam](https://github.com/geohot/twitchslam)

# 环境配置
(依旧是推荐使用conda和系统的环境隔离，防止损坏环境)

- opencv
- pygame
- g2o
- pangolin
- pyopengl
- scikit-image
- numpy

下面是环境配置（因为怕一些文件和之前配置的系统环境有冲突，为了防止损坏环境，使用docker）：

- 使用docker配环境比较好，基础环境使用`opencvcourses/opencv-docker:4.4.0`, 运行如下命令：
`xhost +local:root`

`docker run --name learn-vslam --runtime=nvidia -ti -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/learn-pyhton/slam-visual:/slam-visual opencvcourses/opencv-docker:4.4.0`
~/Downloads:/dataVol
- 文件比较大，耐心等待下载

到这里可以进行刚g2o和panlion的python版本编译
具体参考`https://github.com/uoip/g2opy`、`https://github.com/uoip/pangolin`

twitchslam原作者在这里也折腾了很长时间，要有耐心！

如果只安装g2o可以在上述docker中，但在安装pangolin时候会产生冲突（opengl相关的问题，pangolin安装好，opencv会有问题，或者无法进行图形化界面操作），最后都成功是在电脑实体上（！建议在干净的实体机上，更容易些，使用apt安装包有可能在解决问题时损坏文件依赖关系，请提前做好备份），折腾的时候经过测试python3.6或者3.7都可以成功，一般编译不通过的看issue基本都可以解决。

导入成功后需要测试uiop作者的示例，都通过才可以，我这里没遇到issue中的导入成功后某些类导入失败。

