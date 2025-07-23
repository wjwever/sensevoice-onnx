<h1 align="center">SenseVoiceOnnx </h1>
<div class="column" align="middle">
  <p align="center">
  </p>
  </a>
  <a href="https://en.cppreference.com/w/">
    <img src="https://img.shields.io/badge/Language-C++-blue.svg" alt="language"/>
  </a>
  <img src="https://img.shields.io/badge/platform-Linux-9cf.svg" alt="linux"/>
  <img src="https://img.shields.io/badge/Release-v0.1.0-green.svg" alt="release"/>

<h4 align="center">If you are interested in This project, please kindly give Me a triple `Star`, `Fork` and `Watch`, Thanks!</h4>
</div>

A asr project use silero_vad to dectct voice and sense_voice to do asr.
# Build
After code cloned, run following command to compile the code。
```
sh build.sh
```
# Recognize one wav （16khz）
```
./bin/infer  test.wav
```
![image](https://github.com/user-attachments/assets/392e02bd-993a-4be5-b785-f8b32ac88cda)

# Alsa stream example
```
  ./bin/stream  plughw:0,6   # replace plughw:0,6 to your proper device
```
![选区_005](https://github.com/user-attachments/assets/2d26809f-7a3b-45db-9545-49f05d8333e0)

# Websocket server
``` bash
# ws server
./bin/server
# client
cd scripts; bash multi_run.sh 
```

