# Look, Listen and Learn - A Multimodal LSTM for Speaker Identification

This code is to demonstrate the multimodal LSTM described in the following paper <br>

Jimmy SJ. Ren, Yongtao Hu, Yu-Wing Tai, Chuan Wang, Li Xu, Wenxiu Sun, Qiong Yan, 
"[Look, Listen and Learn - A Multimodal LSTM for Speaker Identification](http://www.jimmyren.com/papers/AAAI16_Ren.pdf)", The 30th AAAI Conference on Artificial Intelligence (AAAI-16). <br>

Please visit [here](https://github.com/jimmy-ren/vLSTM) for a refactored version of the multimodal LSTM and more applications. The training procedure and the pre-processed training data used in this paper are also released there. <br>

## Dataset
Raw dataset can be downloaded [Baidu Pan](http://pan.baidu.com/s/1hrdNhiO) or [Google Drive](https://drive.google.com/folderview?id=0B6nl_KFEGWG0UUJjaWRGQ19PZnc&usp=sharing).

###Dataset summary
This is a multimodal dataset containing both face images and corresponding speaking audio clips, which is extracted from the first two seasons of TV series - "The Big Bang Theory". 

###Face images
We extracted the faces of all the characters (no matter he/she is a leading character or not) from 12 episodes in the TV series, i.e. first 6 episodes from Season 1 and first 6 episodes from Season 2. All the faces have been organized per character per episode. For example, for Season-1-Episode-1, you will find 6 folders in folder `face-images/s01e01`, including 5 of them are for the 5 leading characters (Howard, Leonard, Penny, Raj and Sheldon) and 1 named `other` for all other non-leading characters. In total, we have more than 407K face images (in JPG format).

###Speaking audio
We extracted all the speaking audio segments of the 5 leading characters across the whole two seasons. All speaking audio clips are merged into one per character per episode. For example, for Season-2-Episode-1, you will find 5 WAV files in folder `speaking-audio/s02e01` for the 5 leading characters (Howard, Leonard, Penny, Raj and Sheldon). The matches between the name and labels used in the WAV filenames can be seen in the `labels.txt` file. In total, we have more than 3 hours length of speaking audio (in WAV format). 

###Terms of use
The dataset is provided for <b>research purposes only</b>. Any commercial use is prohibited. Please cite our paper if you use the dataset in your research work

## How to run the code
This code uses pre-processed data in the .mat form. To run the code, please go [here](http://pan.baidu.com/s/1gex1U5H) or [here](https://drive.google.com/folderview?id=0B6nl_KFEGWG0OGpaejB0Q05kdUE&usp=sharing) to download the data. <br>

The code was tested in Ubuntu 14.04, it should also run in Windows. You have to have a NVidia GPU to run the code, graphics memory need to be larger than 4GB.

### Run multimodal LSTM (the full weight sharing mode)
Step 1: Go [here](http://pan.baidu.com/s/1gex1U5H) or [here](https://drive.google.com/folderview?id=0B6nl_KFEGWG0OGpaejB0Q05kdUE&usp=sharing), download the whole `LSTM_sn_full_mm_weight_share` folder overwrite the same folder in the code. <br>
Step 2: Launch Matlab and enter the `LSTM_sn_full_mm_weight_share` folder. Open `speaker_naming/face_audio_5c/`, run `test_FA_all_v52.m`. <br>

Wait for several minutes and you will see the caculated false alarm rate and accuray. You will find that both false alarm rate and accuracy are <b>the highest</b> among all versions of multi/single modal LSTM.

### Run multimodal LSTM (the half weight sharing mode)
Step 1: Go [here](http://pan.baidu.com/s/1gex1U5H) or [here](https://drive.google.com/folderview?id=0B6nl_KFEGWG0OGpaejB0Q05kdUE&usp=sharing), download the whole `LSTM_sn_half_mm_weight_share` folder overwrite the same folder in the code. <br>
Step 2: Launch Matlab and enter the `LSTM_sn_half_mm_weight_share` folder. Open `speaker_naming/face_audio_5c/`, run `test_FA_all_v5.m`.

### Run single modal LSTM
Step 1: Go [here](http://pan.baidu.com/s/1gex1U5H) or [here](https://drive.google.com/folderview?id=0B6nl_KFEGWG0OGpaejB0Q05kdUE&usp=sharing), download the whole `LSTM_sn_no_mm_weight_share` folder overwrite the same folder in the code. <br>
Step 2: Launch Matlab and enter the `LSTM_sn_no_mm_weight_share` folder. Open `speaker_naming/face_audio_5c/`, run `test_FA_all_v61.m`.

### Run single modal LSTM for image alone classification and audio alone classification
Step 1: Go [here](http://pan.baidu.com/s/1gex1U5H) or [here](https://drive.google.com/folderview?id=0B6nl_KFEGWG0OGpaejB0Q05kdUE&usp=sharing), download the whole `LSTM_sn_audio_only` folder as well as `LSTM_sn_face_only` folder, overwrite the same folders in the code. <br>
Step 2: Launch Matlab and enter the `LSTM_sn_audio_only` folder or `LSTM_sn_face_only` folder. Open `speaker_naming/audio_only/` or `speaker_naming/face_only/`, run `test_audio_all.m` or `test_face_all.m`.



