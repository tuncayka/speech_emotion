> ### ``` It’s Not What You Say, But How You Say It```

### Emotion classification using the RAVDESS dataset

You can find detail description about dataset above. Description was provided by Zenodo. Dataset can be downloaded from [here &#8618; .](https://zenodo.org/record/1188976#.Xpaa3i-caAP)

I used audio only dataset and created a model for prediction of emotion audio.

According to dataset, there are 2 sentences for audio, and they are "Kids are talkin by the door" and "Dogs are sitting by the door".
My main purpose on this project is analyze audio files and find the emotion from them. 


##### Description from original page:


The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 7356 files (total size: 24.8 GB). The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. All conditions are available in three modality formats: Audio-only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).  Note, there are no song files for Actor_18.

#### File naming convention

Each of the 7356 RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics: 

**Filename identifiers**

* Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
* Vocal channel (01 = speech, 02 = song).
* Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
* Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
* Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
* Repetition (01 = 1st repetition, 02 = 2nd repetition).
* Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

Filename example: 02-01-06-01-02-01-12.mp4 

- Video-only (02)
- Speech (01)
- Fearful (06)
- Normal intensity (01)
- Statement "dogs" (02)
- 1st Repetition (01)
- 12th Actor (12)
- Female, as the actor ID number is even.


#### What is Audio & Speech Processing

* Audio signal processing is a subfield of signal processing that is concerned with the electronic manipulation of audio signals. Audio signals are electronic representations of sound waves—longitudinal waves which travel through air, consisting of compressions and rarefactions. The energy contained in audio signals is typically measured in decibels. As audio signals may be represented in either digital or analog format, processing may occur in either domain. Analog processors operate directly on the electrical signal, while digital processors operate mathematically on its digital representation. [[1]](https://en.wikipedia.org/wiki/Audio_signal_processing)

* Speech processing is the study of speech signals and the processing methods of signals. The signals are usually processed in a digital representation, so speech processing can be regarded as a special case of digital signal processing, applied to speech signals. Aspects of speech processing includes the acquisition, manipulation, storage, transfer and output of speech signals. The input is called speech recognition and the output is called speech synthesis. [[2]](https://en.wikipedia.org/wiki/Speech_processing)

#### Human Voice

* The human voice consists of sound made by a human being using the vocal tract, such as talking, singing, laughing, crying, screaming, shouting, yelling etc. The human voice frequency is specifically a part of human sound production in which the vocal folds (vocal cords) are the primary sound source. (Other sound production mechanisms produced from the same general area of the body involve the production of unvoiced consonants, clicks, whistling and whispering.)[[3]](https://en.wikipedia.org/wiki/Human_voice)

* A labeled anatomical diagram of the vocal folds or cords.
<img src="https://upload.wikimedia.org/wikipedia/commons/b/bd/Gray1204.png" width="400" height="400" />

* Men and womens have different sizes of vocal folds. These difference causes diferent pitch or loud levels. 
* All voices have different frequencies. A voice frequency (VF) or voice band is one of the frequencies, within part of the audio range, that is being used for the transmission of speech. [[4]](https://en.wikipedia.org/wiki/Voice_frequency) 
* The ability to modulate vocal sounds and generate speech is one of the features which set humans apart from other living beings. The human voice can be characterized by several attributes such as pitch, timbre, loudness, and vocal tone. It has often been observed that humans express their emotions by varying different vocal attributes during speech generation. Hence, deduction of human emotions through voice and speech analysis has a practical plausibility and could potentially be beneficial for improving human conversational and persuasion skills.[[5]](https://arxiv.org/pdf/1710.10198.pdf)

* Below there are 2 tables to show two different emotional state statistics. First one is normal emotional state and second one is angry emotional state [[5]](https://arxiv.org/pdf/1710.10198.pdf). The most apparent information in tables is when you are angry state voice is louder than normal state. And second apparent one is when you are angry time gaps between words is smaller than normal state. 

<b><center>Normal Emotional State</center></b>

|                 | Pitch (Hz) | SPL(dB)     | Timbre ascend time (s) | Timbre descend time (s) | Time gaps between words (s) |
|-----------------|------------|-------------|------------------------|-------------------------|-----------------------------|
| Speech Sample 1 | 1248 Hz    | Gain -50 dB | 0.12 s                 | 0.11 s                  | 0.12 s                      |
| Speech Sample 2 | 1355 Hz    | Gain -48 dB | 0.06 s                 | 0.05 s                  | 0.12 s                      |

<b><center>Angry Emotional State</center></b>

|                 | Pitch (Hz) | SPL(dB)     | Timbre ascend time (s) | Timbre descend time (s) | Time gaps between words (s) |
|-----------------|------------|-------------|------------------------|-------------------------|-----------------------------|
| Speech Sample 1 | 1541 Hz    | Gain -30 dB | 0.13 s                 | 0.10 s                  | 0.09 s                      |
| Speech Sample 2 | 1652 Hz    | Gain -29 dB | 0.06 s                 | 0.04 s                  | 0.10 s                      |

**Emotional state of a human is able to understandable for all humans. Because we know how to process coming voice. In this notebook I will try the understand emotional state of humans.**