import os
import pandas as pd

def get_actor():

    actor = {}

    for i in range(1,25):
        c = '0{}'.format(i) if i < 10 else '{}'.format(i)
        actor[c] = 'female' if i % 2 == 0 else 'male'
    
    return actor

def get_data_files():
    data_files = pd.DataFrame()
    for subdir, dirs, files in os.walk('data'):
        for file in files:
            if not file == '.DS_Store':
                f_split = file.split('.')[0].split('-')
                tmp = pd.DataFrame({
                      'full_path':os.path.join(subdir, file)
                    , 'parent_path':subdir
                    , 'file_name': file
                    , 'actor_num': subdir.split('_')[1]
                    , 'modelity': modality[f_split[0]]
                    , 'vocal_channel': vocal_channel[f_split[1]]
                    , 'emotion': emotions[f_split[2]]
                    , 'emotional_intensity': emotional_intensity[f_split[3]]
                    , 'statement': statement[f_split[4]]
                    , 'repetition': repetition[f_split[5]]
                    , 'actor_gender': get_actor()[f_split[6]]
                }, index=[0])
                data_files = data_files.append(tmp, ignore_index=True, sort=False)    
    return data_files

modality = {
    '01':'full-AV',
    '02':'video-only',
    '03':'audio-only'
}

vocal_channel = {
    '01':'speech',
    '02':'song'
}

emotions = {
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

emotional_intensity = {
    '01':'normal',
    '02':'strong'
}

statement = {
    '01':'Kids are talking by the door',
    '02':'Dogs are sitting by the door'
}

repetition = {
    '01':'1st repetation',
    '02':'2nd repetation'
}
    
actor_gender = get_actor()
