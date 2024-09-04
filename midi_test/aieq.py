from pydub import AudioSegment, scipy_effects
from scipy.signal import butter, lfilter
import torch
from models import *
"""
0:'dianyin', 1:'gudian', 2:'minyao', 3:'pop', 4:'qing', 5:'rock', 6:'xiha'
"""


def my_low_pass_filter(data):
    order = 4
    high = 20000/(44100*2)
    b, a = butter(order, high, btype='low_pass')
    data_filtered = lfilter(b, a, data)
    return data_filtered


def music_add_eq(sound, music_type):

    # sound = sound
    if music_type == '0':
        aaa = scipy_effects.eq(sound, 206, 206, filter_mode="low_shelf", gain_dB=5, order=2)   # """lowshelf，206Hz +5dB Q:1"""
        aaa = scipy_effects.eq(aaa, 560, 280, filter_mode="peak", gain_dB=5.8, order=2)        # """peak/notch：560Hz +5.8dB Q:2"""
        """aaa = my_low_pass_filter(aaa)
            
        """

    elif music_type == '1':
        aaa = scipy_effects.eq(sound, 106, 106, filter_mode="low_shelf", gain_dB=2.7, order=2)  #  """lowshelf，106Hz +2.7dB Q:1"""
        aaa = scipy_effects.eq(aaa, 154, 154/1.2, filter_mode="peak", gain_dB=5.6, order=2)      # """peak/notch：154Hz +5.6dB Q:1.2"""
        aaa = scipy_effects.eq(aaa, 7450, 7450, filter_mode="peak", gain_dB=2.3, order=2)        # """peak/notch：7450Hz +2.3 Q 1"""
        """aaa = my_low_pass_filter(aaa)"""

    elif music_type == '2':
        aaa = scipy_effects.eq(sound, 45, 45, filter_mode="low_shelf", gain_dB=-3.8, order=2)     #  """lowshelf，45Hz -3.8dB Q:1"""
        aaa = scipy_effects.eq(aaa, 486, 486/1.9, filter_mode="peak", gain_dB=3.7, order=2)       #  """peak/notch：486Hz +3.7dB Q:1.9"""
        aaa = scipy_effects.eq(aaa, 945, 945/1.4, filter_mode="peak", gain_dB=2.3, order=2)       #  """peak/notch：945Hz +2.3 Q 1.4"""
        aaa = scipy_effects.eq(aaa, 2200, 2200/1.5, filter_mode="peak", gain_dB=1.1, order=2)     #  ""peak/notch：2200Hz +1.1 Q 1.5"""
        aaa = scipy_effects.eq(aaa, 11900, 11900, filter_mode="high_shelf", gain_dB=-2.3, order=2)#  """highshelf：11900Hz -2.3dB Q：1"""

    elif music_type == '3':
        aaa = scipy_effects.eq(sound, 41, 41, filter_mode="low_shelf", gain_dB=-2.6, order=2)     #  """lowshelf，41Hz -2.6dB Q:1"""
        aaa = scipy_effects.eq(aaa, 610, 610/1.2, filter_mode="peak", gain_dB=6.2, order=2)       #  """peak/notch：610Hz +6.2dB Q:1.2"""
        aaa = scipy_effects.eq(aaa, 1050, 500, filter_mode="peak", gain_dB=2.4, order=2)          #  """peak/notch：1050Hz +2.4 Q 2.1"""
        aaa = scipy_effects.eq(aaa, 8600, 8600, filter_mode="high_shelf", gain_dB=-0.9, order=2)  #  """highshelf：8600Hz -0.9dB Q：1"""
        """aaa = my_low_pass_filter(aaa)"""


    elif music_type == '4':
        aaa = scipy_effects.eq(sound, 98, 98, filter_mode="low_shelf", gain_dB=2.5, order=2)      # """lowshelf，98Hz +2.5dB Q:1"""
        aaa = scipy_effects.eq(aaa, 192, 128, filter_mode="peak", gain_dB=2.6, order=2)           # """peak/notch：192Hz +2.6dB Q:1.5"""
        aaa = scipy_effects.eq(aaa, 515, 515/1.7, filter_mode="peak", gain_dB=1.9, order=2)       # """peak/notch：515Hz +1.9 Q 1.7"""
        aaa = scipy_effects.eq(aaa, 1010, 1010/1.2, filter_mode="peak", gain_dB=1.1, order=2)     # """peak/notch：1010Hz +1.1 Q 1.2"""
        aaa = scipy_effects.eq(aaa, 1980, 1980/0.93, filter_mode="peak", gain_dB=0.4, order=2)    # """peak/notch：1980Hz +0.4 Q 0.93"""
        aaa = scipy_effects.eq(aaa, 5550, 5550, filter_mode="high_shelf", gain_dB=-0.4, order=2)  #  """5550Hz -0.4dB Q：1"""

    elif music_type == '5':
        aaa = scipy_effects.eq(sound, 88, 88, filter_mode="low_shelf", gain_dB=5.8, order=2)           #    """lowshelf，88Hz +5.8dB Q:1"""
        aaa = scipy_effects.eq(aaa, 106, 106/2.1, filter_mode="peak", gain_dB=7, order=2)              #   """peak/notch：106Hz +7dB Q:2.1"""
        aaa = scipy_effects.eq(aaa, 458, 458, filter_mode="peak", gain_dB=8.2, order=2)                #   """peak/notch：458Hz +8.2 Q 1"""
        aaa = scipy_effects.eq(aaa, 750, 750/1.8, filter_mode="peak", gain_dB=-1, order=2)             # """peak/notch：750Hz -1. Q 1.8"""
        aaa = scipy_effects.eq(aaa, 1120, 1120/3.6, filter_mode="high_shelf", gain_dB=-0.6, order=2)   #   """peak/notch：1120Hz -0.6 Q 3.6"""
        """aaa = my_low_pass_filter(aaa)"""

    elif music_type == '6':
        aaa = scipy_effects.eq(sound, 88, 88, filter_mode="low_shelf", gain_dB=5.8, order=2)           #"""lowshelf，88Hz +5.8dB Q:1"""
        aaa = scipy_effects.eq(aaa, 106, 106/2.1, filter_mode="peak", gain_dB=7, order=2)              #"""peak/notch：106Hz +7dB Q:2.1"""
        aaa = scipy_effects.eq(aaa, 458, 458, filter_mode="peak", gain_dB=8.2, order=2)                #"""peak/notch：458Hz +8.2 Q 1"""
        aaa = scipy_effects.eq(aaa, 750, 750/1.8, filter_mode="peak", gain_dB=-1, order=2)             #"""peak/notch：750Hz -1. Q 1.8"""
        aaa = scipy_effects.eq(aaa, 1120, 1120/3.6, filter_mode="high_shelf", gain_dB=-0.6, order=2)   #"""peak/notch：1120Hz -0.6 Q 3.6"""
        """aaa = my_low_pass_filter(aaa)"""

    return aaa

if __name__ == "__main__":
    input_file_path = "test_dir/blues/1.wav"
    sound = AudioSegment.rom_file(input_file_path, format="wav")

    device = torch.device("cuda:6")
    MODEL_PATH = 'v1_output_aimusic/fold_0/checkpoints/checkpoint_19.tar'
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model = MS_SincResNet()
    model.load_state_dict(state_dict['state_dict'])
    model.to(device)
    model.eval()