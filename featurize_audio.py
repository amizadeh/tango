import tools.torch_tools as torch_tools
from models import build_pretrained_models

if __name__ == '__main__':

    wav_path = ''

    pretrained_model_name = "audioldm-s-full"
    vae, stft = build_pretrained_models(pretrained_model_name)
    vae.eval()
    stft.eval()

    ## Encoding
    mel, _, waveform = torch_tools.wav_to_fbank([wav_path], target_length, stft)
