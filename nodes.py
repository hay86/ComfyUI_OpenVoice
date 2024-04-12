import os
import torch
import random
import folder_paths
import soundfile as sf

from .openvoice import se_extractor
from .openvoice.api import BaseSpeakerTTS, ToneColorConverter


class AnyType(str):
    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class OpenVoiceTTS:
    @classmethod
    def INPUT_TYPES(s):
        audio_extensions = ["wav", "mp3", "flac"]
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.lower().split('.')
                if len(file_parts) > 1 and (file_parts[-1] in audio_extensions):
                    files.append(f)
        return {
            "required": {
                    "text": ("STRING", {"default": '', "multiline": True}),
                    "lang": (["English","Chinese"],),
                    "style": (["default","whispering","cheerful","terrified","angry","sad","friendly"],),
                    "speed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                    "ref_voice": (sorted(files),),
            },
        }

    CATEGORY = "OpenVoice"

    RETURN_TYPES = (any, "INT",)
    RETURN_NAMES = ("AUDIO", "SAMPLE_RATE",)
    FUNCTION = "inference"

    def inference(self, text, lang, style, speed, ref_voice):
        local_dir = os.path.join(folder_paths.models_dir, 'openovice')
        if not os.path.exists(local_dir) or not os.path.isdir(local_dir):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Alignment-Lab-AI/OpenVoice", local_dir=local_dir, local_dir_use_symlinks=False)
        
        mark = BaseSpeakerTTS.language_marks.get(lang.lower(), None)
        assert mark is not None, f"language {lang} is not supported"

        ckpt_base = os.path.join(local_dir, f'checkpoints/base_speakers/{mark}')
        ckpt_converter = os.path.join(local_dir, 'checkpoints/converter')
        device="cuda:0" if torch.cuda.is_available() else "cpu"
        
        base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
        base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        style_name = 'default' if style == 'default' else 'style'
        source_se = torch.load(f'{ckpt_base}/{mark.lower()}_{style_name}_se.pth').to(device)
        reference_speaker = os.path.join(folder_paths.get_input_directory(), ref_voice)
        
        temp_dir = folder_paths.get_temp_directory()
        file_prefix = ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir=temp_dir, vad=True)

        save_path = f'{temp_dir}/{file_prefix}_output_{mark.lower()}_{style}.wav'

        # Run the base speaker tts
        src_path = f'{temp_dir}/{file_prefix}_base_{mark.lower()}_{style}.wav'
        base_speaker_tts.tts(text, src_path, speaker=style, language=lang, speed=speed)

        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=save_path,
            message=encode_message)

        audio_samples, sample_rate =sf.read(save_path)
        return (list(audio_samples), sample_rate)

class OpenVoiceSTS:
    @classmethod
    def INPUT_TYPES(s):
        audio_extensions = ["wav", "mp3", "flac"]
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.lower().split('.')
                if len(file_parts) > 1 and (file_parts[-1] in audio_extensions):
                    files.append(f)
        return {
            "required": {
                    "src_voice": (sorted(files),),
                    "ref_voice": (sorted(files),),
            },
        }

    CATEGORY = "OpenVoice"

    RETURN_TYPES = (any, "INT",)
    RETURN_NAMES = ("AUDIO", "SAMPLE_RATE",)
    FUNCTION = "inference"

    def inference(self, src_voice, ref_voice):
        local_dir = os.path.join(folder_paths.models_dir, 'openovice')
        if not os.path.exists(local_dir) or not os.path.isdir(local_dir):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Alignment-Lab-AI/OpenVoice", local_dir=local_dir, local_dir_use_symlinks=False)

        ckpt_converter = os.path.join(local_dir, 'checkpoints/converter')
        device="cuda:0" if torch.cuda.is_available() else "cpu"

        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        temp_dir = folder_paths.get_temp_directory()
        file_prefix = ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))

        source_speaker = os.path.join(folder_paths.get_input_directory(), src_voice)
        source_se, audio_name = se_extractor.get_se(source_speaker, tone_color_converter, target_dir=temp_dir, vad=True)

        reference_speaker = os.path.join(folder_paths.get_input_directory(), ref_voice)
        target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir=temp_dir, vad=True)

        save_path = f'{temp_dir}/{file_prefix}_output_crosslingual.wav'

        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=source_speaker,
            src_se=source_se,
            tgt_se=target_se,
            output_path=save_path,
            message=encode_message)

        audio_samples, sample_rate =sf.read(save_path)
        return (list(audio_samples), sample_rate)
    

NODE_CLASS_MAPPINGS = {
    "D_OpenVoice_TTS": OpenVoiceTTS,
    "D_OpenVoice_STS": OpenVoiceSTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "D_OpenVoice_TTS": "Open Voice TTS",
    "D_OpenVoice_STS": "Open Voice STS",
}