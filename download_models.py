from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('iic/SenseVoiceSmall', cache_dir='/models')
snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', cache_dir='/models')
print('Models downloaded successfully')
