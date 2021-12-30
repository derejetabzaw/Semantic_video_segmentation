import os 
def keyframe_selector(target_video):
    target_video = f'"{target_video}"'
    output_directory = "'" + (os.getcwd() + "/Dataset/keyframes").replace("\\","/")  + "'"
    output_directory = (output_directory).replace("'", '"')

    scenedetect_script = "scenedetect -i " + str(target_video) + " -o " + str(output_directory) + " detect-content" + " save-images" + " split-video"
    print(scenedetect_script)
    os.system(scenedetect_script)
    for root, dirs, files in os.walk('Dataset/keyframes'):
        for filename in files:
            if filename.endswith((".mp4",".flv",".mpg",".avi",".wmv",".mpv")):
                input_file = (os.path.join(os.getcwd() + "/Dataset/keyframes/",filename)).replace("\\","/")
                delete_videos = input_file.replace("/","\\")
                input_file = f'"{input_file}"'
                audio_script = "ffmpeg -i " +  str(input_file) + " -vn -acodec pcm_s16le -ar 16000 -ac 2 "  +str(input_file) + ".wav"
                os.system(audio_script)
                os.remove(delete_videos) 

