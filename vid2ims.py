import cv2
import os,sys
import argparse

video_extensions = [".mp4", ".avi", ".flv", ".gif", ".mov", ".viv", ".svi"]


def load_videos(video_directory):
    loaded_videos = set()
    if os.path.isdir(video_directory):
        files = os.scandir(video_directory)
        for f in files:
            _, ext = os.path.splitext(f)
            if ext in video_extensions:
                loaded_videos.add((cv2.VideoCapture(f.path),f.path))
    elif os.path.isfile(video_directory):
        loaded_videos.add((cv2.VideoCapture(video_directory),video_directory))
    else:
        raise RuntimeError("Specified path does not exist or is not a file or directory, please specify a valid filepath")

    if len(loaded_videos) == 0:
        raise RuntimeError("Unable to find or load any video files, please specify a video file or directory containing video files")
    return loaded_videos

def decompose_video(video, output_dir, video_filename):
    cont = True
    name, ext = os.path.splitext(video_filename)
    name = name.split(os.sep)[-1]
    out_name = os.path.join(output_dir,name+"frame%i"+".png")
    count = 0
    while cont:
        cont, frame = video.read()
        if cont:
            cv2.imwrite(out_name%count,frame)
            count+=1

def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--video-directory",
        "-v",
        action="store",
        dest="video_file",
        required=True,
    )
    args.add_argument(
        "--output-directory",
        "-o",
        dest="output_dir",
        required=False,
        default=os.getcwd(),
        action="store",
    )
    opts = args.parse_args(sys.argv[1:])
    print("Loading videos")
    videos = load_videos(opts.video_file)
    print("%i videos loaded" %len(videos))
    print("Writing %i videos to image" % len(videos))
    for video in videos:
        decompose_video(video[0], opts.output_dir, video[1])
    print("Done decomposing videos")

if __name__ == '__main__':
    main()