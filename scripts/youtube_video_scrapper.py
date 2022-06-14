from pytube import YouTube, Playlist
import datetime
from tqdm import tqdm


playlist_link = input("Enter the link of the playlist: ")
read = input("Read All? (Y of N): ")
if read == "Y":
    read = True
else:
    read = False


video_links = Playlist(playlist_link).video_urls
with open("videos.txt", "w") as ffile:
    if read:
        ffile.write(
            f"- [X] [Youtube: {Playlist(playlist_link).title}]\
                ({Playlist(playlist_link).playlist_url})\n")
    else:
        ffile.write(
            f"- [ ] [Youtube: {Playlist(playlist_link).title}]\
                ({Playlist(playlist_link).playlist_url})\n")

    with tqdm(
        total=len(video_links),
        desc="Saving Videos",
        unit='videos'
    ) as pbar:
        for idx, link in enumerate(video_links):
            if read:
                ffile.write(
                    f"\t- [X] [{YouTube(link).title}]({link}) \
                    `{str(datetime.timedelta(seconds=YouTube(link).length))}`\n")
            else:
                ffile.write(
                    f"\t- [ ] [{YouTube(link).title}]({link}) \
                    `{str(datetime.timedelta(seconds=YouTube(link).length))}`\n")

            pbar.update()
