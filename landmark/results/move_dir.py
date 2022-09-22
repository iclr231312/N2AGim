import os
import shutil
if __name__ == "__main__":
    dirs = os.listdir("./")
    # print(dirlist)
    for dir in dirs:
        if "py" in dir:
            # print(dir)
            continue
        subdir = os.listdir(dir)
        subdir.sort()
        
        files = os.listdir(dir+"/"+subdir[-1])
        # print()
        for file in files:
            if "log" not in file:
                if os.path.exists(dir+"/"+subdir[-1]+"/"+file):
                    # print(dir+"/")
                    shutil.copy(dir+"/"+subdir[-1]+"/"+file,dir+"/")