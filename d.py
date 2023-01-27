import shutil
import os
filename_person = 's-l400.jpg'

dst = 'D:/Python/whatiwear/static/output'

shutil.copyfile("static/result/TOM/test/try-on/" + filename_person, os.path.join(dst, filename_person))
