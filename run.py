import os

base_dir = './pred_result'
cg_dir = 'utils/input/detection-results'

os.mkdir(cg_dir)
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

for file_name in os.listdir(base_dir):
    class_name = file_name.split(".txt", 1)[0]
    lines_list = file_lines_to_list(base_dir+'/'+file_name)
    for line in lines_list:
        img_name, confidence, left, top, right, bottom = line.split()
        file = open('%s/%s.txt' %(cg_dir, img_name), 'a', encoding='utf-8')
        file.write(class_name)
        file.write('\t')
        file.write( confidence)
        file.write('\t')
        file.write(left)
        file.write('\t')
        file.write(top)
        file.write('\t')
        file.write(right)
        file.write('\t')
        file.write(bottom)
        file.write('\n')
        file.close()
