import os


key = 'entities'
train_data = []

# structure to be created:
# TRAIN_DATA = [
#   ('sentence 1',
#   {
#    'entities': [(0, 3, 'TYPE'),
#                 (5, 7, 'TYPE')
#                 ]
#   })]




def collect_txt_ann_files(directory):
    txt_file_list = []
    ann_file_list = []
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if "txt" in fname:
                txt_path = directory + fname
                txt_file_list.append(txt_path)
            else:
                ann_path = directory + fname
                ann_file_list.append(ann_path)

    return ann_file_list, txt_file_list


def create_training_data(directory):
    ann_file_list, txt_file_list = collect_txt_ann_files(directory)
    for fi in range(len(ann_file_list)):
        #print("processing file: ", fi)
        with open(txt_file_list[fi], encoding="utf8") as txtf, open(ann_file_list[fi], encoding="utf8") as annf:
            tup = ()
            for txt_line in txtf:
                txt_line = txt_line.strip()
                tup += (txt_line,)

            ent_list = []
            for line in annf:
                # Split each line of the ann file
                line = line.strip().split()
                tag = line[0]
                if 'T' in tag:
                    ent_type = line[1]
                    if 'B' in ent_type:
                        print(ent_type)
                    start = line[2]
                    end = line[3]
                    entity_text = line[4]

                    # create the tuple
                    tup2 = ()
                    tup2 += (int(start),)
                    tup2 += (int(end),)
                    tup2 += (ent_type,)

                    # create a list
                    ent_list.append(tup2)

        # Create dict for each row.
        d = {}
        d[key] = ent_list

        # add the dictionary to tuple
        tup += (d,)

        # add the tuple to train_data
        train_data.append(tup)

    #print(train_data)

    return train_data

if __name__ == '__main__':
    directory = 'C:/Home/src/Python/ASKE/abstract-sentences-test/'
    create_training_data(directory)
