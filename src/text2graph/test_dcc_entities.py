import os


def test_ner_model(nlp, test_dir, output_dir):
  # first collect all file names in the test directory
  files_list = []
  for root, dirs, files in os.walk(test_dir):
    for file1 in files:
      files_list.append(file1)


  # next go through all files in the test directory, read the text they contain,
  # and predict the entities that exist in them.
  test_files = os.scandir(test_dir)
  file_counter = 0
  for test_file in test_files:
    curr_file = os.path.join(test_file)
    with open(curr_file, "r+", encoding="utf8") as test_file:
      test_text = test_file.read()
      doc = nlp(test_text)

      file_name = str(files_list[file_counter])
      file_name = file_name.replace('.txt', '')
      ents_file = os.path.join(output_dir, file_name + "_ents" + ".txt")
      with open(ents_file, "w+", encoding="utf8") as ef:
        for ent in doc.ents:
          ef.write("%s %s %d %d\n" % (ent.label_.encode("utf-8"), ent.text.encode("utf-8"), ent.start_char, ent.end_char))

      file_counter =+ 1
