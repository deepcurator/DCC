import os

def test_ner_model(nlp, test_dir):
  # directory listing for the test files
  test_files = os.scandir(test_dir)
  for test_file in test_files:
    curr_file = os.path.join(test_file)
    with open(curr_file, "r+", encoding="utf8") as test_file:
      # test_file = open(curr_file, "r+", encoding="utf8")
      test_text = test_file.read()
      doc = nlp(test_text)

      print("\nEntities detected in the text: '%s'" % test_text)
      for ent in doc.ents:
        print(ent.label_, ent.text, ent.start_char, ent.end_char)

      # define the name of the file that contains the entities and their spans
      file_name = curr_file.replace('.txt', '')
      ents_file = os.path.join(file_name + "_ents" + ".txt")
      with open(ents_file, "w+", encoding="utf8") as ef:
        for ent in doc.ents:
          ef.write("%s %s %d %d\n" % (ent.label_.encode("utf-8"), ent.text.encode("utf-8"), ent.start_char, ent.end_char))
