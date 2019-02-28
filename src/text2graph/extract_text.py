from PyPDF2 import PdfFileWriter, PdfFileReader

output = PdfFileWriter()
input1 = PdfFileReader(open("C:\\Users\\z003z47y\\Documents\\git\\dcc\\src\\paperswithcode\\data\\0\\1810.13409v1.pdf", "rb"))

# print how many pages input1 has:
print("document1.pdf has %d pages." % input1.getNumPages())

# add page 1 from input1 to output document, unchanged
print(input1.getPage(0))
