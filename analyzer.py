from difflib import Differ

with open('file1.txt') as file_1, open('file2.txt') as file_2:
	differ = Differ()

	for line in differ.compare(file_1.readlines(), file_2.readlines()):
		print(line)
