import json
import os
#input_dir = '/home/nishant/Wipro/work/output/'

def func(input_dir):
	listOfFiles = []
	L_json_data = []
	json_file_list = []

	for (dirpath, dirnames, filenames) in os.walk(input_dir):
		listOfFiles += [os.path.join(dirpath, file) for file in filenames]
	#print("listOfFiles: ",listOfFiles)
	for file_path in listOfFiles:
		if '.json' in file_path:
			f = open(file_path)
			json_file_list.append(file_path)
			# returns JSON object as
			# a dictionary
			data = json.load(f)
			L_json_data.append(data)
			# Iterating through the json
			# list
			# Closing file
			f.close()
	print("json_file_list : ",json_file_list)
	print("length of json_file_list : ",len(json_file_list))
	print()
	print()
	print()
	print()
	print()
	#for json_data in L_json_data:
		#print(" json_data: ",json_data)
		#print()
		#print()
		#print()
	return L_json_data

"""
L = []
import glob, os
os.chdir(input_dir)
def func():
	for file in glob.glob("*.json"):
		f = open(file)

		# returns JSON object as
		# a dictionary
		data = json.load(f)
		L.append(data)
		# Iterating through the json
		# list
		# Closing file
		f.close()    
	print(L)
	return L
"""
