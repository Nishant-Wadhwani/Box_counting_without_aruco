import sysrsync
import os
def data_download(task_id):
	#src = "192.168.1.12:/home/nishant/Wipro/work/file_server/task_1/Input"
	src = '/home/nishant/Wipro/work/file_server/task_1/Input/'
	path = "/home/nishant/Wipro/work/Dataset/task_id_" + str(task_id) + "/"
	if not os.path.exists(path):
		os.mkdir(path)
	dest = path

	sysrsync.run(source=src,
		destination=dest,
		destination_ssh='nishant@192.168.1.12',
		private_key="/home/nishant/.ssh/id_rsa",
		options=['-avzhP'],
		sync_source_contents=False)

	return dest

