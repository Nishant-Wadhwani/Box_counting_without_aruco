import sysrsync
def data_upload(file_path):
	src = "/home/nishant/Wipro/work/output/processed_images"
	dest = file_path
	sysrsync.run(source=src,
		destination=dest,
		destination_ssh='nishant@192.168.1.12',
		private_key="/home/nishant/.ssh/id_rsa",
		options=['-avzhP'],
		sync_source_contents=False)