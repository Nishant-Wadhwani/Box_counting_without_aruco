import pika
import time
import master_new
import json
import json_compile
import subprocess
import os
import server_to_dataset_rsync
import output_to_server_rsync
# from scheduler.send import AOSToDrone
# amqp://guest:guest@localhost:5672/%2F
params = pika.ConnectionParameters(
    heartbeat=600, blocked_connection_timeout=300, host="localhost"
)
try:
    connection = pika.BlockingConnection(params)
except Exception as e:
    print(e)
    time.sleep(20)
    print("waiting to reconnect")
    connection = pika.BlockingConnection(params)
# connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.exchange_declare(exchange='test2', exchange_type='topic')
# result = channel.queue_declare(queue='', exclusive=True)
# queue_name = result.method.queue
channel.queue_declare(queue='aos.vision', durable=True)
channel.queue_bind(exchange='test2', queue='aos.vision',routing_key='aos.vision')
print(' [*] Waiting for messages. To exit press CTRL+C')


# This file_path should be extracted from body part of callback function
# If it doesn't stop how will we force quit the script

def callback(ch, method, properties, body):
    print(" [x] {}".format(body))
    payload = body.decode()
    message1 = json.loads(payload)

    #print("File_path of the data: ",file_path)
    
    # print(type(body))
    # L = list(body)
    # print(L)
    #print("File_path: ",body["data_tranfer_status"]["task_status"])
    #Pass this data to master_new.py to integrate the whole json_data
    file_path = server_to_dataset_rsync.data_download(message1["task_id"])
    
    #Run master_new.py here
    file_path = file_path + "Input/*"
    print("File_path: ", file_path)
    master_new.master_script(file_path,message1["task_id"])
    # Json Compilation
    L = json_compile.func("../output/")
    print("Json data: ",L)
    dictionary = dict()
    dictionary["results"] = L
    # dictionary_append = dict()
    # dictionary_append.update(message1)
    dictionary.update(message1)
    print("Final Dictionary: ",dictionary)
    del dictionary["data_tranfer_status"]
    del dictionary["message"]
    dictionary["task_status"] = "Completed"
    out_file_path = message1["processed_image_path"]
    out_file_path = out_file_path.split(":")[1]
    output_to_server_rsync.data_upload(out_file_path)
    with open("../output/sample.json", "w") as outfile:
        outfile.seek(0)
        json.dump(dictionary, outfile)
        outfile.truncate()
    subprocess.call("python3 vision_report_sender.py", shell=True)

    # json_final
    # Trigger vision_report_sender.py
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_consume(
    queue='aos.vision', on_message_callback=callback,)
channel.start_consuming()
