import pika
import time
import json
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
channel.exchange_declare(exchange='test3', exchange_type='topic')
# result = channel.queue_declare(queue='', exclusive=True)
# queue_name = result.method.queue
channel.queue_declare(queue='vision.aos', durable=True)
channel.queue_bind(exchange='test3', queue='vision.aos',routing_key='vision.aos')
print(' [*] Waiting for messages. To exit press CTRL+C')


# This file_path should be extracted from body part of callback function
# If it doesn't stop how will we force quit the script

def callback(ch, method, properties, body):
    print(" [x] {}".format(body))
    
    # print(type(body))
    # L = list(body)
    # print(L)
    #print("File_path: ",body["data_tranfer_status"]["task_status"])
    #Pass this data to master_new.py to integrate the whole json_data
    #Run master_new.py here
    
    
    # json_final
    # Trigger vision_report_sender.py
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_consume(
    queue='vision.aos', on_message_callback=callback,)
channel.start_consuming()
