import pika
import json
import os
import time
import datetime

#First open your box_count's json file, read it, load it and then dump it
# f = open ('./nav_command.json', "r")
 
# Reading from file
# msg = json.loads(f.read())
f_path = '../output/sample.json'
f = open(f_path,)

# returns JSON object as
# a dictionary
data = json.load(f)
"""
message = {
    "task_id": '240',
    "task_status":"Completed",
    "zone_name": "Technovation",
    "drone_ids": [0],
    "actual_duration": "00:45",
    "results": [
        {
            "drone_id": 1,
            "sku_code": "SKU2001",
            "sku_name": "Giffy 140ml x 32N",
            "box_count": 58,
            "aisle_id": "AI101",
            "rack_id": "RA101",
            "shelf_id": "SH101",
            "pallet_id": "PA101",
            "verification_required": 'false',
            "remarks": ""
        },
        {
            "drone_id": 1,
            "sku_code": "SKU2002",
            "sku_name": "Santoor Classic",
            "box_count": 78,
            "aisle_id": "AI101",
            "rack_id": "RA102",
            "shelf_id": "SH102",
            "pallet_id": "PA102",
            "verification_required": 'true',
            "remarks": ""
        }
    ]
}
"""

print()
print()
print()
print("\n \n \n \n \n")
print(json.dumps(data))
print("FLow has reached!!!")

params = pika.ConnectionParameters(
    heartbeat=600, blocked_connection_timeout=300, host="localhost"
)
connection = pika.BlockingConnection(params)
channel = connection.channel()
channel.exchange_declare(exchange="test3", exchange_type="topic")
channel.confirm_delivery()
channel.basic_publish(
    exchange="test3", routing_key="vision.aos", body=json.dumps(data)
)
channel.close()
f.close()
if os.path.exists(f_path):
    os.remove(f_path)
else:
    print("Can not delete the file as it doesn't exists")
tmstmp= "../output_" + str(datetime.datetime.now())
os.rename("../output",tmstmp)