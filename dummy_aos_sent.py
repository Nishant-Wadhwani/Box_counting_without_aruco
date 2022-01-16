import pika
import json
#First open your box_count's json file, read it, load it and then dump it
# f = open ('./nav_command.json', "r")
 
# Reading from file
# msg = json.loads(f.read())

msg_to_vision = {
                    "task_id": 25,
                    "processed_image_path":"192.168.1.12:/home/nishant/Wipro/work/file_server/task_1/Output/",
                    "zone_name": "Technovation",
                    "actual_duration": str(1.00),
                    "data_tranfer_status": {
                                            "status": "Completed",
                                            "filepath":"192.168.1.12:/home/nishant/Wipro/work/file_server/task_1/Input/*"
                                            },
                    "message": "The message has been sent"
                    }



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

print(json.dumps(msg_to_vision))
params = pika.ConnectionParameters(
    heartbeat=600, blocked_connection_timeout=300, host="localhost"
)
connection = pika.BlockingConnection(params)
channel = connection.channel()
channel.exchange_declare(exchange="test2", exchange_type="topic")
channel.confirm_delivery()
channel.basic_publish(
    exchange="test2", routing_key="aos.vision", body=json.dumps(msg_to_vision)
)
channel.close()
