#!/usr/bin/env python3
# Backend/mqtt_forwarder.py

import os
import json
import binascii
import requests
import paho.mqtt.client as mqtt
from urllib.parse import urlencode
from dotenv import load_dotenv

# 加载 .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# 从环境变量读取配置
MQTT_BROKER       = os.getenv('MQTT_BROKER')
MQTT_PORT         = int(os.getenv('MQTT_PORT', '1884'))
MQTT_USERNAME     = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD     = os.getenv('MQTT_PASSWORD')
RAW_TOPIC         = os.getenv('RAW_TOPIC', 'topictest')
PARSED_TOPIC      = os.getenv('PARSED_TOPIC', 'parsedTopic')
PARSING_SERVICE_URL = os.getenv('PARSING_SERVICE_URL', 'http://127.0.0.1:5000/main')
DEVICE_ID         = os.getenv('DEVICE_ID', '1839560736610762752')

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ MQTT connected")
        client.subscribe(RAW_TOPIC)
        print(f"Subscribed to raw topic: {RAW_TOPIC}")
    else:
        print(f"❌ MQTT connect failed, code {rc}")

def on_message(client, userdata, msg):
    try:
        # 1. 转 hex
        raw_bytes = msg.payload
        hex_data = binascii.hexlify(raw_bytes).decode('ascii')
        print(f"📩 Received on {msg.topic}: {hex_data[:60]}...")

        # 2. 构造带 device_id 和 hex 的请求
        params = {
            "device_id": DEVICE_ID,
            "hex": hex_data
        }
        url = f"{PARSING_SERVICE_URL}?{urlencode(params)}"
        print(f"▶ Calling parser: {url}")

        # 3. 调用解析服务
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        parsed = resp.json()
        print("✅ Parser returned:", parsed)

        # 4. 发布解析结果
        client.publish(PARSED_TOPIC, json.dumps(parsed))
        print(f"📤 Published parsed JSON to {PARSED_TOPIC}")

    except Exception as e:
        print("❌ Error in on_message:", e)

def main():
    client = mqtt.Client()
    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_forever()

if __name__ == "__main__":
    main()
