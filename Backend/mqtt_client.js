// // Backend/mqtt_client.js
// require('dotenv').config()
// const mqtt  = require('mqtt')
// const axios = require('axios')

// const {
//   MQTT_BROKER,
//   MQTT_PORT,
//   MQTT_USERNAME,
//   MQTT_PASSWORD,
//   RAW_TOPIC,
//   PARSED_TOPIC,
//   PARSE_SERVICE_URL
// } = process.env

// const client = mqtt.connect(`mqtt://${MQTT_BROKER}:${MQTT_PORT}`, {
//   username: MQTT_USERNAME,
//   password: MQTT_PASSWORD,
// })

// client.on('connect', () => {
//   console.log('✅ MQTT connected')
//   client.subscribe(RAW_TOPIC, err => {
//     if (err) console.error('❌ Subscribe error', err)
//     else console.log('Subscribed to topic:', RAW_TOPIC)
//   })
// })

// client.on('message', async (topic, payload) => {
//   if (topic !== RAW_TOPIC) return

//   try {
//     // 把二进制 Buffer 转成 hex 字符串
//     const hexString = payload.toString('hex')
//     console.log('Received raw (hex):', hexString)

//     // 调用后端解析服务
//     const resp = await axios.post(PARSE_SERVICE_URL, {
//       rawText: hexString,
//       // type 字段如果需要可以加上
//     })
//     const parsed = resp.data
//     console.log('Parsed result:', parsed)

//     // 发布到解析后主题
//     client.publish(PARSED_TOPIC, JSON.stringify(parsed))
//     console.log(`📤 Published parsed data to ${PARSED_TOPIC}`)
//   } catch (e) {
//     console.error('Error processing message:', e.message)
//   }
// })

// client.on('error', err => {
//   console.error('MQTT error:', err)
//   client.end()
// })
