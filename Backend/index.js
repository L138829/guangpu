// Backend/index.js
require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
app.use(bodyParser.json());

// Helper: 调用 Python 脚本
function callPython(scriptName, args, callback) {
  const py = spawn('python', [path.join(__dirname, 'services', scriptName), ...args]);
  let output = '';
  py.stdout.on('data', data => (output += data.toString()));
  py.stderr.on('data', data => console.error(`Python error: ${data}`));
  py.on('close', code => {
    try {
      callback(null, JSON.parse(output));
    } catch (e) {
      callback(e);
    }
  });
}

// 通用解析接口：body = { rawText: "...", type: "3" or "4" }
app.post('/parse', (req, res) => {
  const { rawText, type } = req.body;
  const script = type === '4' ? 'shuizhi4.py' : 'shuizhi3_replace.py';
  callPython(script, [rawText], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json(result);
  });
});

// 健康检查
app.get('/health', (req, res) => res.json({ status: 'ok' }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Backend listening on http://localhost:${PORT}`));
