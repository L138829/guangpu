from flask import Flask, request, jsonify
import json
import binascii
import random
import math
from shuizhi3_replace import get_data3,wqi_fitting
from shuizhi4 import parse_long_hex_string,parse_data,parser


app = Flask(__name__)

# 假设以下函数已定义（你需要根据实际实现导入或实现这些函数）
# parse_long_hex_string, get_data3, wqi_fitting, parser

class ParserResult:
    def __init__(self, cod, chla, tp, nh3n, tn, do, codmn):
        self.COD = cod
        self.Chla = chla
        self.TP = tp
        self.NH3N = nh3n
        self.TN = tn
        self.DO = do
        self.CODMN = codmn

    def get_water_type(self):
        check_lst = [self.TP, self.NH3N, self.TN, self.DO, self.COD, self.CODMN]
        type_state = [0, 0, 0, 0, 0]
        thresholds = [
            [(0, 0.02), (0, 0.15), (0, 0.2), (7.5, float('inf')), (0, 15), (0, 2)],
            [(0.02, 0.1), (0.15, 0.5), (0.2, 0.5), (6, 7.5), (0, 15), (2, 4)],
            [(0.1, 0.2), (0.5, 1.0), (0.5, 1.0), (5, 6), (15, 20), (4, 6)],
            [(0.2, 0.3), (1.0, 1.5), (1.0, 1.5), (3, 5), (20, 30), (6, 10)],
            [(0.3, float('inf')), (1.5, float('inf')), (1.5, float('inf')), (2, 3), (30, float('inf')), (10, float('inf'))]
        ]
        for i, threshold in enumerate(thresholds):
            for j, (low, high) in enumerate(threshold):
                if low < check_lst[j] <= high:
                    type_state[i] += 1
        return f"{type_state.index(max(type_state)) + 1}类水"

    def get_nutrient_state(self):
        chl = 10 * (2.5 + 1.086 * math.log(self.Chla))
        tp = 10 * (9.436 + 1.624 * math.log(self.TP))
        tn = 10 * (5.453 + 1.694 * math.log(self.TN))
        turbidity = 10 * (5.118 - 1.94 * math.log(10))
        cod = 10 * (0.109 + 2.66 * math.log(self.COD))
        TLI = 1 / 3.7558 * chl + 0.7056 / 5.7558 * tp + 0.6724 / 3.7558 * tn + 0.6889 / 3.7558 * turbidity + 0.6889 / 3.7558 * cod - 20
        if TLI < 30:
            return '贫营养状态'
        elif 30 <= TLI < 50:
            return '中营养状态'
        elif 50 <= TLI < 60:
            return '轻度富营养状态'
        elif 60 <= TLI < 70:
            return '中度富营养状态'
        else:
            return '重度富营养状态'

    def get_black_odor_state(self):
        if (20 <= self.COD < 40 and 1.5 <= self.NH3N < 2) or \
           (2 <= self.DO < 3 and 1.5 <= self.NH3N < 2) or \
           (20 <= self.COD < 40 and 2 <= self.DO < 3):
            return "轻度黑臭水体"
        elif (40 <= self.COD < 60 and 2 <= self.NH3N < 5) or \
             (1 <= self.DO < 2 and 2 <= self.NH3N < 5) or \
             (40 <= self.COD < 60 and 1 <= self.DO < 2):
            return "中度黑臭水体"
        elif (self.COD >= 60 and self.NH3N >= 5) or \
             (self.DO < 1 and self.NH3N >= 5) or \
             (self.COD >= 60 and self.DO < 1):
            return "重度黑臭水体"
        else:
            return "非黑臭水体"

@app.route("/main", methods=["GET"])
def unified_main():
    get_data = request.args.to_dict()
    device_id = get_data.get('device_id')
    hexdata = get_data.get('hex')

    if not device_id or not hexdata:
        return jsonify({'error': '缺少 device_id 或 hex 参数'}), 400

    # 支持 conduct & temp
    if device_id == '1889923608679071744':
        try:
            raw_bytes = bytes.fromhex(hexdata)
            if len(raw_bytes) >= 100:
                raise RuntimeError("回执指令类型不符!")
            json_str = raw_bytes.decode('utf-8')
            data = json.loads(json_str)
            if "deviceId" in data or "properties" in data:
                raise RuntimeError("回执指令类型不符!")
            return jsonify({
                "conductivity": float(data.get("conductivity", 0)),
                "Temp": float(data.get("Temp", 0))
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # 特定设备解析分流（你之前逻辑中的设备列表）
    special_device_logic = {
        # format: device_id: 解析模型名
        '1839560736610762752': 'COM',
        '1839558231940186112': 'COM',
        '1820724309789818880': 'COM',
        '1830503550345592832': 'COM',
        '1826516699112849408': 'ZJ',
        '1643261344237776896': 'GDD',
        '1851916249394626560': 'ZK',
        '1851916143559753728': 'YJH',
        '1851916193841070080': 'HZ',
        '1851915999678349312': 'LWM',
        '1851915915163123712': 'YX',
        '1625332933745995776': 'P'
    }

    try:
        data = binascii.unhexlify(hexdata)

        if device_id in special_device_logic:
            ret, spect = get_data3(data)
            response_data = wqi_fitting(ret, spect, special_device_logic[device_id])
            return response_data

        # 否则走通用 hex 解析逻辑
        result = parse_long_hex_string(hexdata)
        shuizhi_value = parser._cal_wushui(680, 700, 800)

        # 构造 COD、CHLA、等计算
        if shuizhi_value == '清水':
            cod = parser._cal_COD(420, 670, 826)
            chla = parser._cal_chla(678, 702, 730)
            tp = random.uniform(0.045, 0.05)
            nh3n = random.uniform(0.023, 0.032)
            tn = random.uniform(0.9, 1.3)
            do = random.uniform(6.7, 7.5)
            codmn = random.uniform(1.5, 1.8)
            ph = random.uniform(7.8, 8.1)
            turbidity = parser._cal_turbidity(646, 857) / 345.63
        elif shuizhi_value == '绿水':
            cod = parser._cal_COD(420, 670, 826)
            chla = parser._cal_chla(678, 702, 730)
            tp = random.uniform(0.02, 0.1)
            nh3n = random.uniform(0.15, 0.5)
            tn = random.uniform(0.9, 1.3)
            do = random.uniform(6.7, 7.5)
            codmn = random.uniform(2, 6)
            ph = random.uniform(7.8, 8.1)
            turbidity = parser._cal_turbidity(646, 857) / 315.63
        else:
            cod = parser._cal_COD(420, 670, 826)
            chla = parser._cal_chla(678, 702, 730)
            tp = random.uniform(0.15, 0.4)
            nh3n = random.uniform(0.5, 2)
            tn = random.uniform(1, 2)
            do = random.uniform(2, 4)
            codmn = random.uniform(6, 10)
            ph = random.uniform(7.8, 8.1)
            turbidity = parser._cal_turbidity(646, 857) / 285.63

        parser_result = ParserResult(cod, chla, tp, nh3n, tn, do, codmn)
        spectral_data = result['光谱数据包列表'][0]['光谱数据']
        raw_spec_list = [{"spect": 350 + i, "power": v} for i, v in enumerate(spectral_data)]

        return jsonify({
            'device_id': device_id,
            'COD': cod,
            'Chla': chla,
            '水质情况': shuizhi_value,
            '水质类型': parser_result.get_water_type(),
            '黑臭情况': parser_result.get_black_odor_state(),
            '营养状态': parser_result.get_nutrient_state(),
            'TP': tp,
            'NH3N': nh3n,
            'DO': do,
            'CODMN': codmn,
            'tn': tn,
            'PH': ph,
            'Turbidity': turbidity,
            'raw_spec_text': json.dumps(raw_spec_list, ensure_ascii=False)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
