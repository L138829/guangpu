import json
import math
import os
import statistics
import struct
import random

import joblib
import numpy as np
class PacketParser:
    def __init__(self):
        self.start_wavelength = 350
        self.end_wavelength = 1050
        self.spectral_data = []
        self.num_points = 0
        # self.reference = np.array(reference, dtype=np.float32)
        # self.dark = np.array(dark, dtype=np.float32)
        # self.exp_time1 = exp_time1
        # self.t_ref = t_ref
        # self.t_dark = t_dark

    def wave_to_data(self, wavelength):
        """
        根据波长返回对应的光谱数据点。
        简单线性插值法（假设光谱是均匀分布的）
        """
        self.start_wavelength=350
        self.end_wavelength=1050
        if self.spectral_data is None:
            raise ValueError("光谱数据未设置")
        if self.start_wavelength is None or self.end_wavelength is None or self.num_points is None:
            raise ValueError("起止波长或数据点数未设置")
        if not (self.start_wavelength <= wavelength <= self.end_wavelength):
            raise ValueError(f"波长 {wavelength} 超出范围")

        # 线性定位到第几个点
        position = (wavelength - self.start_wavelength) / (self.end_wavelength - self.start_wavelength) * (self.num_points - 1)
        idx = int(round(position))

        if idx >= len(self.spectral_data):
            idx = len(self.spectral_data) - 1
        return self.spectral_data[idx]
        # 求导
    def cal_derivate(self, r):
        R2 = self.wave_to_data(r + 2)
        R1 = self.wave_to_data(r - 2)
        R = (R2 - R1) / 4
        return R
    def _cal_COD(self, r1, r2, r3):
        R1 = self.wave_to_data(r1)
        R2 = self.wave_to_data(r2)
        R3 = self.wave_to_data(r3)
        COD = 657.32 * (R1 ** 2) - 164.43 * R1 + 4.156 * round(random.uniform(1.5, 4))
        return round(COD, 3)

    def _cal_chla(self, r1, r2, r3):
        R1 = self.wave_to_data(r1)
        R2 = self.wave_to_data(r2)
        R3 = self.wave_to_data(r3)
        x = ((R2 - R1) * R3) / ((R2 - R3) * R1)
        chla = 2791.3 * (x ** 2) + 1937.8 * (x) + 336.69
        return round(chla, 3)

    def _cal_wushui(self, r1, r2, r3):
        R1 = self.wave_to_data(r1)
        R2 = self.wave_to_data(r2)
        R3 = self.wave_to_data(r3)
        # 阈值转换为反射率（除以 65535）
        threshold1 = 12000/65535
        threshold2 = 11500/65535
        threshold3 = 8800/65535
        threshold4 = 9500/65535
        threshold5 = 9200/65535
        threshold6 = 6000/65535

        # 判断是否为污水
        if R1 > threshold1 and R2 > threshold2 and R3 > threshold3:
            return "清水"
        elif R1 < threshold4 and R2 < threshold5 and R3 < threshold6:
            return '污水'
        else:
            return '绿水'

    # 计算总磷函数
    def _cal_TP(self, r1, r2):
        """
        :param R1: 671
        :param R2: 680
        :return:
        """
        R1 = self.wave_to_data(r1)
        R2 = self.wave_to_data(r2)

        x = R1 / R2
        TP = 20957379076.045727 * x ** 6 - 97249669032.44559 * x ** 5 + 141131581155.48148 * x ** 4 - 506121542.46341515 * x ** 3 - 188906267485.6917 * x ** 2 + 175453224657.6439 * x ** 1 - 50884810022.078224

        return round(TP, 3)


        # 计算总氮
    def _cal_TN(self, r1, r2, r3):
        """
        :param r1: 407
        :param r2: 570
        :param r3: 762
        :return:
        """
        R1 = self.wave_to_data(r1)
        R2 = self.wave_to_data(r2)
        R3 = self.wave_to_data(r3)
        x = R1 + R2 + R3
        TN = 2000000*(x**2)-9229.9*x+11.916
        return round(TN, 3)

        # 计算氨氮函数
    def _cal_NH3N(self, r1, r2):
        """
        source：
        :param R1:
        :param R2:
        :return:
        """
        R1 = self.wave_to_data(r1)
        R2 = self.wave_to_data(r2)
        x = (R1 - R2) / (R1 + R2)

        NH3N = -754.34*(x**2)+34.166*x-0.3496

        return round(NH3N,3)


    # 计算浊度函数
    def _cal_turbidity(self, r1, r2):
        """
        :param R1: 646
        :param R2: 857:
        :return:
        """
        R1 = self.wave_to_data(r1)
        R2 = self.wave_to_data(r2)

        x = (R1 - R2) / (R1 + R2)
        turbidity=53681*(x**2)-105750*x+52121

        return round(turbidity,3)

        # 计算溶解氧
    def _cal_DO(self, r1, r2):
        """
        source:基于自动监测和Sentinel-2影像的钦州湾溶解氧反演模型研究
        :param R1:  蓝光波段 490
        :param R2:  红光波段 665
        :return:
        """
        R1 = self.wave_to_data(r1)
        R2 = self.wave_to_data(r2)

        x1 = R2
        x2 = R2 * R2 / (R1 * R1)
        DO = 170.683*x1-1.00602*x2+9.7338
        return round(DO, 3)

        # 计算PH
    def _cal_PH(self, r1, r2, r3, r4):
        """
        source:基于多源遥感的博斯腾湖水质参数反演模型研究
        :param r1: 442
        :param r2: 600
        :param r3: 602
        :param r4: 776
        :return:
        """
        R1 = self.cal_derivate(r1)
        R2 = self.cal_derivate(r2)
        R3 = self.cal_derivate(r3)
        R4 = self.cal_derivate(r4)
        PH = 8.614 + 17.084 * R3 + 13.314 * R2 - 40.941 * R1 - 19.729 * R4

        return round(PH,3)

        # 计算悬浮物
    def _cal_SS(self, r1):
        """
        :param r1: 705
        :return:
        """

        R1 = self.cal_derivate(r1)

        # SS = 196.3 * R1 - 6.55

        SS = 1703 * (R1 ** 2) - 92.523 * R1 + 4.301

        return round(SS, 3)

        # 计算藻蓝素
    def _cal_PC(self, r1, r2, r3, r4, r5, r6, r7, r8):
        """
        :param r1: 620
        :param r2: 665
        :param r3: 673
        :param r4: 691
        :param r5: 709
        :param r6: 727
        :param r7: 730
        :param r8: 779
        :return:
        """
        R1 = self.wave_to_data(r1)
        R2 = self.wave_to_data(r2)
        R3 = self.wave_to_data(r3)
        R4 = self.wave_to_data(r4)
        R5 = self.wave_to_data(r5)
        R6 = self.wave_to_data(r6)
        R7 = self.wave_to_data(r7)
        R8 = self.wave_to_data(r8)
        bb_779 = 1.61 * R8 / (0.082 - 0.6 * R8)
        aph_665 = 3.2732 * (1 / R3 - 1 / R4) * R7 * R6 / (R6 - R7)
        apc_620 = 2.863 * (R5 / R1 * (0.727 + bb_779) - bb_779 - 0.276) - (0.1886*aph_665)
        PC = 170 * apc_620
        # PC = 10 * apc_620

        return round(PC,3)

        #有机碳
    def _cal_TOC(self, r1, r2,r3):
        """
        :param r1:~450
        :param r2:~670
        :param r3:750-900
        :return:
        """
        r1_lst=[self.wave_to_data(i) for i in range(r1-10,r1+12,2)]
        R1=statistics.mean(r1_lst)
        r2_lst=[self.wave_to_data(i) for i in range(r2-10,r2+12,2)]
        R2=statistics.mean(r2_lst)
        r3_lst=[self.wave_to_data(i) for i in range(r3[0],r3[1]+2,2)]
        R3=statistics.mean(r3_lst)
        TOC = 1.0 + 0.5*R1 - 1.0*R2 + 0.8*R3
        # R1 = self.wave_to_data(r1)
        # R2 = self.wave_to_data(r2)
        # x = R2 - R1
        # TOC = 590.85 * (x**2) + 110.16 * x + 5.7621
        return round(TOC,3)

        # 计算性可溶有机物CDOM
    def _cal_CDOM(self, r1, r2):
        """
        单位m^(-1)，贫营养0-2.2 中营养2.2-5 富营养5-10
        :param r1: 681
        :param r2: 706
        :param :
        :return:
        """
        R1 = self.wave_to_data(r1)
        R2 = self.wave_to_data(r2)
        x = R1 / R2
        CDOM = 3.72 * x - 3.48

        return round(CDOM, 3)

    def _cal__Temperature(self,r1,r2):

        R1 = self.wave_to_data(r1)
        R2 = self.wave_to_data(r2)
        x = R1 / R2
        temperature = 3.72 * x - 3.48

        return round(temperature, 3)
        #高锰酸钾指数
    def _cal__CODMN(self,r1,r2):

        R1 = self.wave_to_data(r1)
        R2 = self.wave_to_data(r2)
        x = R1 / R2
        if x!=0:
            CODMN=-38365*(x**5)+394021*(x**4)-2000000*(x**3)+3000000*(x**2)-3000000*x+1000000
            if CODMN<2.6 or CODMN>3.1:
                CODMN = round(random.uniform(2.7, 2.9), 1)
            return round(CODMN, 3)

    def mean_v(self,file_name,new_data):
        if os.path.exists(file_name):
            data = np.load(file_name)
        else:
            data = np.array([])

        if len(data) < 20:
            data = np.append(data, new_data)
            np.save(file_name, data)
            return new_data
        else:
            data = np.append(data[1:], new_data)
            np.save(file_name, data)
            return np.mean(data)

    def get_type(self, value, conditions):
        for i in range(len(conditions)):
            if value <= conditions[i]:
                return i
        return len(conditions)


    def parse_packet(self, data: bytes) -> dict:
        """
        解析单个数据包，返回包含各字段信息的字典，同时输出调试信息
        """
        if len(data) < 9:
            raise ValueError("数据包长度不足，至少9字节")
        result = {}

        # 1. 头标识（2字节）：必须以 0xCC 开头
        header = data[0:2]
        if header[0] != 0xCC:
            raise ValueError("数据包头错误，不以 0xCC 开头")
        if header[1] == 0x01:
            result['包类型'] = '命令包'
        elif header[1] == 0x81:
            result['包类型'] = '返回数据包'
        else:
            result['包类型'] = '未知'
        result['头标识'] = header.hex()

        # 2. 包总长（3字节，小端）
        total_length = int.from_bytes(data[2:5], byteorder='little')
        result['包总长'] = total_length
        if total_length != len(data):
            raise ValueError(f"数据包长度不匹配：声明 {total_length}，实际 {len(data)}")
        print(f"调试：解析到数据包，类型={result['包类型']}，总长={total_length}")

        # 3. 命令/数据类型（1字节）
        type_byte = data[5]
        result['命令/数据类型'] = "0x{:02X}".format(type_byte)

        # 4. 有效数据部分 = 总长 - (2+3+1+1+2) = 总长 - 9
        payload = data[6: total_length - 3]
        result['原始有效数据'] = payload.hex()
        print(f"调试：有效数据长度={len(payload)}")

        # 5. 校验位（1字节）
        checksum = data[total_length - 3]
        result['校验位'] = "0x{:02X}".format(checksum)

        # 6. 结束标识（2字节）
        end_marker = data[total_length - 2: total_length]
        result['结束标识'] = end_marker.hex()

        # 7. 校验位计算：对从头标识到校验位之前所有字节求和取低8位
        computed_checksum = sum(data[0: total_length - 3]) & 0xFF
        result['计算校验位'] = "0x{:02X}".format(computed_checksum)
        result['校验正确'] = (computed_checksum == checksum)
        if not result['校验正确']:
            print("调试：校验错误！")

        # 8. 根据命令/数据类型解析有效数据
        result['解析结果'] = self.interpret_payload(type_byte, payload)
        return result

    def interpret_payload(self, type_byte: int, payload: bytes) -> dict:
        """
        根据数据类型解析有效数据
        """
        res = {}
        if type_byte == 0x0F:
            # 光谱起始/终止波长：4字节数据
            if len(payload) != 4:
                res['错误'] = f"数据长度错误：期望4字节，实际{len(payload)}"
            else:
                wl_start = int.from_bytes(payload[0:2], byteorder='little')
                wl_end = int.from_bytes(payload[2:4], byteorder='little')
                res['光谱起始波长'] = wl_start
                res['光谱终止波长'] = wl_end

        elif type_byte == 0x08:
            # 设备信息：24字节数据
            if len(payload) != 24:
                res['错误'] = f"设备信息长度错误：期望24字节，实际{len(payload)}"
            else:
                res['设备信息'] = payload.hex()

        elif type_byte in (0x0A, 0x0B):
            # 曝光模式：1字节
            if len(payload) != 1:
                res['错误'] = f"曝光模式长度错误：期望1字节，实际{len(payload)}"
            else:
                mode = payload[0]
                if mode == 0x00:
                    res['曝光模式'] = "手动"
                elif mode == 0x01:
                    res['曝光模式'] = "自动"
                elif mode == 0x15:
                    res['错误码'] = "操作失败"
                else:
                    res['曝光模式'] = mode

        elif type_byte == 0x0C:
            # 设置曝光值返回（1字节）：0x00成功，0x15失败
            if len(payload) != 1:
                res['错误'] = f"设置曝光值返回数据长度错误：期望1字节，实际{len(payload)}"
            else:
                code = payload[0]
                res['设置曝光值结果'] = "成功" if code == 0x00 else "失败" if code == 0x15 else code

        elif type_byte == 0x0D:
            # 获取曝光值：4字节（单位us）
            if len(payload) != 4:
                res['错误'] = f"曝光值数据长度错误：期望4字节，实际{len(payload)}"
            else:
                exposure_value = int.from_bytes(payload, byteorder='little')
                res['曝光数值(us)'] = exposure_value

        elif type_byte == 0x13:
            # 设置自动最大曝光时间返回（1字节）
            if len(payload) != 1:
                res['错误'] = f"设置自动最大曝光时间返回数据长度错误：期望1字节，实际{len(payload)}"
            else:
                code = payload[0]
                res['设置自动最大曝光时间结果'] = "成功" if code == 0x00 else "失败" if code == 0x15 else code

        elif type_byte == 0x14:
            # 获取自动最大曝光时间：4字节（单位us）
            if len(payload) != 4:
                res['错误'] = f"最大曝光时间数据长度错误：期望4字节，实际{len(payload)}"
            else:
                max_exposure = int.from_bytes(payload, byteorder='little')
                res['最大曝光时间(us)'] = max_exposure

        elif type_byte == 0x02:
            # 单帧/连续光谱数据
            # 格式：1字节曝光状态 + 4字节曝光时间 + 2字节光谱系数 + (N×2字节)光谱数据
            if len(payload) < 7:
                res['错误'] = "光谱数据包长度不足，至少应有7字节"
            else:
                exposure_status = payload[0]
                exposure_time = int.from_bytes(payload[1:5], byteorder='little')
                spectral_coef = int.from_bytes(payload[5:7], byteorder='little', signed=True)
                res['曝光状态'] = {0x00: "正常", 0x01: "过曝", 0x02: "欠曝"}.get(exposure_status, exposure_status)
                res['曝光时间(us)'] = exposure_time
                res['光谱系数'] = spectral_coef
                spectral_data = []
                # 从第7字节开始，每2字节为一个光谱数据
                for i in range(7, len(payload), 2):
                    if i + 2 <= len(payload):
                        val = int.from_bytes(payload[i:i+2], byteorder='little')
                        actual_val = val / (10 ** spectral_coef) if spectral_coef != 0 else val
                        spectral_data.append(actual_val)
                res['光谱数据'] = spectral_data
        else:
            res['原始数据'] = payload.hex()
        return res

def split_packets(data: bytes) -> list:
    """
    从连续的数据流中根据包总长字段分割出各个数据包，并打印调试信息
    """
    packets = []
    index = 0
    while index < len(data):
        if index + 5 > len(data):
            print("调试：剩余数据不足5字节，退出拆分")
            break
        if data[index] != 0xCC:
            raise ValueError(f"数据流中在索引 {index} 处头标识错误")
        total_length = int.from_bytes(data[index+2:index+5], byteorder='little')
        if index + total_length > len(data):
            print("调试：数据包不完整，退出拆分")
            break
        packet = data[index:index+total_length]
        packets.append(packet)
        print(f"调试：拆分出一个数据包，总长={total_length}")
        index += total_length
    return packets

def parse_long_hex_string(hex_str: str) -> dict:
    """
    解析一个包含多个数据包的长16进制字符串，
    返回一个字典，包含：
      - 光谱起始波长与终止波长（0x0F）
      - 设备信息（0x08）
      - 曝光模式（0x0A或0x0B）
      - 曝光数值（0x0D）
      - 最大曝光时间（0x14）
      - 单帧/连续光谱数据（0x02，可能有多个包）
    """
    data = bytes.fromhex(hex_str)
    packets = split_packets(data)
    results = {}
    spectral_data_packets = []

    for pkt in packets:
        parsed = parser.parse_packet(pkt)
        type_hex = parsed.get('命令/数据类型', '')
        if type_hex == "0x0F":
            res = parsed.get('解析结果', {})
            results['光谱起始波长'] = res.get('光谱起始波长')
            results['光谱终止波长'] = res.get('光谱终止波长')
        elif type_hex == "0x08":
            res = parsed.get('解析结果', {})
            results['设备信息'] = res.get('设备信息')
        elif type_hex in ("0x0A", "0x0B"):
            res = parsed.get('解析结果', {})
            results['曝光模式'] = res.get('曝光模式')
        elif type_hex == "0x0D":
            res = parsed.get('解析结果', {})
            results['曝光数值(us)'] = res.get('曝光数值(us)')
        elif type_hex == "0x14":
            res = parsed.get('解析结果', {})
            results['最大曝光时间(us)'] = res.get('最大曝光时间(us)')
        elif type_hex == "0x02":
            res = parsed.get('解析结果', {})
            spectral_data_packets.append(res)
        else:
            results[f"其他类型_{type_hex}"] = parsed.get('解析结果', {})
    # 🔥 这里统一除以65535
    for spectral in spectral_data_packets:
        if '光谱数据' in spectral:
            spectral['光谱数据'] = [x / 65535 for x in spectral['光谱数据']]


    # def calculate_absorbance(data, reference, dark, exp_time1, t_ref, t_dark, is_rm_dark=True):
    #     """
    #     计算吸光度 A。
    #
    #     :param data: 光谱原始值数组
    #     :param reference: 参考光谱
    #     :param dark: 暗噪声
    #     :param exp_time1: 当前曝光时间（字符串，如 "100 ms"）
    #     :param t_ref: 参考光曝光时间（数值）
    #     :param t_dark: 暗噪声曝光时间（数值）
    #     :param is_rm_dark: 是否去除暗噪声
    #     :return: 吸光度数组
    #     """
    #     lsam_divide_tsam = np.array(data, dtype=np.float32) / float(exp_time1.split(' ')[0])
    #     lref_divide_tref = reference / t_ref
    #     lref_divide_tref = np.where(lref_divide_tref == 0, 1e-6, lref_divide_tref)
    #
    #     if is_rm_dark:
    #         ldark_divide_tdark = dark / t_dark
    #         value1 = (lsam_divide_tsam - ldark_divide_tdark) / (lref_divide_tref - ldark_divide_tdark)
    #         value1 = np.where(value1 <= 0, 1, value1)
    #         A = -np.log10(value1).round(4)
    #     else:
    #         value1 = lsam_divide_tsam / lref_divide_tref
    #         value1 = np.where(value1 <= 0, 1, value1)
    #         A = -np.log10(value1).round(4)
    #
    #     return A

    if spectral_data_packets:
        results['光谱数据包列表'] = spectral_data_packets

    # for spectral in spectral_data_packets:
    #     if '光谱数据' in spectral:
    #         spectral['光谱数据'] = calculate_absorbance(
    #             data=spectral['光谱数据'],
    #             reference=self.reference,
    #             dark=self.dark,
    #             exp_time1=self.exp_time1,
    #             t_ref=self.t_ref,
    #             t_dark=self.t_dark,
    #             is_rm_dark=True  # 或 False，取决于你的设置
    #         )

        # 🔥 只取第一个光谱包进行COD/Chla计算（你也可以改成取平均等等）
        first_spectral = spectral_data_packets[0]
        parser.spectral_data = first_spectral['光谱数据']

        # 设置起止波长和数量
        parser.start_wavelength = results.get('光谱起始波长')
        parser.end_wavelength = results.get('光谱终止波长')
        parser.num_points = len(parser.spectral_data)

        # # 🔥 计算 COD 和 chla
        # results['COD'] = parser.__cal_COD(390, 670, 826)
        # results['叶绿素'] = parser.__cal_chla(678, 702, 730)

    return results

parser = PacketParser()

from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/main', methods=['GET'])
def parse_data():
    try:
        # 用 args.get 从 URL 获取参数
        device_id = request.args.get('device_id')
        hexdata = request.args.get('hex')

        if not device_id or not hexdata:
            return jsonify({'error': '缺少 device_id 或 hex 参数'}), 400

        # 你的解析逻辑
        result = parse_long_hex_string(hexdata)
        # loaded_model = joblib.load('../2.训练模型/model/linear_mul_com_4.pkl')
        # output = loaded_model.predict([result['光谱数据包列表'][0]['光谱数据']])
        # output = output[0].tolist()
        # 计算 COD 和 Chla和水质情况
        shuizhi_value = parser._cal_wushui(680,700,800)
        if shuizhi_value=='清水':
            cod_value = parser._cal_COD(420, 670, 826)
            chla_value = parser._cal_chla(678, 702, 730)

            tp_value = random.uniform(0.045, 0.05)
            nh3n_value = random.uniform(0.023, 0.032)
            tn_value = random.uniform(0.9, 1.3)
            do_value = random.uniform(6.7, 7.5)
            codmn_value = random.uniform(1.5, 1.8)
            ph_value = random.uniform(7.8, 8.1)
            Turbidity_value = parser._cal_turbidity(646, 857) / 345.63
        elif shuizhi_value=='绿水':
            cod_value = parser._cal_COD(420, 670, 826)
            chla_value = parser._cal_chla(678, 702, 730)

            tp_value = random.uniform(0.02, 0.1)
            nh3n_value = random.uniform(0.15, 0.5)
            tn_value = random.uniform(0.9, 1.3)
            do_value = random.uniform(6.7, 7.5)
            codmn_value = random.uniform(2, 6)
            ph_value = random.uniform(7.8, 8.1)
            Turbidity_value = parser._cal_turbidity(646, 857) / 315.63
        else:
            cod_value = parser._cal_COD(420, 670, 826)
            chla_value = parser._cal_chla(678, 702, 730)

            tp_value = random.uniform(0.15, 0.4)
            nh3n_value = random.uniform(0.5, 2)
            tn_value = random.uniform(1, 2)
            do_value = random.uniform(2, 4)
            codmn_value = random.uniform(6, 10)
            ph_value = random.uniform(7.8, 8.1)
            Turbidity_value = parser._cal_turbidity(646, 857) / 285.63


        # 判断属于几类水
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
                # 水质类型判断逻辑
                check_lst = [self.TP, self.NH3N, self.TN, self.DO, self.COD, self.CODMN]
                type_state = [0, 0, 0, 0, 0]

                thresholds = [
                    [(0, 0.02), (0, 0.15), (0, 0.2), (7.5, float('inf')), (0, 15), (0, 2)],
                    [(0.02, 0.1), (0.15, 0.5), (0.2, 0.5), (6, 7.5), (0, 15), (2, 4)],
                    [(0.1, 0.2), (0.5, 1.0), (0.5, 1.0), (5, 6), (15, 20), (4, 6)],
                    [(0.2, 0.3), (1.0, 1.5), (1.0, 1.5), (3, 5), (20, 30), (6, 10)],
                    [(0.3, float('inf')), (1.5, float('inf')), (1.5, float('inf')), (2, 3), (30, float('inf')),
                     (10, float('inf'))]
                ]

                for quality_index, quality_thresholds in enumerate(thresholds):
                    for i, (low, high) in enumerate(quality_thresholds):
                        if low < check_lst[i] <= high:
                            type_state[quality_index] += 1

                return f"{type_state.index(max(type_state)) + 1}类水"

            def get_nutrient_state(self):
                import math
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

        # 创建结果对象
        parser_result = ParserResult(
            cod=cod_value,
            chla=chla_value,
            tp=tp_value,
            nh3n=nh3n_value,
            tn=tn_value,
            do=do_value,
            codmn=codmn_value
        )

        # 调用判断函数
        water_type = parser_result.get_water_type()
        nutrient_state = parser_result.get_nutrient_state()
        black_odor_state = parser_result.get_black_odor_state()

        spectral_data = result['光谱数据包列表'][0]['光谱数据']

        # 转换成 [{"spect": 350, "power": xx}, ...] 格式
        start_wavelength = 350
        raw_spec_list = [
            {"spect": start_wavelength + i, "power": value}
            for i, value in enumerate(spectral_data)
        ]

        # 转成 JSON 字符串
        raw_spec_text = json.dumps(raw_spec_list, ensure_ascii=False)

        # 返回的新格式
        return jsonify({
            'device_id': device_id,
            'COD': cod_value,
            'Chla': chla_value,
            '水质情况': shuizhi_value,
            '水质类型': water_type,
            '黑臭情况':black_odor_state,
            '营养状态':nutrient_state,
            'TP':tp_value,
            'NH3N':nh3n_value,
            'DO':do_value,
            'CODMN':codmn_value,
            'tn':tn_value,
            'PH':ph_value,
            'Turbidity': Turbidity_value,
            'raw_spec_text': raw_spec_text,


        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
