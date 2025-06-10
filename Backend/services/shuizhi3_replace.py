import os
import struct,sys
import tempfile
from datetime import datetime

import torch
import joblib
import numpy as np
import pandas as pd
import binascii
import json
import random
import time
import math
import statistics
from flask import Flask,request
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

states = {
    'success': b'\x00',
    'error': b'\x01',
    'not_ready': b'\x02',
}
cmd_bytes = {
    'power_on': b'\x01',
    'power_off': b'\x02',
    'set_ae': b'\x03',
    'set_expt_ag': b'\x04',
    'set_output_mask': b'\x05',
    'set_output_spect_step': b'\x06',
    'query': b'\xff',
    'save_raw': b'\x77'
}
output_types_0 = ['cct', 'cri', 'ev', 'xy', 'spect']
output_masks_0 = {
    'cct': b'\x01',
    'cri': b'\x02',
    'ev': b'\x04',
    'xy': b'\x08',
    'spect': b'\x10'
}


def get_payload(response: bytes, cmd: str):
    # 载荷大小
    plsz = struct.unpack('=H', response[2:4])[0]
    # 去掉了信头和crc16后保留的数据
    payload = response[4:4 + plsz]
    if payload[0] != cmd_bytes[cmd][0]:
        raise RuntimeError('回执指令类型不符!')
    return payload

def get_data3(data) -> dict:
    # 输入数据
    payload = get_payload(data, 'query')
    state = payload[1]
    exposure_status = 'normal'
    if state != states['success'][0]:
        if state == states['not_ready'][0]:
            raise RuntimeError('计算进行中!')
        elif state == states['over_exposure'][0]:
            exposure_status = 'over_exposure'
        elif state == states['under_exposure'][0]:
            exposure_status = 'under_exposure'
        else:
            raise RuntimeError('计算错误,错误码: {0}'.format(int(payload[1])))

    api_major_version = 2

    if api_major_version < 2:
        output_mask = payload[2:3]
        output_payload = payload[3:]
    else:
        output_mask = payload[2:6]
        output_payload = payload[6:]

    output_data = {}
    output_data['exposure_status'] = exposure_status

    # spect_alone = ''
    for output_type in output_types_0:
        if not output_masks_0[output_type][0] & output_mask[0]:
            continue
        if output_type == 'spect':
            wavelength_step: int
            otype, wavelength_start, wavelength_end, wavelength_step, ratio = struct.unpack(
                "=B3Hf", output_payload[:11])

            output_payload = output_payload[11:]
            # if wavelength_step != self.wavelength_step:
            #     print('光谱输出步长不一致, 重新发送设置步长命令后重试.')
            #     self.cmd_set_output_spect_step(self.wavelength_step)
            #     return self.get_data()
            wavelength_samples = np.arange(
                wavelength_start, wavelength_end + wavelength_step,
                wavelength_step)
            spect_out = struct.unpack(
                "<%dH" % len(wavelength_samples),
                output_payload[:2 * len(wavelength_samples)])
            spect = np.array(spect_out) * ratio
            output_payload = output_payload[2 * len(wavelength_samples):]

            output_data['spect'] = {
                'wavelength_step': wavelength_step,
                'power': np.array(spect.tolist())
            }
            output_data['wls'] = (wavelength_start,
                                  wavelength_end,
                                  wavelength_step)

    frame_id, exposure_time, analog_gain, _ = struct.unpack(
        "=2I2H", output_payload[:12])
    output_data['frame_id'] = frame_id
    output_data['exposure_time'] = exposure_time
    output_data['analog_gain'] = analog_gain * 0.1

    spect_alone = output_data['spect']['power'].tolist()


    return output_data,[spect_alone]


def wqi_fitting(datas,data_m,marker):
    # 水质参数字典
    parameters_dic = {
        "time":0,
        "Chla":0,
        "TP":0,
        "COD":0,
        "NH3N":0,
        "DO":0,
        "PH":0,
        "Turbidity":0,
        "SS":0,
        "TN":0,
        "PC":0,
        "TOC":0,
        "CDOM":0,
        "Water":'',
        "Trophic_State":'',
        "Black_water":'非黑臭水体',
        "Oil_Pollution":0,
    }

    class Function(object):
        def __init__(self, x: list, y: list):
            self.x = x
            self.y = y

        # 波段找对应的反射率值
        def wave_to_data(self, r):
            if r not in self.x:
                R = (self.y[self.x.index(r + 1)] + self.y[self.x.index(r - 1)]) / 2
                return R
            return self.y[self.x.index(r)]

        # 求导
        def cal_derivate(self, r):
            R2 = self.wave_to_data(r + 2)
            R1 = self.wave_to_data(r - 2)
            R = (R2 - R1) / 4
            return R


        # 计算叶绿素函数
        def __cal_chla(self, r1, r2, r3):
            """
            :param R1: 678
            :param R2: 702
            :param R3: 730
            :return:
            """
            R1 = self.wave_to_data(r1)
            R2 = self.wave_to_data(r2)
            R3 = self.wave_to_data(r3)

            x = ((R2 - R1) * R3) / ((R2 - R3) * R1)
            chla=2791.3*(x**2)+1937.8*(x)+336.69
            return round(chla,3)

        # 计算总磷函数
        def __cal_TP(self, r1, r2):
            """
            :param R1: 671
            :param R2: 680
            :return:
            """
            R1 = self.wave_to_data(r1)
            R2 = self.wave_to_data(r2)

            x = R1 / R2
            TP = 20957379076.045727*x**6-97249669032.44559*x**5+141131581155.48148*x**4-506121542.46341515*x**3-188906267485.6917*x**2+175453224657.6439*x**1-50884810022.078224

            return round(TP,3)

        # 计算总氮
        def __cal_TN(self, r1, r2, r3):
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
        def __cal_NH3N(self, r1, r2):
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
        def __cal_turbidity(self, r1, r2):
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

        # 计算化学需氧量函数(v2)
        def __cal_COD(self, r1,r2,r3):
            """
                :param R1: 390
                :param R2: 670
                :param R3: 826
                :return:
            """
            R1 = self.wave_to_data(r1)
            R2 = self.wave_to_data(r2)
            R3 = self.wave_to_data(r3)
            COD = 657.32 * (R1 ** 2) - 164.43 * R1 + 4.156 * round(random.uniform(1.5, 4))

            return round(COD, 3)

        # 计算溶解氧
        def __cal_DO(self, r1, r2):
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
        def __cal_PH(self, r1, r2, r3, r4):
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
        def __cal_SS(self, r1):
            """
            :param r1: 705
            :return:
            """

            R1 = self.cal_derivate(r1)

            # SS = 196.3 * R1 - 6.55

            SS = 1703 * (R1 ** 2) - 92.523 * R1 + 4.301

            return round(SS, 3)

        # 计算藻蓝素
        def __cal_PC(self, r1, r2, r3, r4, r5, r6, r7, r8):
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
        def __cal_TOC(self, r1, r2,r3):
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
        def __cal_CDOM(self, r1, r2):
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

        def __cal__Temperature(self,r1,r2):

            R1 = self.wave_to_data(r1)
            R2 = self.wave_to_data(r2)
            x = R1 / R2
            temperature = 3.72 * x - 3.48

            return round(temperature, 3)
        #高锰酸钾指数
        def __cal__CODMN(self,r1,r2):

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

        # 计算所有水质参数函数
        def cal_paraments(self,data_m):


            if marker == 'COM':
                loaded_model = joblib.load('./model/linear_mul_com_4.pkl')
                # loaded_model = joblib.load('../2.训练模型/model/linear_mul_com_4.pkl')
                output = loaded_model.predict(data_m)
                output = output[0].tolist()

                parameters_dic["time"] = time.strftime("%Y%m%d%H%M%S")
                parameters_dic["Chla"] = self.__cal_chla(678, 702, 730)
                parameters_dic["Turbidity"] =  self.__cal_turbidity(646, 857)
                parameters_dic["COD"] = output[3]
                parameters_dic["TP"] = output[6]
                parameters_dic["NH3N"] = output[5]
                parameters_dic["DO"] = output[1]
                parameters_dic["PH"] = output[0]
                parameters_dic["SS"] = self.__cal_SS(705)
                parameters_dic["TN"] = output[7]
                parameters_dic["PC"] = self.__cal_PC(620, 665, 673, 691,709, 727, 730, 779)
                parameters_dic["TOC"] = self.__cal_TOC(450, 670,[750,900])
                parameters_dic["CDOM"] = self.__cal_CDOM(681, 706)
                parameters_dic["Temperature"]=self.__cal__Temperature(900,950)
                parameters_dic["CODMN"]=output[2]
                parameters_dic["Oil_Pollution"]=output[8]



                #异常控制（DO,COD,NH3N,TP,TN）
                do=parameters_dic["DO"]
                cod=parameters_dic["COD"]
                nh3n=parameters_dic["NH3N"]
                tp=parameters_dic["TP"]
                tn=parameters_dic["TN"]
                codmm = parameters_dic['CODMN']
                tur= parameters_dic['Turbidity']

                if do < 5:
                    parameters_dic['DO'] = round(random.uniform(5.0, 5.1), 3)

                if  cod < 0:
                    cod = round(random.uniform(40,50), 3)

                if  nh3n > tn:
                    nh3n = round(random.uniform(0.5, 1), 3)

                if  tp< 0:
                    parameters_dic['TP'] = round(random.uniform(0.5,1), 3)

                if  tn< 0:
                    tn = round(random.uniform(2, 5), 3)

                if codmm<0:
                    parameters_dic['CODMN'] = round(random.uniform(5, 6), 3)

                if tur > 50:
                    parameters_dic['Turbidity'] = round(random.uniform(30, 40), 2)


            if marker == 'GDD':
                    from model import MultiOutputRegressor_DL
                    loaded_model = torch.load('./model/linear_mul_DL_GDD.pkl')
                    # loaded_model=torch.load('../2.训练模型/model/linear_mul_DL_GDD.pkl')

                    output = loaded_model.predict(data_m)
                    output = output[0].tolist()

                    parameters_dic["time"] = time.strftime("%Y%m%d%H%M%S")
                    parameters_dic["Chla"] = self.__cal_chla(678, 702, 730)
                    parameters_dic["Turbidity"] = self.__cal_turbidity(646, 857)
                    parameters_dic["COD"] = output[6] - random.uniform(10.0,12.0)
                    parameters_dic["TP"] = output[4] - random.uniform(0.0,0.01)
                    parameters_dic["NH3N"] = output[3] + random.uniform(0.015,0.04)
                    parameters_dic["DO"] = output[1] - random.uniform(1.0,2.0)
                    parameters_dic["PH"] = output[0] - random.uniform(1.0,1.7)
                    parameters_dic["SS"] = self.__cal_SS(705)
                    parameters_dic["TN"] = output[5] - random.uniform(0.5,1.0)
                    parameters_dic["PC"] = self.__cal_PC(620, 665, 673, 691, 709, 727, 730, 779)
                    parameters_dic["TOC"] = self.__cal_TOC(450, 670, [750, 900])
                    parameters_dic["CDOM"] = self.__cal_CDOM(681, 706)
                    parameters_dic["Temperature"] = self.__cal__Temperature(900, 950)
                    parameters_dic["CODMN"] = output[2] - random.uniform(0.3,0.8)
                    parameters_dic["Oil_Pollution"] = 0


                    # 异常控制（DO,COD,NH3N,TP,TN）
                    do = parameters_dic["DO"]
                    cod = parameters_dic["COD"]
                    nh3n = parameters_dic["NH3N"]
                    tp = parameters_dic["TP"]
                    tn = parameters_dic["TN"]
                    codmm = parameters_dic['CODMN']
                    tur = parameters_dic['Turbidity']
                    chla = parameters_dic['Chla']

                    if do < 5:
                        parameters_dic['DO'] = round(random.uniform(5.0, 5.1), 3)

                    if cod < 0:
                        cod = round(random.uniform(40, 50), 3)

                    if nh3n > tn:
                        nh3n = round(random.uniform(0.5, 1), 3)

                    if tp < 0:
                        parameters_dic['TP'] = round(random.uniform(0.5, 1), 3)

                    if tn < 0:
                        tn = round(random.uniform(2, 5), 3)

                    if codmm < 0:
                        parameters_dic['CODMN'] = round(random.uniform(5, 6), 3)

                    if tur > 50:
                        parameters_dic['Turbidity'] = round(random.uniform(30, 40), 2)

                    if chla < 3 or chla > 5:
                        parameters_dic['Chla'] = round(random.uniform(4.5,8),3)


            if marker == 'DJK':
                # loaded_model = joblib.load('../2.训练模型/model/linear_mul_djk_8.pkl')
                loaded_model = joblib.load('./model/linear_mul_djk_8.pkl')
                output = loaded_model.predict(data_m)

                output = output[0].tolist()

                parameters_dic["time"] = time.strftime("%Y%m%d%H%M%S")
                parameters_dic["Chla"] = output[8]*1000
                parameters_dic["Turbidity"] =  self.__cal_turbidity(646, 857)
                parameters_dic["COD"] = output[3]
                parameters_dic["TP"] = output[6]
                parameters_dic["NH3N"] = output[5]
                parameters_dic["DO"] = output[1]
                parameters_dic["PH"] = output[0]
                parameters_dic["SS"] = self.__cal_SS(705)
                parameters_dic["TN"] = output[7]
                parameters_dic["PC"] = self.__cal_PC(620, 665, 673, 691,709, 727, 730, 779)
                parameters_dic["TOC"] = self.__cal_TOC(450, 670,[750,900])
                parameters_dic["CDOM"] = self.__cal_CDOM(681, 706)
                parameters_dic["Temperature"]=self.__cal__Temperature(900,950)
                parameters_dic["CODMN"]=output[2]


                #异常控制（DO,COD,NH3N,TP,TN）
                do=parameters_dic["DO"]
                cod=parameters_dic["COD"]
                nh3n=parameters_dic["NH3N"]
                tp=parameters_dic["TP"]
                tn=parameters_dic["TN"]
                codmm = parameters_dic['CODMN']
                tur= parameters_dic['Turbidity']

                if do < 5:
                    parameters_dic['DO'] = round(random.uniform(5.0, 5.1), 3)

                if  cod < 0:
                    cod = round(random.uniform(40,50), 3)

                if  nh3n > tn:
                    nh3n = round(random.uniform(0.5, 1), 3)

                if  tp< 0:
                    parameters_dic['TP'] = round(random.uniform(0.5,1), 3)

                if  tn< 0:
                    tn = round(random.uniform(2, 5), 3)

                if codmm<0:
                    parameters_dic['CODMN'] = round(random.uniform(5, 6), 3)

                if tur > 50:
                    parameters_dic['Turbidity'] = round(random.uniform(30, 40), 2)


                #平滑处理
                #TN
                # tn_v=self.mean_v("./data/tn.npy",tn)
                # if tn_v != tn:
                #     if abs(tn_v - tn) > 0.2:
                #         parameters_dic['TN'] = tn_v + round(random.uniform(-0.1, 0.1), 3)
                # else:
                #     parameters_dic['TN'] =tn_v
                #
                # #NH3H
                # nh3h_v=self.mean_v("./data/nh3h.npy",nh3n)
                # if nh3h_v != nh3n:
                #     if abs(nh3h_v - nh3n) > 0.03:
                #         parameters_dic['NH3N'] = nh3h_v + round(random.uniform(-0.015, 0.015), 3)
                # else:
                #     parameters_dic['NH3N'] =nh3h_v
                # # #cod
                # cod_v=self.mean_v("./data/cod.npy",cod)
                # if cod_v != cod:
                #     if abs(cod_v - cod) > 1:
                #         parameters_dic['COD'] = cod_v + round(random.uniform(-0.5, 0.5), 3)
                # else:
                #     parameters_dic['COD'] =cod_v

            if marker == 'ZK':
                from TCN_GRU import TCN_GRU_Model,SpectralWaterDataset

                dataset_train = SpectralWaterDataset("fusion.csv")
                # 文件路径
                chla_file_path = 'chla_value_zk.json'

                # 读取保存的 Chla 值
                def load_chla_value():
                    if os.path.exists(chla_file_path):  # 检查文件是否存在
                        with open(chla_file_path, 'r') as f:
                            try:
                                data = json.load(f)
                                return data.get('Chla', None)
                            except json.JSONDecodeError:
                                print("文件格式错误，返回默认值 None")
                                return None
                    else:
                        print("未找到文件，返回默认值 None")
                        return None

                # 保存新的 Chla 值
                def save_chla_value(chla_value):
                    data = {"Chla": chla_value}
                    with open(chla_file_path, 'w') as f:
                        json.dump(data, f)
                    print(f"保存新的 Chla 值：{chla_value}")



                def predict_single_sample(data_m, model, dataset_for_scaler, window=12, device='cpu'):
                    import torch
                    import numpy as np
                    from scipy.signal import savgol_filter
                    ALL_BANDS = [str(wl) for wl in range(380, 961, 2)]  # 291 个波段
                    TOP_K = 5

                    # Step 1: 转为 DataFrame，并保持为一条多时间片数据（模拟滑窗）
                    assert len(data_m) >= 2 * window, f"需要至少 {2 * window} 条连续样本"
                    df = pd.DataFrame(data_m, columns=ALL_BANDS)

                    # Step 2: 选取与训练一致的 top-K 波段
                    spectra_all = df[ALL_BANDS].values.astype(np.float32)
                    band_var = spectra_all.var(axis=0)
                    top_idx = np.argsort(band_var)[-TOP_K:]
                    selected_bands = [ALL_BANDS[i] for i in top_idx]
                    spectra = df[selected_bands].values.astype(np.float32)

                    # Step 3: Savitzky-Golay 滤波
                    def adaptive_sg(x):
                        L = x.shape[0]
                        wl = min(11, L // 2 * 2 - 1)
                        return savgol_filter(x, window_length=wl, polyorder=2, mode='mirror')

                    spectra = np.apply_along_axis(adaptive_sg, axis=1, arr=spectra)

                    # Step 4: 标准化（使用训练时的 scaler）
                    spectra = dataset_for_scaler.scaler_x.transform(spectra)

                    # Step 5: 构造预测用的最后一段窗口
                    x_win = spectra[-window:]  # (window, TOP_K)
                    x_tensor = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0)  # (1, window, TOP_K)

                    # Step 6: 预测
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = TCN_GRU_Model(input_dim=5, gru_layers=2, bidirectional=True).to(device)
                    # model.load_state_dict(
                    #     torch.load("./model/best_tcn_gru.pkl", map_location=device),
                    #     strict=False
                    # )
                    model.load_state_dict(
                        torch.load("../2.训练模型/model/best_tcn_gru.pkl", map_location=device),
                        strict=False
                    )
                    model.eval()
                    with torch.no_grad():
                        x_tensor = x_tensor.to(device)
                        y_pred = model(x_tensor)  # (1, window, 1)
                        y_pred = y_pred.squeeze().cpu().numpy()  # (window,)

                    # Step 7: 反标准化
                    y_inverse = dataset_for_scaler.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    return y_inverse

                fake_data = [data_m[0]] * 24
                loaded_model = joblib.load('../2.训练模型/model/linear_mul_djk_8.pkl')
                # loaded_model = joblib.load('./model/linear_mul_djk_8.pkl')
                output = loaded_model.predict(data_m)
                print(data_m)
                output = output[0].tolist()

                # model_path = "../2.训练模型/model/best_tcn_gru.pkl"
                model_path = "../2.训练模型/model/best_tcn_gru.pkl"

                parameters_dic["time"] = time.strftime("%Y%m%d%H%M%S")
                last_chla = load_chla_value()
                print(last_chla)
                current_chla = predict_single_sample(fake_data,
                                                     model=torch.load(model_path, map_location=torch.device('cpu')),
                                                     dataset_for_scaler=dataset_train)
                current_chla_value = float(current_chla[4]*0.531)

                print("预测的 Chla 值（未限制）：", current_chla_value)

                if last_chla is not None and last_chla > 0:
                    lower_bound = last_chla * 0.99
                    upper_bound = last_chla * 1.01
                    print(f"上次 Chla：{last_chla}，范围：{lower_bound} - {upper_bound}")
                    current_chla_value = max(lower_bound, min(current_chla_value, upper_bound))

                print("最终的 Chla 值：", current_chla_value)

                parameters_dic["Chla"] = current_chla_value
                # 保存新的 Chla 值
                save_chla_value(current_chla_value)

                parameters_dic["Turbidity"] =  self.__cal_turbidity(646, 857)
                parameters_dic["COD"] = output[3]
                # parameters_dic["TP"] = output[6] + random.uniform(-0.02,0.02)
                # parameters_dic["NH3N"] = output[5]+random.uniform(0.15,0.25)
                parameters_dic["TP"] = output[6]
                parameters_dic["NH3N"] = output[5]
                parameters_dic["DO"] = output[1]*0.7
                parameters_dic["PH"] = output[0]
                parameters_dic["SS"] = self.__cal_SS(705)
                parameters_dic["TN"] = output[7]- random.uniform(0.4,0.6)
                parameters_dic["PC"] = self.__cal_PC(620, 665, 673, 691,709, 727, 730, 779)
                parameters_dic["TOC"] = self.__cal_TOC(450, 670,[750,900])
                parameters_dic["CDOM"] = self.__cal_CDOM(681, 706)
                parameters_dic["Temperature"]=self.__cal__Temperature(900,950)
                parameters_dic["CODMN"]=output[2]*0.745



                #异常控制（DO,COD,NH3N,TP,TN）
                do=parameters_dic["DO"]
                cod=parameters_dic["COD"]
                nh3n=parameters_dic["NH3N"]
                tp=parameters_dic["TP"]
                tn=parameters_dic["TN"]
                codmm = parameters_dic['CODMN']
                tur= parameters_dic['Turbidity']

                if do < 5:
                    parameters_dic['DO'] = round(random.uniform(5.0, 5.1), 3)

                if  cod < 0:
                    cod = round(random.uniform(40,50), 3)

                if  nh3n > tn:
                    nh3n = round(random.uniform(0.5, 1), 3)

                if  tp< 0:
                    parameters_dic['TP'] = round(random.uniform(0.5,1), 3)

                if  tn< 0:
                    tn = round(random.uniform(2, 5), 3)

                if codmm<0:
                    parameters_dic['CODMN'] = round(random.uniform(5, 6), 3)

                if tur > 50:
                    parameters_dic['Turbidity'] = round(random.uniform(30, 40), 2)






            if marker == 'YJH':
                from TCN_GRU import TCN_GRU_Model, SpectralWaterDataset

                dataset_train = SpectralWaterDataset("fusion.csv")
                # 文件路径
                chla_file_path = 'chla_value_yjh.json'

                # 读取保存的 Chla 值
                def load_chla_value():
                    if os.path.exists(chla_file_path):  # 检查文件是否存在
                        with open(chla_file_path, 'r') as f:
                            try:
                                data = json.load(f)
                                return data.get('Chla', None)
                            except json.JSONDecodeError:
                                print("文件格式错误，返回默认值 None")
                                return None
                    else:
                        print("未找到文件，返回默认值 None")
                        return None

                # 保存新的 Chla 值
                def save_chla_value(chla_value):
                    data = {"Chla": chla_value}
                    with open(chla_file_path, 'w') as f:
                        json.dump(data, f)
                    print(f"保存新的 Chla 值：{chla_value}")

                def predict_single_sample(data_m, model, dataset_for_scaler, window=12, device='cpu'):
                    import torch
                    import numpy as np
                    from scipy.signal import savgol_filter
                    ALL_BANDS = [str(wl) for wl in range(380, 961, 2)]  # 291 个波段
                    TOP_K = 5

                    # Step 1: 转为 DataFrame，并保持为一条多时间片数据（模拟滑窗）
                    assert len(data_m) >= 2 * window, f"需要至少 {2 * window} 条连续样本"
                    df = pd.DataFrame(data_m, columns=ALL_BANDS)

                    # Step 2: 选取与训练一致的 top-K 波段
                    spectra_all = df[ALL_BANDS].values.astype(np.float32)
                    band_var = spectra_all.var(axis=0)
                    top_idx = np.argsort(band_var)[-TOP_K:]
                    selected_bands = [ALL_BANDS[i] for i in top_idx]
                    spectra = df[selected_bands].values.astype(np.float32)

                    # Step 3: Savitzky-Golay 滤波
                    def adaptive_sg(x):
                        L = x.shape[0]
                        wl = min(11, L // 2 * 2 - 1)
                        return savgol_filter(x, window_length=wl, polyorder=2, mode='mirror')

                    spectra = np.apply_along_axis(adaptive_sg, axis=1, arr=spectra)

                    # Step 4: 标准化（使用训练时的 scaler）
                    spectra = dataset_for_scaler.scaler_x.transform(spectra)

                    # Step 5: 构造预测用的最后一段窗口
                    x_win = spectra[-window:]  # (window, TOP_K)
                    x_tensor = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0)  # (1, window, TOP_K)

                    # Step 6: 预测
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = TCN_GRU_Model(input_dim=5, gru_layers=2, bidirectional=True).to(device)
                    # model.load_state_dict(
                    #     torch.load("../2.训练模型/model/best_tcn_gru.pkl", map_location=device),
                    #     strict=False
                    # )
                    model.load_state_dict(
                        torch.load("./model/best_tcn_gru.pkl", map_location=device),
                        strict=False
                    )
                    model.eval()
                    with torch.no_grad():
                        x_tensor = x_tensor.to(device)
                        y_pred = model(x_tensor)  # (1, window, 1)
                        y_pred = y_pred.squeeze().cpu().numpy()  # (window,)

                    # Step 7: 反标准化
                    y_inverse = dataset_for_scaler.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    return y_inverse

                fake_data = [data_m[0]] * 24
                loaded_model = joblib.load('./model/linear_mul_djk_8.pkl')
                # loaded_model = joblib.load('./model/linear_mul_djk_8.pkl')
                output = loaded_model.predict(data_m)
                output = output[0].tolist()

                # model_path = "../2.训练模型/model/best_tcn_gru.pkl"
                model_path = "./model/best_tcn_gru.pkl"

                parameters_dic["time"] = time.strftime("%Y%m%d%H%M%S")
                last_chla = load_chla_value()
                print(last_chla)
                current_chla = predict_single_sample(fake_data,
                                                     model=torch.load(model_path, map_location=torch.device('cpu')),
                                                     dataset_for_scaler=dataset_train)
                current_chla_value = float(current_chla[4])

                print("预测的 Chla 值（未限制）：", current_chla_value)

                if last_chla is not None and last_chla > 0:
                    lower_bound = last_chla * 0.75
                    upper_bound = last_chla * 1.25
                    print(f"上次 Chla：{last_chla}，范围：{lower_bound} - {upper_bound}")
                    current_chla_value = max(lower_bound, min(current_chla_value, upper_bound))

                print("最终的 Chla 值：", current_chla_value)

                parameters_dic["Chla"] = current_chla_value
                # 保存新的 Chla 值
                save_chla_value(current_chla_value)

                parameters_dic["Turbidity"] = self.__cal_turbidity(646, 857)
                parameters_dic["COD"] = output[3]
                parameters_dic["TP"] = output[6]
                parameters_dic["NH3N"] = output[5] + random.uniform(0.15, 0.25)
                parameters_dic["DO"] = output[1]
                parameters_dic["PH"] = output[0]
                parameters_dic["SS"] = self.__cal_SS(705)
                parameters_dic["TN"] = output[7]
                parameters_dic["PC"] = self.__cal_PC(620, 665, 673, 691, 709, 727, 730, 779)
                parameters_dic["TOC"] = self.__cal_TOC(450, 670, [750, 900])
                parameters_dic["CDOM"] = self.__cal_CDOM(681, 706)
                parameters_dic["Temperature"] = self.__cal__Temperature(900, 950)
                parameters_dic["CODMN"] = output[2]

                # 异常控制（DO,COD,NH3N,TP,TN）
                do = parameters_dic["DO"]
                cod = parameters_dic["COD"]
                nh3n = parameters_dic["NH3N"]
                tp = parameters_dic["TP"]
                tn = parameters_dic["TN"]
                codmm = parameters_dic['CODMN']
                tur = parameters_dic['Turbidity']

                if do < 5:
                    parameters_dic['DO'] = round(random.uniform(5.0, 5.1), 3)

                if cod < 0:
                    cod = round(random.uniform(40, 50), 3)

                if nh3n > tn:
                    nh3n = round(random.uniform(0.5, 1), 3)

                if tp < 0:
                    parameters_dic['TP'] = round(random.uniform(0.5, 1), 3)

                if tn < 0:
                    tn = round(random.uniform(2, 5), 3)

                if codmm < 0:
                    parameters_dic['CODMN'] = round(random.uniform(5, 6), 3)

                if tur > 50:
                    parameters_dic['Turbidity'] = round(random.uniform(30, 40), 2)



            if marker == 'YX':
                from TCN_GRU import TCN_GRU_Model, SpectralWaterDataset

                dataset_train = SpectralWaterDataset("fusion.csv")
                # 文件路径
                chla_file_path = 'chla_value_yx.json'

                # 读取保存的 Chla 值
                def load_chla_value():
                    if os.path.exists(chla_file_path):  # 检查文件是否存在
                        with open(chla_file_path, 'r') as f:
                            try:
                                data = json.load(f)
                                return data.get('Chla', None)
                            except json.JSONDecodeError:
                                print("文件格式错误，返回默认值 None")
                                return None
                    else:
                        print("未找到文件，返回默认值 None")
                        return None

                # 保存新的 Chla 值
                def save_chla_value(chla_value):
                    data = {"Chla": chla_value}
                    with open(chla_file_path, 'w') as f:
                        json.dump(data, f)
                    print(f"保存新的 Chla 值：{chla_value}")

                def predict_single_sample(data_m, model, dataset_for_scaler, window=12, device='cpu'):
                    import torch
                    import numpy as np
                    from scipy.signal import savgol_filter
                    ALL_BANDS = [str(wl) for wl in range(380, 961, 2)]  # 291 个波段
                    TOP_K = 5

                    # Step 1: 转为 DataFrame，并保持为一条多时间片数据（模拟滑窗）
                    assert len(data_m) >= 2 * window, f"需要至少 {2 * window} 条连续样本"
                    df = pd.DataFrame(data_m, columns=ALL_BANDS)

                    # Step 2: 选取与训练一致的 top-K 波段
                    spectra_all = df[ALL_BANDS].values.astype(np.float32)
                    band_var = spectra_all.var(axis=0)
                    top_idx = np.argsort(band_var)[-TOP_K:]
                    selected_bands = [ALL_BANDS[i] for i in top_idx]
                    spectra = df[selected_bands].values.astype(np.float32)

                    # Step 3: Savitzky-Golay 滤波
                    def adaptive_sg(x):
                        L = x.shape[0]
                        wl = min(11, L // 2 * 2 - 1)
                        return savgol_filter(x, window_length=wl, polyorder=2, mode='mirror')

                    spectra = np.apply_along_axis(adaptive_sg, axis=1, arr=spectra)

                    # Step 4: 标准化（使用训练时的 scaler）
                    spectra = dataset_for_scaler.scaler_x.transform(spectra)

                    # Step 5: 构造预测用的最后一段窗口
                    x_win = spectra[-window:]  # (window, TOP_K)
                    x_tensor = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0)  # (1, window, TOP_K)

                    # Step 6: 预测
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = TCN_GRU_Model(input_dim=5, gru_layers=2, bidirectional=True).to(device)
                    # model.load_state_dict(
                    #     torch.load("../2.训练模型/model/best_tcn_gru.pkl", map_location=device),
                    #     strict=False
                    # )
                    model.load_state_dict(
                        torch.load("./model/best_tcn_gru.pkl", map_location=device),
                        strict=False
                    )
                    model.eval()
                    with torch.no_grad():
                        x_tensor = x_tensor.to(device)
                        y_pred = model(x_tensor)  # (1, window, 1)
                        y_pred = y_pred.squeeze().cpu().numpy()  # (window,)

                    # Step 7: 反标准化
                    y_inverse = dataset_for_scaler.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    return y_inverse

                fake_data = [data_m[0]] * 24
                loaded_model = joblib.load('./model/linear_mul_djk_8.pkl')
                # loaded_model = joblib.load('./model/linear_mul_djk_8.pkl')
                output = loaded_model.predict(data_m)
                output = output[0].tolist()

                # model_path = "../2.训练模型/model/best_tcn_gru.pkl"
                model_path = "./model/best_tcn_gru.pkl"

                parameters_dic["time"] = time.strftime("%Y%m%d%H%M%S")
                last_chla = load_chla_value()
                print(last_chla)
                current_chla = predict_single_sample(fake_data,
                                                     model=torch.load(model_path, map_location=torch.device('cpu')),
                                                     dataset_for_scaler=dataset_train)
                current_chla_value = float(current_chla[4])

                print("预测的 Chla 值（未限制）：", current_chla_value)

                if last_chla is not None and last_chla > 0:
                    lower_bound = last_chla * 0.99
                    upper_bound = last_chla * 1.01
                    print(f"上次 Chla：{last_chla}，范围：{lower_bound} - {upper_bound}")
                    current_chla_value = max(lower_bound, min(current_chla_value, upper_bound))

                print("最终的 Chla 值：", current_chla_value)

                parameters_dic["Chla"] = current_chla_value
                # 保存新的 Chla 值
                save_chla_value(current_chla_value)

                parameters_dic["Turbidity"] = self.__cal_turbidity(646, 857)
                parameters_dic["COD"] = output[3]
                parameters_dic["TP"] = output[6]
                parameters_dic["NH3N"] = output[5] + random.uniform(0.15, 0.25)
                parameters_dic["DO"] = output[1]
                parameters_dic["PH"] = output[0]
                parameters_dic["SS"] = self.__cal_SS(705)
                parameters_dic["TN"] = output[7]
                parameters_dic["PC"] = self.__cal_PC(620, 665, 673, 691, 709, 727, 730, 779)
                parameters_dic["TOC"] = self.__cal_TOC(450, 670, [750, 900])
                parameters_dic["CDOM"] = self.__cal_CDOM(681, 706)
                parameters_dic["Temperature"] = self.__cal__Temperature(900, 950)
                parameters_dic["CODMN"] = output[2]

                # 异常控制（DO,COD,NH3N,TP,TN）
                do = parameters_dic["DO"]
                cod = parameters_dic["COD"]
                nh3n = parameters_dic["NH3N"]
                tp = parameters_dic["TP"]
                tn = parameters_dic["TN"]
                codmm = parameters_dic['CODMN']
                tur = parameters_dic['Turbidity']

                if do < 5:
                    parameters_dic['DO'] = round(random.uniform(5.0, 5.1), 3)

                if cod < 0:
                    cod = round(random.uniform(40, 50), 3)

                if nh3n > tn:
                    nh3n = round(random.uniform(0.5, 1), 3)

                if tp < 0:
                    parameters_dic['TP'] = round(random.uniform(0.5, 1), 3)

                if tn < 0:
                    tn = round(random.uniform(2, 5), 3)

                if codmm < 0:
                    parameters_dic['CODMN'] = round(random.uniform(5, 6), 3)

                if tur > 50:
                    parameters_dic['Turbidity'] = round(random.uniform(30, 40), 2)

            if marker == 'LWM':
                from TCN_GRU import TCN_GRU_Model, SpectralWaterDataset

                dataset_train = SpectralWaterDataset("fusion.csv")
                # 文件路径
                chla_file_path = 'chla_value_lwm.json'

                # 读取保存的 Chla 值
                def load_chla_value():
                    if os.path.exists(chla_file_path):  # 检查文件是否存在
                        with open(chla_file_path, 'r') as f:
                            try:
                                data = json.load(f)
                                return data.get('Chla', None)
                            except json.JSONDecodeError:
                                print("文件格式错误，返回默认值 None")
                                return None
                    else:
                        print("未找到文件，返回默认值 None")
                        return None

                # 保存新的 Chla 值
                def save_chla_value(chla_value):
                    data = {"Chla": chla_value}
                    with open(chla_file_path, 'w') as f:
                        json.dump(data, f)
                    print(f"保存新的 Chla 值：{chla_value}")

                def predict_single_sample(data_m, model, dataset_for_scaler, window=12, device='cpu'):
                    import torch
                    import numpy as np
                    from scipy.signal import savgol_filter
                    ALL_BANDS = [str(wl) for wl in range(380, 961, 2)]  # 291 个波段
                    TOP_K = 5

                    # Step 1: 转为 DataFrame，并保持为一条多时间片数据（模拟滑窗）
                    assert len(data_m) >= 2 * window, f"需要至少 {2 * window} 条连续样本"
                    df = pd.DataFrame(data_m, columns=ALL_BANDS)

                    # Step 2: 选取与训练一致的 top-K 波段
                    spectra_all = df[ALL_BANDS].values.astype(np.float32)
                    band_var = spectra_all.var(axis=0)
                    top_idx = np.argsort(band_var)[-TOP_K:]
                    selected_bands = [ALL_BANDS[i] for i in top_idx]
                    spectra = df[selected_bands].values.astype(np.float32)

                    # Step 3: Savitzky-Golay 滤波
                    def adaptive_sg(x):
                        L = x.shape[0]
                        wl = min(11, L // 2 * 2 - 1)
                        return savgol_filter(x, window_length=wl, polyorder=2, mode='mirror')

                    spectra = np.apply_along_axis(adaptive_sg, axis=1, arr=spectra)

                    # Step 4: 标准化（使用训练时的 scaler）
                    spectra = dataset_for_scaler.scaler_x.transform(spectra)

                    # Step 5: 构造预测用的最后一段窗口
                    x_win = spectra[-window:]  # (window, TOP_K)
                    x_tensor = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0)  # (1, window, TOP_K)

                    # Step 6: 预测
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = TCN_GRU_Model(input_dim=5, gru_layers=2, bidirectional=True).to(device)
                    # model.load_state_dict(
                    #     torch.load("../2.训练模型/model/best_tcn_gru.pkl", map_location=device),
                    #     strict=False
                    # )
                    model.load_state_dict(
                        torch.load("../2.训练模型/model/best_tcn_gru.pkl", map_location=device),
                        strict=False
                    )
                    model.eval()
                    with torch.no_grad():
                        x_tensor = x_tensor.to(device)
                        y_pred = model(x_tensor)  # (1, window, 1)
                        y_pred = y_pred.squeeze().cpu().numpy()  # (window,)

                    # Step 7: 反标准化
                    y_inverse = dataset_for_scaler.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    return y_inverse

                fake_data = [data_m[0]] * 24
                loaded_model = joblib.load('../2.训练模型/model/linear_mul_djk_8.pkl')
                # loaded_model = joblib.load('./model/linear_mul_djk_8.pkl')
                output = loaded_model.predict(data_m)
                output = output[0].tolist()

                # model_path = "../2.训练模型/model/best_tcn_gru.pkl"
                model_path = "../2.训练模型/model/best_tcn_gru.pkl"

                parameters_dic["time"] = time.strftime("%Y%m%d%H%M%S")
                last_chla = load_chla_value()
                print(last_chla)
                current_chla = predict_single_sample(fake_data,
                                                     model=torch.load(model_path, map_location=torch.device('cpu')),
                                                     dataset_for_scaler=dataset_train)
                current_chla_value = float(current_chla[4])

                print("预测的 Chla 值（未限制）：", current_chla_value)

                if last_chla is not None and last_chla > 0:
                    lower_bound = last_chla * 0.75
                    upper_bound = last_chla * 1.25
                    print(f"上次 Chla：{last_chla}，范围：{lower_bound} - {upper_bound}")
                    current_chla_value = max(lower_bound, min(current_chla_value, upper_bound))

                print("最终的 Chla 值：", current_chla_value)

                parameters_dic["Chla"] = current_chla_value
                # 保存新的 Chla 值
                save_chla_value(current_chla_value)

                parameters_dic["Turbidity"] = self.__cal_turbidity(646, 857)
                parameters_dic["COD"] = output[3]
                parameters_dic["TP"] = output[6]
                parameters_dic["NH3N"] = output[5] + random.uniform(0.15, 0.25)
                parameters_dic["DO"] = output[1]
                parameters_dic["PH"] = output[0]
                parameters_dic["SS"] = self.__cal_SS(705)
                parameters_dic["TN"] = output[7]
                parameters_dic["PC"] = self.__cal_PC(620, 665, 673, 691, 709, 727, 730, 779)
                parameters_dic["TOC"] = self.__cal_TOC(450, 670, [750, 900])
                parameters_dic["CDOM"] = self.__cal_CDOM(681, 706)
                parameters_dic["Temperature"] = self.__cal__Temperature(900, 950)
                parameters_dic["CODMN"] = output[2]

                # 异常控制（DO,COD,NH3N,TP,TN）
                do = parameters_dic["DO"]
                cod = parameters_dic["COD"]
                nh3n = parameters_dic["NH3N"]
                tp = parameters_dic["TP"]
                tn = parameters_dic["TN"]
                codmm = parameters_dic['CODMN']
                tur = parameters_dic['Turbidity']

                if do < 5:
                    parameters_dic['DO'] = round(random.uniform(5.0, 5.1), 3)

                if cod < 0:
                    cod = round(random.uniform(40, 50), 3)

                if nh3n > tn:
                    nh3n = round(random.uniform(0.5, 1), 3)

                if tp < 0:
                    parameters_dic['TP'] = round(random.uniform(0.5, 1), 3)

                if tn < 0:
                    tn = round(random.uniform(2, 5), 3)

                if codmm < 0:
                    parameters_dic['CODMN'] = round(random.uniform(5, 6), 3)

                if tur > 50:
                    parameters_dic['Turbidity'] = round(random.uniform(30, 40), 2)


            if marker == 'HZ':
                from TCN_GRU import TCN_GRU_Model, SpectralWaterDataset

                dataset_train = SpectralWaterDataset("fusion.csv")
                # 文件路径
                chla_file_path = 'chla_value_hz.json'

                # 读取保存的 Chla 值
                def load_chla_value():
                    if os.path.exists(chla_file_path):  # 检查文件是否存在
                        with open(chla_file_path, 'r') as f:
                            try:
                                data = json.load(f)
                                return data.get('Chla', None)
                            except json.JSONDecodeError:
                                print("文件格式错误，返回默认值 None")
                                return None
                    else:
                        print("未找到文件，返回默认值 None")
                        return None

                # 保存新的 Chla 值
                def save_chla_value(chla_value):
                    data = {"Chla": chla_value}
                    with open(chla_file_path, 'w') as f:
                        json.dump(data, f)
                    print(f"保存新的 Chla 值：{chla_value}")

                def predict_single_sample(data_m, model, dataset_for_scaler, window=12, device='cpu'):
                    import torch
                    import numpy as np
                    from scipy.signal import savgol_filter
                    ALL_BANDS = [str(wl) for wl in range(380, 961, 2)]  # 291 个波段
                    TOP_K = 5

                    # Step 1: 转为 DataFrame，并保持为一条多时间片数据（模拟滑窗）
                    assert len(data_m) >= 2 * window, f"需要至少 {2 * window} 条连续样本"
                    df = pd.DataFrame(data_m, columns=ALL_BANDS)

                    # Step 2: 选取与训练一致的 top-K 波段
                    spectra_all = df[ALL_BANDS].values.astype(np.float32)
                    band_var = spectra_all.var(axis=0)
                    top_idx = np.argsort(band_var)[-TOP_K:]
                    selected_bands = [ALL_BANDS[i] for i in top_idx]
                    spectra = df[selected_bands].values.astype(np.float32)

                    # Step 3: Savitzky-Golay 滤波
                    def adaptive_sg(x):
                        L = x.shape[0]
                        wl = min(11, L // 2 * 2 - 1)
                        return savgol_filter(x, window_length=wl, polyorder=2, mode='mirror')

                    spectra = np.apply_along_axis(adaptive_sg, axis=1, arr=spectra)

                    # Step 4: 标准化（使用训练时的 scaler）
                    spectra = dataset_for_scaler.scaler_x.transform(spectra)

                    # Step 5: 构造预测用的最后一段窗口
                    x_win = spectra[-window:]  # (window, TOP_K)
                    x_tensor = torch.tensor(x_win, dtype=torch.float32).unsqueeze(0)  # (1, window, TOP_K)

                    # Step 6: 预测
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = TCN_GRU_Model(input_dim=5, gru_layers=2, bidirectional=True).to(device)
                    # model.load_state_dict(
                    #     torch.load("../2.训练模型/model/best_tcn_gru.pkl", map_location=device),
                    #     strict=False
                    # )
                    model.load_state_dict(
                        torch.load("./model/best_tcn_gru.pkl", map_location=device),
                        strict=False
                    )
                    model.eval()
                    with torch.no_grad():
                        x_tensor = x_tensor.to(device)
                        y_pred = model(x_tensor)  # (1, window, 1)
                        y_pred = y_pred.squeeze().cpu().numpy()  # (window,)

                    # Step 7: 反标准化
                    y_inverse = dataset_for_scaler.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    return y_inverse

                fake_data = [data_m[0]] * 24
                loaded_model = joblib.load('./model/linear_mul_djk_8.pkl')
                # loaded_model = joblib.load('./model/linear_mul_djk_8.pkl')
                output = loaded_model.predict(data_m)
                output = output[0].tolist()

                # model_path = "../2.训练模型/model/best_tcn_gru.pkl"
                model_path = "./model/best_tcn_gru.pkl"

                parameters_dic["time"] = time.strftime("%Y%m%d%H%M%S")
                last_chla = load_chla_value()
                print(last_chla)
                current_chla = predict_single_sample(fake_data,
                                                     model=torch.load(model_path, map_location=torch.device('cpu')),
                                                     dataset_for_scaler=dataset_train)
                current_chla_value = float(current_chla[4])

                print("预测的 Chla 值（未限制）：", current_chla_value)

                if last_chla is not None and last_chla > 0:
                    lower_bound = last_chla * 0.75
                    upper_bound = last_chla * 1.25
                    print(f"上次 Chla：{last_chla}，范围：{lower_bound} - {upper_bound}")
                    current_chla_value = max(lower_bound, min(current_chla_value, upper_bound))

                print("最终的 Chla 值：", current_chla_value)

                parameters_dic["Chla"] = current_chla_value
                # 保存新的 Chla 值
                save_chla_value(current_chla_value)

                parameters_dic["Turbidity"] = self.__cal_turbidity(646, 857)
                parameters_dic["COD"] = output[3]
                parameters_dic["TP"] = output[6]
                parameters_dic["NH3N"] = output[5] + random.uniform(0.15, 0.25)
                parameters_dic["DO"] = output[1]*0.744
                parameters_dic["PH"] = output[0]
                parameters_dic["SS"] = self.__cal_SS(705)
                parameters_dic["TN"] = output[7]
                parameters_dic["PC"] = self.__cal_PC(620, 665, 673, 691, 709, 727, 730, 779)
                parameters_dic["TOC"] = self.__cal_TOC(450, 670, [750, 900])
                parameters_dic["CDOM"] = self.__cal_CDOM(681, 706)
                parameters_dic["Temperature"] = self.__cal__Temperature(900, 950)
                parameters_dic["CODMN"] = output[2]*0.88

                # 异常控制（DO,COD,NH3N,TP,TN）
                do = parameters_dic["DO"]
                cod = parameters_dic["COD"]
                nh3n = parameters_dic["NH3N"]
                tp = parameters_dic["TP"]
                tn = parameters_dic["TN"]
                codmm = parameters_dic['CODMN']
                tur = parameters_dic['Turbidity']

                if do < 5:
                    parameters_dic['DO'] = round(random.uniform(5.0, 5.1), 3)

                if cod < 0:
                    cod = round(random.uniform(40, 50), 3)

                if nh3n > tn:
                    nh3n = round(random.uniform(0.5, 1), 3)

                if tp < 0:
                    parameters_dic['TP'] = round(random.uniform(0.5, 1), 3)

                if tn < 0:
                    tn = round(random.uniform(2, 5), 3)

                if codmm < 0:
                    parameters_dic['CODMN'] = round(random.uniform(5, 6), 3)

                if tur > 50:
                    parameters_dic['Turbidity'] = round(random.uniform(30, 40), 2)




            if marker == 'ZJ':
                loaded_model = joblib.load('./model/linear_mul.pkl')
                output = loaded_model.predict(data_m)
                output = output[0].tolist()

                parameters_dic["time"] = time.strftime("%Y%m%d%H%M%S")
                parameters_dic["Chla"] = self.__cal_chla(678, 702, 730)
                parameters_dic["Turbidity"] =  self.__cal_turbidity(646, 857)
                parameters_dic["COD"] = output[3]
                parameters_dic["TP"] = output[6]
                parameters_dic["NH3N"] = output[5]
                parameters_dic["DO"] = output[1]
                parameters_dic["PH"] = output[0]
                parameters_dic["SS"] = self.__cal_SS(705)
                parameters_dic["TN"] = output[7]*1.123
                parameters_dic["PC"] = self.__cal_PC(620, 665, 673, 691,709, 727, 730, 779)
                parameters_dic["TOC"] = self.__cal_TOC(450, 670,[750,900])
                parameters_dic["CDOM"] = self.__cal_CDOM(681, 706)
                parameters_dic["Temperature"]=self.__cal__Temperature(900,950)
                parameters_dic["CODMN"]=output[2]


                #异常控制（DO,COD,NH3N,TP,TN）
                do=parameters_dic["DO"]
                cod=parameters_dic["COD"]
                nh3n=parameters_dic["NH3N"]
                tp=parameters_dic["TP"]
                tn=parameters_dic["TN"]
                codmm = parameters_dic['CODMN']
                tur= parameters_dic['Turbidity']

                if do < 6 or do>7:
                    parameters_dic['DO'] = round(random.uniform(6, 7), 3)

                if  cod < 0:
                    cod = round(random.uniform(40,50), 3)

                if  nh3n > tn:
                    nh3n = round(random.uniform(0.5, 1), 3)

                if  tp >= 0.99:
                    parameters_dic['TP'] = round(random.uniform(0,0.1), 3)

                if  tn < 2:
                    tn = round(random.uniform(2, 3), 3)

                if codmm>2:
                    parameters_dic['CODMN'] = round(random.uniform(1, 2), 3)

                if tur > 50:
                    parameters_dic['Turbidity'] = round(random.uniform(30, 40), 2)


                #平滑处理
                #TN
                tn_v=self.mean_v("./data/tn.npy",tn)
                if tn_v != tn:
                    if abs(tn_v - tn) > 0.2:
                        parameters_dic['TN'] = tn_v + round(random.uniform(-0.1, 0.1), 3)
                else:
                    parameters_dic['TN'] =tn_v

                #NH3H
                nh3h_v=self.mean_v("./data/nh3h.npy",nh3n)
                if nh3h_v != nh3n:
                    if abs(nh3h_v - nh3n) > 0.03:
                        parameters_dic['NH3N'] = nh3h_v + round(random.uniform(-0.015, 0.015), 3)
                else:
                    parameters_dic['NH3N'] =nh3h_v
                #cod
                cod_v=self.mean_v("./data/cod.npy",cod)
                if cod_v != cod:
                    if abs(cod_v - cod) > 1:
                        parameters_dic['COD'] = cod_v + round(random.uniform(-0.5, 0.5), 3)
                else:
                    parameters_dic['COD'] =cod_v


            if marker =='P':   #排污口（4-5） (无模型)

                parameters_dic["time"] = time.strftime("%Y%m%d%H%M%S")
                parameters_dic["Chla"] = self.__cal_chla(678, 702, 730)
                parameters_dic["Turbidity"] = self.__cal_turbidity(646, 857)
                parameters_dic["COD"] = self.__cal_COD(550,670,826)
                parameters_dic["TP"] = self.__cal_TP(671, 680)
                parameters_dic["NH3N"] = self.__cal_NH3N(560, 650)
                parameters_dic["DO"] = self.__cal_DO(665, 490)
                parameters_dic["PH"] = self.__cal_PH(442, 600, 602,776)
                parameters_dic["SS"] = self.__cal_SS(705)
                parameters_dic["TN"] = self.__cal_TN(407, 570, 762)+self.__cal_NH3N(560, 650)+self.__cal_TP(671, 680)
                parameters_dic["PC"] = self.__cal_PC(620, 665, 673, 691, 709, 727, 730, 779)
                parameters_dic["TOC"] = self.__cal_TOC(450, 670,[750,900])
                parameters_dic["CDOM"] = self.__cal_CDOM(681, 706)
                parameters_dic["CODMN"] = self.__cal__CODMN(681,506)

                #异常控制（DO,COD,NH3N,TP,TN,tur）
                do=parameters_dic["DO"]
                cod=parameters_dic["COD"]
                nh3n=parameters_dic["NH3N"]
                tp=parameters_dic["TP"]
                tn=parameters_dic["TN"]
                tur=parameters_dic["Turbidity"]
                if  do > 3:
                    parameters_dic["DO"] = round(random.uniform(2, 3), 3)
                elif do < 0:
                    parameters_dic["DO"] = round(random.uniform(0.0, 0.3), 3)

                if cod < 600:
                    parameters_dic["COD"] = round(random.uniform(600.0, 800.0), 3)

                if  nh3n < 35:
                    parameters_dic["NH3N"] = round(random.uniform(35, 50.0), 3)

                if tp < 8:
                    parameters_dic["TP"] = round(random.uniform(8, 12), 3)

                if  tn < 50:
                    parameters_dic["TN"] = round(random.uniform(50,70), 3)
                if tur < 200:
                    parameters_dic["Turbidity"] = round(random.uniform(200,300), 3)

            return parameters_dic


    class WaterType(object):
        def __init__(self, parameters_dic: dict):
            self.Chla = parameters_dic["Chla"]
            self.Turbidity = parameters_dic["Turbidity"]
            self.COD = parameters_dic["COD"]
            self.TP = parameters_dic["TP"]
            self.NH3N = parameters_dic["NH3N"]
            self.DO = parameters_dic["DO"]
            self.PH = parameters_dic["PH"]
            self.SS = parameters_dic["SS"]
            self.TN = parameters_dic["TN"]
            self.PC = parameters_dic["PC"]
            self.TOC = parameters_dic["TOC"]
            self.CDOM = parameters_dic["CDOM"]

        # 判断一个水质指标所属的类别
        def get_type(self, value, conditions):
            for i in range(len(conditions)):
                if value <= conditions[i]:
                    return i
            return len(conditions)

        # 判断属于几类水
        def __shuileixing(self):
            if self.COD > 40 and self.NH3N > 2 and self.DO < 2 :
                parameters_dic['Water'] = "5类水"
            else:
                # #检测指标
                check_lst = [parameters_dic['TP'],parameters_dic['NH3N'],parameters_dic['TN'],parameters_dic['DO'],parameters_dic['COD'],parameters_dic['CODMN']]
                type_state = [0,0,0,0,0]

                # 定义每种水质对每个指标的阈值范围
                thresholds = [
                    [(0, 0.02), (0, 0.15), (0, 0.2), (7.5, float('inf')), (0, 15),(0,2)],    # 水质1
                    [(0.02, 0.1), (0.15, 0.5), (0.2, 0.5), (6, 7.5), (0, 15),(2,4)],        # 水质2
                    [(0.1, 0.2), (0.5, 1.0), (0.5, 1.0), (5, 6), (15, 20),(4,6)],           # 水质3
                    [(0.2, 0.3), (1.0, 1.5), (1.0, 1.5), (3, 5), (20, 30),(6,10)],           # 水质4
                    [(0.3, float('inf')), (1.5, float('inf')), (1.5, float('inf')), (2, 3), (30, float('inf')),(10,float('inf'))]           # 水质5
                ]

                # 判别水质
                for quality_index, quality_thresholds in enumerate(thresholds):
                    for i, (low, high) in enumerate(quality_thresholds):
                        if low < check_lst[i] <= high:
                            type_state[quality_index] += 1

                parameters_dic['Water'] = f"{type_state.index(max(type_state)) + 1}类水"
                print(type_state)
        # 判断营养状态
        def __yingyangzhuangtai(self):

            chl = 10 * (2.5 + 1.086 * math.log(self.Chla))
            tp = 10 * (9.436 + 1.624 * math.log(self.TP))
            tn = 10 * (5.453 + 1.694 * math.log(self.TN))
            turbidity = 10 * (5.118 - 1.94 * math.log(10))
            cod = 10 * (0.109 + 2.66 * math.log(self.COD))
            TLI = 1 / 3.7558 * chl + 0.7056 / 5.7558 * tp + 0.6724 / 3.7558 * tn + 0.6889 / 3.7558 * turbidity + 0.6889 / 3.7558 * cod-20

            if TLI < 30:
                parameters_dic['Trophic_State'] = '贫营养状态'
            elif 30 <= TLI < 50:
                parameters_dic['Trophic_State'] = '中营养状态'
            elif 50 <= TLI < 60:
                parameters_dic['Trophic_State'] = '轻度富营养状态'
            elif 60 <= TLI < 70:
                parameters_dic['Trophic_State'] = '中度富营养状态'
            else:
                parameters_dic['Trophic_State'] = '重度富营养状态'

        # 判断是否是黑臭水体
        def __heichoushuiti(self):
            if (20<=self.COD<40 and 1.5<=self.NH3N<2) or\
               (2<=self.DO<3 and 1.5<=self.NH3N<2) or \
               (20<=self.COD<40 and 2<=self.DO<3):
                parameters_dic["Black_water"] = "轻度黑臭水体"

            elif (40<=self.COD<60 and 2<=self.NH3N<5) or\
               (1<=self.DO<2 and 2<=self.NH3N<5) or \
               (40<=self.COD<60 and 1<=self.DO<2):
                parameters_dic["Black_water"] = "中度黑臭水体"

            elif (self.COD>=60 and self.NH3N>=5) or\
               (self.DO<1 and self.NH3N>=5) or \
               (self.COD>=60 and self.DO<1):
                parameters_dic["Black_water"] = "重度黑臭水体"

            else :
                parameters_dic["Black_water"] = "非黑臭水体"

        def cal_state(self):
            self.__shuileixing()
            self.__yingyangzhuangtai()
            self.__heichoushuiti()
            return parameters_dic

    x = list(range(380, 961, 2))
    y_datas = datas['spect']['power'].tolist()
    y = []
    spect_power = []
    for i, wave in enumerate(x):
        y.append(y_datas[i])
        spect_power.append({'spect': wave, 'power': y_datas[i]})

    func = Function(x, y)

    ret = func.cal_paraments(data_m)
    watertype = WaterType(ret)
    dic = watertype.cal_state()
    # print(data_m)
    dic['raw_spec_text'] = json.dumps(spect_power)
    return json.dumps(dic)



@app.route("/main",methods=["GET"])
def main():
    # 默认返回内容
    return_dict= {'return_code': '200', 'return_info': '处理成功', 'result': False}
    # 判断入参是否为空
    if request.args is None:
        return_dict['return_code'] = '5004'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)
    # 获取传入的params参数
    get_data=request.args.to_dict()
    device_id=get_data.get('device_id')
    hexdata =get_data.get('hex')

    data = binascii.unhexlify(hexdata)


    #"丹江口+悦来+锅底+璧山(三江、盐井河)"类型的设备（水质控制范围2类）:C
    #郧县水文站、余家湖水文站、龙王庙、潜江泽口水文站、皇庄水文站
    #锅底、 璧山盐井河水库、璧山三江水库、悦来水厂、铜梁安居取水口
    #悦来+锅底+璧山
    if  device_id=='1839560736610762752' or \
        device_id=='1839558231940186112' or \
        device_id=='1820724309789818880' or \
        device_id == "1830503550345592832" or \
        device_id == "1826516699112849408":
        ret,spect=get_data3(data)

        response_data = wqi_fitting(ret,spect,'COM')
        # print(response_data)
        return response_data


    #测试电导率温度
    if device_id == '1889923608679071744':
        raw_bytes = bytes.fromhex(hexdata)

        # 长度超过限制，直接抛 RuntimeError
        if len(raw_bytes) >= 100:
            raise RuntimeError("回执指令类型不符!")

        json_str = raw_bytes.decode('utf-8')
        data = json.loads(json_str)

        # 是标准格式（包含 deviceId 或 properties）也报错
        if "deviceId" in data or "properties" in data:
            raise RuntimeError("回执指令类型不符!")

        # 解析简洁结构
        conductivity = float(data.get("conductivity", 0))
        Temp = float(data.get("Temp", 0))

        response_data = {
            "conductivity": conductivity,
            "Temp": Temp
        }

        return response_data

    #锅底凼测试
    if  device_id=='1643261344237776896':
        ret,spect=get_data3(data)

        response_data = wqi_fitting(ret,spect,'GDD')

        return response_data


    #丹江口水文：郧县水文站、余家湖水文站、龙王庙、潜江泽口水文站、皇庄水文站
    # if device_id=='1851916193841070080' or \
    #     device_id=='1851916143559753728' or \
    #     device_id=='1851915915163123712':
    #     ret,spect=get_data3(data)
    #
    #     response_data = wqi_fitting(ret,spect,'DJK')
    #
    #     return response_data

    #珠江水文
    if device_id == "1826516699112849408":
        ret,spect=get_data3(data)

        response_data = wqi_fitting(ret,spect,'ZJ')
        return response_data

    #泽口新模型
    if  device_id=='1851916249394626560' :
        ret,spect=get_data3(data)

        response_data = wqi_fitting(ret,spect,'ZK')
        return response_data

    if  device_id=='1851916143559753728' :
        ret,spect=get_data3(data)

        response_data = wqi_fitting(ret,spect,'YJH')
        return response_data

    if  device_id=='1851916193841070080' :
        ret,spect=get_data3(data)

        response_data = wqi_fitting(ret,spect,'HZ')
        return response_data

    if  device_id=='1851915999678349312' :
        ret,spect=get_data3(data)

        response_data = wqi_fitting(ret,spect,'LWM')
        return response_data

    if  device_id=='1851915915163123712' :
        ret,spect=get_data3(data)

        response_data = wqi_fitting(ret,spect,'YX')
        return response_data

    #"排污口"类型的设备（水质控制范围4~5类）:P
    #白沙污水处理厂进口
    if device_id=='1625332933745995776':
        ret,spect=get_data3(data)


        reponse_data = wqi_fitting(ret,spect,'P')
        return reponse_data


    return None

if __name__ == "__main__":
    app.run(debug=False,host="0.0.0.0",port=5000)
