# 파일이름 : mlops_kafka_exam.py
# 코드설명 : 1.1.1.1. 현장(화인테크) 적용을 위한 인터페이스와 예측 모델 간의 통신 모듈
# 입/출력 : 통신 메시지 (예시 DATA 포함)
# 유의 사항 : 
# 1. Nan 값이 JSON에서 허용되지 않음, 임의의 문자열 ‘nan’로 변경하여 전달 후, 내부(mlops_make_mlflow_model.py)에서 nan 값으로 변경 후 예측 수행
# 2. 입력하는 데이터 첫 번째 행에 문자열 값이 존재하면 오류 발생 -> 임의의 실수값으로 구성된 첫 행을 추가 후, 내부에서 첫 행을 제거 후 예측 수행
# 최종수정 : 2023년 11월 24일
# 제 작 자 : 김학성 (hakseong@micube.co.kr) {전체 개발}, 홍민성 (mshong@micube.co.kr) {메시지의 데이터 양식 수정 및 코드 테스트}
# Copyright : MICUBE Solution, Inc. 

from kafka3 import KafkaConsumer, KafkaProducer
from multiprocessing import Process
import json, time
import datetime

class MlopsKafkaExam:
    def __init__(self):
        # Kafka 서버 설정
        self.kafka_host = '192.168.40.94'  # Kafka 브로커의 주소
        self.kafka_port = '9093'
        self.topic_request = 'MLOPS_MODEL'
        self.topic_response = 'MLOPS_LEVELING'
        self.consumer = None
        self.producer = None

        print('host:', self.kafka_host)
        print('port:', self.kafka_port)
        print('topic_request:', self.topic_request)
        print('topic_response:', self.topic_response)

    def init_kafka_consumer(self):
        print('init_kafka_consumer')
        while True:
            try:
                self.consumer = KafkaConsumer(self.topic_response, bootstrap_servers=f"{self.kafka_host}:{self.kafka_port}", auto_offset_reset="latest", enable_auto_commit=True)
                # consumer_timeout_ms=10000)
                break
            except:
                print('time.sleep')
                time.sleep(30)

    def run_consumer(self):
        self.init_kafka_consumer()
        while True:
            print("ResultConsumer()")
            for message in self.consumer:
                print('test5')
                message: ConsumerRecord
                print('test6')
                resp_dict = json.loads(message.value.decode())                
                print('message : {}'.format(resp_dict))
                for i in range(5):
                    print('*' * 100)
                for k, v in resp_dict.items():
                    print(k, ':', v)
                print('\nend of message!')


    def init_kafka_producer(self):
        while True:
            print("ResultProducer()")
            try:
                print('test1')
                self.producer = KafkaProducer(bootstrap_servers=[f"{self.kafka_host}:{self.kafka_port}"])
                print('test2')
                break
            except Exception as e:
                print(e)
                time.sleep(30)

    def run_producer(self):
        self.init_kafka_producer()
        try:
            data: dict
            message = {
                "ITEM_CODE": '84711G6900WK',
                "MC_CODE": 'INJ_MAIN_01',
                "DATA_ID": '301670',
                "MODEL_ID": '20231006_test',
                "MODEL_NAME": '20231006_test_1',
                "MODEL_CATEGORY": 'CLASSIFICATION',
                "EVENTTIME": datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f"),
                "DATA":
                    # [
                    #     # {'INSP_TIME': 1.6, 'INSP_TEMP': 0.4, 'RPM1': 0.1, 'RPM2': 0.0}
                    #     # {'INSP_TIME': 1.6, 'INSP_TEMP': 0.4, 'RPM1': 'CV1', 'RPM2': 'CV1'}
                    #     {'INSP_TIME': 8.0, 'INSP_TEMP': 180.0, 'RPM1': 99999, 'RPM2': 99999, 'RPM3': 99999, 'RPM4': 99999, 'RPM5': 99999, 'RPM6': 99999, 'RPM7': 99999, 'RPM8': 99999, 'RPM9': 99999, 'RPM10': 99999, 'RPM11': 99999, 'RPM12': 99999, 'RPM13': 99999, 'RPM14': 99999, 'RPM15': 99999, 'RPM16': 99999, 'RPM17': 99999, 'RPM18': 99999, 'RPM19': 99999, 'RPM20': 99999, 'RPM21': 99999, 'RPM22': 99999, 'RPM23': 99999, 'TEMP1': 99999, 'TEMP2': 99999, 'TEMP3': 99999, 'TEMP4': 99999, 'TEMP5': 99999, 'TEMP6': 99999, 'TEMP7': 99999, 'TEMP8': 99999, 'TEMP9': 99999, 'TEMP10': 99999, 'TEMP11': 99999, 'TEMP12': 99999, 'TEMP13': 99999, 'TEMP14': 99999, 'TEMP15': 99999, 'TEMP16': 99999, 'TEMP17': 99999, 'TEMP18': 99999, 'TEMP19': 99999, 'TEMP20': 99999, 'TEMP21': 99999, 'TEMP22': 99999, 'TEMP23': 99999, 'TIME1': 99999, 'TIME2': 99999, 'TIME3': 99999, 'TIME4': 99999, 'TIME5': 99999, 'TIME6': 99999, 'TIME7': 99999, 'TIME8': 99999, 'TIME9': 99999, 'TIME10': 99999, 'TIME11': 99999, 'TIME12': 99999, 'TIME13': 99999, 'TIME14': 99999, 'TIME15': 99999, 'TIME16': 99999, 'TIME17': 99999, 'TIME18': 99999, 'TIME19': 99999, 'TIME20': 99999, 'TIME21': 99999, 'TIME22': 99999, 'TIME23': 99999, 'JUK1': 99999, 'JUK2': 99999, 'JUK3': 99999, 'JUK4': 99999, 'JUK5': 99999, 'JUK6': 99999, 'JUK7': 99999, 'JUK8': 99999, 'JUK9': 99999, 'JUK10': 99999, 'JUK11': 99999, 'JUK12': 99999, 'JUK13': 99999, 'JUK14': 99999, 'JUK15': 99999, 'JUK16': 99999, 'JUK17': 99999, 'JUK18': 99999, 'JUK19': 99999, 'JUK20': 99999, 'JUK21': 99999, 'JUK22': 99999, 'JUK23': 99999, 'JRCODE1': 'EE011A', 'JRCODE2': 'EE001A', 'JRCODE3': 'CB003A', 'JRCODE4': 'CB001C', 'JRCODE5': 'CC001A', 'JRCODE6': 'CC012B', 'JRCODE7': 'CD005A', 'JRCODE8': 'NN550A', 'JRCODE9': 'OP004A', 'JRCODE10': 99999, 'JRCODE11': 'CV013A', 'JRCODE12': 'CV011A', 'JRCODE13': 99999, 'JRCODE14': 99999, 'JRCODE15': 99999, 'JRCODE16': 99999, 'JRCODE17': 99999, 'JRCODE18': 99999, 'JRCODE19': 99999, 'JRCODE20': 99999, 'JRCODE21': 99999, 'JRCODE22': 99999, 'JRCODE23': 99999, 'JRCODE24': 99999, 'JRCODE25': 99999, 'PHR1': 36.477331943720685, 'PHR2': 15.633142261594578, 'PHR3': 2.605523710265764, 'PHR4': 0.7295466388744137, 'PHR5': 0.5211047420531527, 'PHR6': 0.2605523710265763, 'PHR7': 0.5211047420531527, 'PHR8': 35.95622720166753, 'PHR9': 7.2954663887441376, 'PHR10': 99999, 'PHR11': 2.9702970297029703, 'PHR12': 1.0422094841063054, 'PHR13': 99999, 'PHR14': 99999, 'PHR15': 99999, 'PHR16': 99999, 'PHR17': 99999, 'PHR18': 99999, 'PHR19': 99999, 'PHR20': 99999, 'PHR21': 99999, 'PHR22': 99999, 'PHR23': 99999, 'PHR24': 99999, 'PHR25': 99999, 'PUTGB1': 99999, 'PUTGB2': 99999, 'PUTGB3': 99999, 'PUTGB4': 99999, 'PUTGB5': 99999, 'PUTGB6': 99999, 'PUTGB7': 99999, 'PUTGB8': 99999, 'PUTGB9': 99999, 'PUTGB10': 99999, 'PUTGB11': 99999, 'PUTGB12': 99999, 'PUTGB13': 99999, 'PUTGB14': 99999, 'PUTGB15': 99999, 'PUTGB16': 99999, 'PUTGB17': 99999, 'PUTGB18': 99999, 'PUTGB19': 99999, 'PUTGB20': 99999, 'PUTGB21': 99999, 'PUTGB22': 99999, 'PUTGB23': 99999, 'PUTGB24': 99999, 'PUTGB25': 99999}
                    # ]                    
                    [
                        {'INSP_TIME': 0.1, 'INSP_TEMP': 0.1, 'RPM1': 0.1, 'RPM2': 0.1, 'RPM3': 0.1, 'RPM4': 0.1, 'RPM5': 0.1, 'RPM6': 0.1, 'RPM7': 0.1, 'RPM8': 0.1, 'RPM9': 0.1, 'RPM10': 0.1, 'RPM11': 0.1, 'RPM12': 0.1, 'RPM13': 0.1, 'RPM14': 0.1, 'RPM15': 0.1, 'RPM16': 0.1, 'RPM17': 0.1, 'RPM18': 0.1, 'RPM19': 0.1, 'RPM20': 0.1, 'RPM21': 0.1, 'RPM22': 0.1, 'RPM23': 0.1, 'TEMP1': 0.1, 'TEMP2': 0.1, 'TEMP3': 0.1, 'TEMP4': 0.1, 'TEMP5': 0.1, 'TEMP6': 0.1, 'TEMP7': 0.1, 'TEMP8': 0.1, 'TEMP9': 0.1, 'TEMP10': 0.1, 'TEMP11': 0.1, 'TEMP12': 0.1, 'TEMP13': 0.1, 'TEMP14': 0.1, 'TEMP15': 0.1, 'TEMP16': 0.1, 'TEMP17': 0.1, 'TEMP18': 0.1, 'TEMP19': 0.1, 'TEMP20': 0.1, 'TEMP21': 0.1, 'TEMP22': 0.1, 'TEMP23': 0.1, 'TIME1': 0.1, 'TIME2': 0.1, 'TIME3': 0.1, 'TIME4': 0.1, 'TIME5': 0.1, 'TIME6': 0.1, 'TIME7': 0.1, 'TIME8': 0.1, 'TIME9': 0.1, 'TIME10': 0.1, 'TIME11': 0.1, 'TIME12': 0.1, 'TIME13': 0.1, 'TIME14': 0.1, 'TIME15': 0.1, 'TIME16': 0.1, 'TIME17': 0.1, 'TIME18': 0.1, 'TIME19': 0.1, 'TIME20': 0.1, 'TIME21': 0.1, 'TIME22': 0.1, 'TIME23': 0.1, 'JUK1': 0.1, 'JUK2': 0.1, 'JUK3': 0.1, 'JUK4': 0.1, 'JUK5': 0.1, 'JUK6': 0.1, 'JUK7': 0.1, 'JUK8': 0.1, 'JUK9': 0.1, 'JUK10': 0.1, 'JUK11': 0.1, 'JUK12': 0.1, 'JUK13': 0.1, 'JUK14': 0.1, 'JUK15': 0.1, 'JUK16': 0.1, 'JUK17': 0.1, 'JUK18': 0.1, 'JUK19': 0.1, 'JUK20': 0.1, 'JUK21': 0.1, 'JUK22': 0.1, 'JUK23': 0.1, 'JRCODE1': 0.1, 'JRCODE2': 0.1, 'JRCODE3': 0.1, 'JRCODE4': 0.1, 'JRCODE5': 0.1, 'JRCODE6': 0.1, 'JRCODE7': 0.1, 'JRCODE8': 0.1, 'JRCODE9': 0.1, 'JRCODE10': 0.1, 'JRCODE11': 0.1, 'JRCODE12': 0.1, 'JRCODE13': 0.1, 'JRCODE14': 0.1, 'JRCODE15': 0.1, 'JRCODE16': 0.1, 'JRCODE17': 0.1, 'JRCODE18': 0.1, 'JRCODE19': 0.1, 'JRCODE20': 0.1, 'JRCODE21': 0.1, 'JRCODE22': 0.1, 'JRCODE23': 0.1, 'JRCODE24': 0.1, 'JRCODE25': 0.1, 'PHR1': 0.1, 'PHR2': 0.1, 'PHR3': 0.1, 'PHR4': 0.1, 'PHR5': 0.1, 'PHR6': 0.1, 'PHR7': 0.1, 'PHR8': 0.1, 'PHR9': 0.1, 'PHR10': 0.1, 'PHR11': 0.1, 'PHR12': 0.1, 'PHR13': 0.1, 'PHR14': 0.1, 'PHR15': 0.1, 'PHR16': 0.1, 'PHR17': 0.1, 'PHR18': 0.1, 'PHR19': 0.1, 'PHR20': 0.1, 'PHR21': 0.1, 'PHR22': 0.1, 'PHR23': 0.1, 'PHR24': 0.1, 'PHR25': 0.1, 'PUTGB1': 0.1, 'PUTGB2': 0.1, 'PUTGB3': 0.1, 'PUTGB4': 0.1, 'PUTGB5': 0.1, 'PUTGB6': 0.1, 'PUTGB7': 0.1, 'PUTGB8': 0.1, 'PUTGB9': 0.1, 'PUTGB10': 0.1, 'PUTGB11': 0.1, 'PUTGB12': 0.1, 'PUTGB13': 0.1, 'PUTGB14': 0.1, 'PUTGB15': 0.1, 'PUTGB16': 0.1, 'PUTGB17': 0.1, 'PUTGB18': 0.1, 'PUTGB19': 0.1, 'PUTGB20': 0.1, 'PUTGB21': 0.1, 'PUTGB22': 0.1, 'PUTGB23': 0.1, 'PUTGB24': 0.1, 'PUTGB25': 0.1},
                        {'INSP_TIME': 10.0, 'INSP_TEMP': 170.0, 'RPM1': 'nan', 'RPM2': 'nan', 'RPM3': 'nan', 'RPM4': 'nan', 'RPM5': 'nan', 'RPM6': 'nan', 'RPM7': 'nan', 'RPM8': 'nan', 'RPM9': 'nan', 'RPM10': 'nan', 'RPM11': 'nan', 'RPM12': 'nan', 'RPM13': 'nan', 'RPM14': 'nan', 'RPM15': 'nan', 'RPM16': 'nan', 'RPM17': 'nan', 'RPM18': 'nan', 'RPM19': 'nan', 'RPM20': 'nan', 'RPM21': 'nan', 'RPM22': 'nan', 'RPM23': 'nan', 'TEMP1': 'nan', 'TEMP2': 'nan', 'TEMP3': 'nan', 'TEMP4': 'nan', 'TEMP5': 'nan', 'TEMP6': 'nan', 'TEMP7': 'nan', 'TEMP8': 'nan', 'TEMP9': 'nan', 'TEMP10': 'nan', 'TEMP11': 'nan', 'TEMP12': 'nan', 'TEMP13': 'nan', 'TEMP14': 'nan', 'TEMP15': 'nan', 'TEMP16': 'nan', 'TEMP17': 'nan', 'TEMP18': 'nan', 'TEMP19': 'nan', 'TEMP20': 'nan', 'TEMP21': 'nan', 'TEMP22': 'nan', 'TEMP23': 'nan', 'TIME1': 'nan', 'TIME2': 'nan', 'TIME3': 'nan', 'TIME4': 'nan', 'TIME5': 'nan', 'TIME6': 'nan', 'TIME7': 'nan', 'TIME8': 'nan', 'TIME9': 'nan', 'TIME10': 'nan', 'TIME11': 'nan', 'TIME12': 'nan', 'TIME13': 'nan', 'TIME14': 'nan', 'TIME15': 'nan', 'TIME16': 'nan', 'TIME17': 'nan', 'TIME18': 'nan', 'TIME19': 'nan', 'TIME20': 'nan', 'TIME21': 'nan', 'TIME22': 'nan', 'TIME23': 'nan', 'JUK1': 'nan', 'JUK2': 'nan', 'JUK3': 'nan', 'JUK4': 'nan', 'JUK5': 'nan', 'JUK6': 'nan', 'JUK7': 'nan', 'JUK8': 'nan', 'JUK9': 'nan', 'JUK10': 'nan', 'JUK11': 'nan', 'JUK12': 'nan', 'JUK13': 'nan', 'JUK14': 'nan', 'JUK15': 'nan', 'JUK16': 'nan', 'JUK17': 'nan', 'JUK18': 'nan', 'JUK19': 'nan', 'JUK20': 'nan', 'JUK21': 'nan', 'JUK22': 'nan', 'JUK23': 'nan', 'JRCODE1': 'EE011A', 'JRCODE2': 'CB003A', 'JRCODE3': 'CB001C', 'JRCODE4': 'CC001A', 'JRCODE5': 'CC012B', 'JRCODE6': 'CD005A', 'JRCODE7': 'NN031A', 'JRCODE8': 'OP008A', 'JRCODE9': 'nan', 'JRCODE10': 'CV013A', 'JRCODE11': 'CV011A', 'JRCODE12': 'nan', 'JRCODE13': 'nan', 'JRCODE14': 'nan', 'JRCODE15': 'nan', 'JRCODE16': 'nan', 'JRCODE17': 'nan', 'JRCODE18': 'nan', 'JRCODE19': 'nan', 'JRCODE20': 'nan', 'JRCODE21': 'nan', 'JRCODE22': 'nan', 'JRCODE23': 'nan', 'JRCODE24': 'nan', 'JRCODE25': 'nan', 'PHR1': 56.49717514124294, 'PHR2': 2.824858757062147, 'PHR3': 0.5649717514124294, 'PHR4': 0.5649717514124294, 'PHR5': 0.5649717514124294, 'PHR6': 0.5649717514124294, 'PHR7': 22.598870056497177, 'PHR8': 15.819209039548024, 'PHR9': 'nan', 'PHR10': 3.389830508474576, 'PHR11': 1.1299435028248588, 'PHR12': 'nan', 'PHR13': 'nan', 'PHR14': 'nan', 'PHR15': 'nan', 'PHR16': 'nan', 'PHR17': 'nan', 'PHR18': 'nan', 'PHR19': 'nan', 'PHR20': 'nan', 'PHR21': 'nan', 'PHR22': 'nan', 'PHR23': 'nan', 'PHR24': 'nan', 'PHR25': 'nan', 'PUTGB1': 'nan', 'PUTGB2': 'nan', 'PUTGB3': 'nan', 'PUTGB4': 'nan', 'PUTGB5': 'nan', 'PUTGB6': 'nan', 'PUTGB7': 'nan', 'PUTGB8': 'nan', 'PUTGB9': 'nan', 'PUTGB10': 'nan', 'PUTGB11': 'nan', 'PUTGB12': 'nan', 'PUTGB13': 'nan', 'PUTGB14': 'nan', 'PUTGB15': 'nan', 'PUTGB16': 'nan', 'PUTGB17': 'nan', 'PUTGB18': 'nan', 'PUTGB19': 'nan', 'PUTGB20': 'nan', 'PUTGB21': 'nan', 'PUTGB22': 'nan', 'PUTGB23': 'nan', 'PUTGB24': 'nan', 'PUTGB25': 'nan'},
                        {'INSP_TIME': 10.0, 'INSP_TEMP': 170.0, 'RPM1': 'nan', 'RPM2': 'nan', 'RPM3': 'nan', 'RPM4': 'nan', 'RPM5': 'nan', 'RPM6': 'nan', 'RPM7': 'nan', 'RPM8': 'nan', 'RPM9': 'nan', 'RPM10': 'nan', 'RPM11': 'nan', 'RPM12': 'nan', 'RPM13': 'nan', 'RPM14': 'nan', 'RPM15': 'nan', 'RPM16': 'nan', 'RPM17': 'nan', 'RPM18': 'nan', 'RPM19': 'nan', 'RPM20': 'nan', 'RPM21': 'nan', 'RPM22': 'nan', 'RPM23': 'nan', 'TEMP1': 'nan', 'TEMP2': 'nan', 'TEMP3': 'nan', 'TEMP4': 'nan', 'TEMP5': 'nan', 'TEMP6': 'nan', 'TEMP7': 'nan', 'TEMP8': 'nan', 'TEMP9': 'nan', 'TEMP10': 'nan', 'TEMP11': 'nan', 'TEMP12': 'nan', 'TEMP13': 'nan', 'TEMP14': 'nan', 'TEMP15': 'nan', 'TEMP16': 'nan', 'TEMP17': 'nan', 'TEMP18': 'nan', 'TEMP19': 'nan', 'TEMP20': 'nan', 'TEMP21': 'nan', 'TEMP22': 'nan', 'TEMP23': 'nan', 'TIME1': 'nan', 'TIME2': 'nan', 'TIME3': 'nan', 'TIME4': 'nan', 'TIME5': 'nan', 'TIME6': 'nan', 'TIME7': 'nan', 'TIME8': 'nan', 'TIME9': 'nan', 'TIME10': 'nan', 'TIME11': 'nan', 'TIME12': 'nan', 'TIME13': 'nan', 'TIME14': 'nan', 'TIME15': 'nan', 'TIME16': 'nan', 'TIME17': 'nan', 'TIME18': 'nan', 'TIME19': 'nan', 'TIME20': 'nan', 'TIME21': 'nan', 'TIME22': 'nan', 'TIME23': 'nan', 'JUK1': 'nan', 'JUK2': 'nan', 'JUK3': 'nan', 'JUK4': 'nan', 'JUK5': 'nan', 'JUK6': 'nan', 'JUK7': 'nan', 'JUK8': 'nan', 'JUK9': 'nan', 'JUK10': 'nan', 'JUK11': 'nan', 'JUK12': 'nan', 'JUK13': 'nan', 'JUK14': 'nan', 'JUK15': 'nan', 'JUK16': 'nan', 'JUK17': 'nan', 'JUK18': 'nan', 'JUK19': 'nan', 'JUK20': 'nan', 'JUK21': 'nan', 'JUK22': 'nan', 'JUK23': 'nan', 'JRCODE1': 'EN005A', 'JRCODE2': 'EN011A', 'JRCODE3': 'CB009C', 'JRCODE4': 'CB001C', 'JRCODE5': 'CC006A', 'JRCODE6': 'CC001A', 'JRCODE7': 'CC002B', 'JRCODE8': 'CC010A', 'JRCODE9': 'CC014A', 'JRCODE10': 'CV001D', 'JRCODE11': 'NN031A', 'JRCODE12': 'NN774A', 'JRCODE13': 'OD004B', 'JRCODE14': 'nan', 'JRCODE15': 'CA011D', 'JRCODE16': 'CA008D', 'JRCODE17': 'CC003D', 'JRCODE18': 'CW001A', 'JRCODE19': 'CC009B', 'JRCODE20': 'nan', 'JRCODE21': 'nan', 'JRCODE22': 'nan', 'JRCODE23': 'nan', 'JRCODE24': 'nan', 'JRCODE25': 'nan', 'PHR1': 26.638252530633995, 'PHR2': 26.638252530633995, 'PHR3': 3.1965903036760794, 'PHR4': 0.5327650506126799, 'PHR5': 1.0655301012253595, 'PHR6': 1.0655301012253595, 'PHR7': 1.0655301012253595, 'PHR8': 1.0655301012253595, 'PHR9': 1.1720831113478958, 'PHR10': 0.2663825253063399, 'PHR11': 9.057005860415558, 'PHR12': 23.97442727757059, 'PHR13': 4.262120404901439, 'PHR14': 'nan', 'PHR15': 0.7991475759190199, 'PHR16': 1.0655301012253595, 'PHR17': 0.5327650506126799, 'PHR18': 0.5327650506126799, 'PHR19': 1.0655301012253595, 'PHR20': 'nan', 'PHR21': 'nan', 'PHR22': 'nan', 'PHR23': 'nan', 'PHR24': 'nan', 'PHR25': 'nan', 'PUTGB1': 'nan', 'PUTGB2': 'nan', 'PUTGB3': 'nan', 'PUTGB4': 'nan', 'PUTGB5': 'nan', 'PUTGB6': 'nan', 'PUTGB7': 'nan', 'PUTGB8': 'nan', 'PUTGB9': 'nan', 'PUTGB10': 'nan', 'PUTGB11': 'nan', 'PUTGB12': 'nan', 'PUTGB13': 'nan', 'PUTGB14': 'nan', 'PUTGB15': 'nan', 'PUTGB16': 'nan', 'PUTGB17': 'nan', 'PUTGB18': 'nan', 'PUTGB19': 'nan', 'PUTGB20': 'nan', 'PUTGB21': 'nan', 'PUTGB22': 'nan', 'PUTGB23': 'nan', 'PUTGB24': 'nan', 'PUTGB25': 'nan'},
                        {'INSP_TIME': 'nan', 'INSP_TEMP': 160.0, 'RPM1': 51.0, 'RPM2': 51.0, 'RPM3': 51.0, 'RPM4': 51.0, 'RPM5': 51.0, 'RPM6': 51.0, 'RPM7': 51.0, 'RPM8': 51.0, 'RPM9': 51.0, 'RPM10': 51.0, 'RPM11': 51.0, 'RPM12': 'nan', 'RPM13': 'nan', 'RPM14': 'nan', 'RPM15': 'nan', 'RPM16': 'nan', 'RPM17': 'nan', 'RPM18': 'nan', 'RPM19': 'nan', 'RPM20': 'nan', 'RPM21': 'nan', 'RPM22': 'nan', 'RPM23': 'nan', 'TEMP1': 128.3520231213873, 'TEMP2': 86.0757225433526, 'TEMP3': 88.12543352601156, 'TEMP4': 94.13121387283238, 'TEMP5': 89.56127167630058, 'TEMP6': 106.61156069364162, 'TEMP7': 115.05953757225431, 'TEMP8': 139.22196531791909, 'TEMP9': 146.3578034682081, 'TEMP10': 168.7635838150289, 'TEMP11': 149.91445086705204, 'TEMP12': 'nan', 'TEMP13': 'nan', 'TEMP14': 'nan', 'TEMP15': 'nan', 'TEMP16': 'nan', 'TEMP17': 'nan', 'TEMP18': 'nan', 'TEMP19': 'nan', 'TEMP20': 'nan', 'TEMP21': 'nan', 'TEMP22': 'nan', 'TEMP23': 'nan', 'TIME1': 12.579190751445086, 'TIME2': 10.457803468208091, 'TIME3': 54.54046242774567, 'TIME4': 16.40057803468208, 'TIME5': 31.07745664739884, 'TIME6': 24.590173410404628, 'TIME7': 16.432947976878612, 'TIME8': 31.22890173410405, 'TIME9': 7.420231213872833, 'TIME10': 33.42485549132948, 'TIME11': 16.887283236994218, 'TIME12': 'nan', 'TIME13': 'nan', 'TIME14': 'nan', 'TIME15': 'nan', 'TIME16': 'nan', 'TIME17': 'nan', 'TIME18': 'nan', 'TIME19': 'nan', 'TIME20': 'nan', 'TIME21': 'nan', 'TIME22': 'nan', 'TIME23': 'nan', 'JUK1': 32.76878612716763, 'JUK2': 24.739884393063583, 'JUK3': 114.44508670520231, 'JUK4': 24.427745664739884, 'JUK5': 49.17919075144509, 'JUK6': 47.190751445086704, 'JUK7': 34.60115606936416, 'JUK8': 55.6878612716763, 'JUK9': 11.46242774566474, 'JUK10': 76.20231213872832, 'JUK11': 48.58381502890173, 'JUK12': 'nan', 'JUK13': 'nan', 'JUK14': 'nan', 'JUK15': 'nan', 'JUK16': 'nan', 'JUK17': 'nan', 'JUK18': 'nan', 'JUK19': 'nan', 'JUK20': 'nan', 'JUK21': 'nan', 'JUK22': 'nan', 'JUK23': 'nan', 'JRCODE1': 'EN006A', 'JRCODE2': 'CB009', 'JRCODE3': 'CJ001A', 'JRCODE4': 'CB001', 'JRCODE5': 'CC001', 'JRCODE6': 'CD008', 'JRCODE7': 'CD013', 'JRCODE8': 'CT001', 'JRCODE9': 'FF005', 'JRCODE10': 'NN774', 'JRCODE11': 'NN550', 'JRCODE12': 'OD006', 'JRCODE13': 'CV001', 'JRCODE14': 'CA011', 'JRCODE15': 'CA008', 'JRCODE16': 'nan', 'JRCODE17': 'nan', 'JRCODE18': 'nan', 'JRCODE19': 'nan', 'JRCODE20': 'nan', 'JRCODE21': 'nan', 'JRCODE22': 'nan', 'JRCODE23': 'nan', 'JRCODE24': 'nan', 'JRCODE25': 'nan', 'PHR1': 91.26984126984128, 'PHR2': 6.38888888888889, 'PHR3': 0.096763464538031, 'PHR4': 0.9126984126984126, 'PHR5': 1.369047619047619, 'PHR6': 0.9126984126984126, 'PHR7': 0.9126984126984126, 'PHR8': 7.3015873015873, 'PHR9': 31.89742178181485, 'PHR10': 32.071749701807505, 'PHR11': 37.10317460317461, 'PHR12': 21.945820717497018, 'PHR13': 0.1954365079365079, 'PHR14': 0.5873015873015873, 'PHR15': 0.5873015873015873, 'PHR16': 'nan', 'PHR17': 'nan', 'PHR18': 'nan', 'PHR19': 'nan', 'PHR20': 'nan', 'PHR21': 'nan', 'PHR22': 'nan', 'PHR23': 'nan', 'PHR24': 'nan', 'PHR25': 'nan', 'PUTGB1': 1.0, 'PUTGB2': 1.0, 'PUTGB3': 1.0, 'PUTGB4': 1.0, 'PUTGB5': 1.0, 'PUTGB6': 1.0, 'PUTGB7': 1.0, 'PUTGB8': 1.0, 'PUTGB9': 2.0, 'PUTGB10': 3.0, 'PUTGB11': 3.0, 'PUTGB12': 5.0, 'PUTGB13': 6.0, 'PUTGB14': 6.0, 'PUTGB15': 6.0, 'PUTGB16': 'nan', 'PUTGB17': 'nan', 'PUTGB18': 'nan', 'PUTGB19': 'nan', 'PUTGB20': 'nan', 'PUTGB21': 'nan', 'PUTGB22': 'nan', 'PUTGB23': 'nan', 'PUTGB24': 'nan', 'PUTGB25': 'nan'},
                        {'INSP_TIME': 8.0, 'INSP_TEMP': 180.0, 'RPM1': 'nan', 'RPM2': 'nan', 'RPM3': 'nan', 'RPM4': 'nan', 'RPM5': 'nan', 'RPM6': 'nan', 'RPM7': 'nan', 'RPM8': 'nan', 'RPM9': 'nan', 'RPM10': 'nan', 'RPM11': 'nan', 'RPM12': 'nan', 'RPM13': 'nan', 'RPM14': 'nan', 'RPM15': 'nan', 'RPM16': 'nan', 'RPM17': 'nan', 'RPM18': 'nan', 'RPM19': 'nan', 'RPM20': 'nan', 'RPM21': 'nan', 'RPM22': 'nan', 'RPM23': 'nan', 'TEMP1': 'nan', 'TEMP2': 'nan', 'TEMP3': 'nan', 'TEMP4': 'nan', 'TEMP5': 'nan', 'TEMP6': 'nan', 'TEMP7': 'nan', 'TEMP8': 'nan', 'TEMP9': 'nan', 'TEMP10': 'nan', 'TEMP11': 'nan', 'TEMP12': 'nan', 'TEMP13': 'nan', 'TEMP14': 'nan', 'TEMP15': 'nan', 'TEMP16': 'nan', 'TEMP17': 'nan', 'TEMP18': 'nan', 'TEMP19': 'nan', 'TEMP20': 'nan', 'TEMP21': 'nan', 'TEMP22': 'nan', 'TEMP23': 'nan', 'TIME1': 'nan', 'TIME2': 'nan', 'TIME3': 'nan', 'TIME4': 'nan', 'TIME5': 'nan', 'TIME6': 'nan', 'TIME7': 'nan', 'TIME8': 'nan', 'TIME9': 'nan', 'TIME10': 'nan', 'TIME11': 'nan', 'TIME12': 'nan', 'TIME13': 'nan', 'TIME14': 'nan', 'TIME15': 'nan', 'TIME16': 'nan', 'TIME17': 'nan', 'TIME18': 'nan', 'TIME19': 'nan', 'TIME20': 'nan', 'TIME21': 'nan', 'TIME22': 'nan', 'TIME23': 'nan', 'JUK1': 'nan', 'JUK2': 'nan', 'JUK3': 'nan', 'JUK4': 'nan', 'JUK5': 'nan', 'JUK6': 'nan', 'JUK7': 'nan', 'JUK8': 'nan', 'JUK9': 'nan', 'JUK10': 'nan', 'JUK11': 'nan', 'JUK12': 'nan', 'JUK13': 'nan', 'JUK14': 'nan', 'JUK15': 'nan', 'JUK16': 'nan', 'JUK17': 'nan', 'JUK18': 'nan', 'JUK19': 'nan', 'JUK20': 'nan', 'JUK21': 'nan', 'JUK22': 'nan', 'JUK23': 'nan', 'JRCODE1': 'EE011A', 'JRCODE2': 'EE001A', 'JRCODE3': 'CB003A', 'JRCODE4': 'CB001C', 'JRCODE5': 'CC001A', 'JRCODE6': 'CC012B', 'JRCODE7': 'CD005A', 'JRCODE8': 'NN550A', 'JRCODE9': 'OP004A', 'JRCODE10': 'nan', 'JRCODE11': 'CV013A', 'JRCODE12': 'CV011A', 'JRCODE13': 'nan', 'JRCODE14': 'nan', 'JRCODE15': 'nan', 'JRCODE16': 'nan', 'JRCODE17': 'nan', 'JRCODE18': 'nan', 'JRCODE19': 'nan', 'JRCODE20': 'nan', 'JRCODE21': 'nan', 'JRCODE22': 'nan', 'JRCODE23': 'nan', 'JRCODE24': 'nan', 'JRCODE25': 'nan', 'PHR1': 36.477331943720685, 'PHR2': 15.633142261594578, 'PHR3': 2.605523710265764, 'PHR4': 0.7295466388744137, 'PHR5': 0.5211047420531527, 'PHR6': 0.2605523710265763, 'PHR7': 0.5211047420531527, 'PHR8': 35.95622720166753, 'PHR9': 7.2954663887441376, 'PHR10': 'nan', 'PHR11': 2.9702970297029703, 'PHR12': 1.0422094841063054, 'PHR13': 'nan', 'PHR14': 'nan', 'PHR15': 'nan', 'PHR16': 'nan', 'PHR17': 'nan', 'PHR18': 'nan', 'PHR19': 'nan', 'PHR20': 'nan', 'PHR21': 'nan', 'PHR22': 'nan', 'PHR23': 'nan', 'PHR24': 'nan', 'PHR25': 'nan', 'PUTGB1': 'nan', 'PUTGB2': 'nan', 'PUTGB3': 'nan', 'PUTGB4': 'nan', 'PUTGB5': 'nan', 'PUTGB6': 'nan', 'PUTGB7': 'nan', 'PUTGB8': 'nan', 'PUTGB9': 'nan', 'PUTGB10': 'nan', 'PUTGB11': 'nan', 'PUTGB12': 'nan', 'PUTGB13': 'nan', 'PUTGB14': 'nan', 'PUTGB15': 'nan', 'PUTGB16': 'nan', 'PUTGB17': 'nan', 'PUTGB18': 'nan', 'PUTGB19': 'nan', 'PUTGB20': 'nan', 'PUTGB21': 'nan', 'PUTGB22': 'nan', 'PUTGB23': 'nan', 'PUTGB24': 'nan', 'PUTGB25': 'nan'},
                        {'INSP_TIME': 10.0, 'INSP_TEMP': 160.0, 'RPM1': 'nan', 'RPM2': 'nan', 'RPM3': 'nan', 'RPM4': 'nan', 'RPM5': 'nan', 'RPM6': 'nan', 'RPM7': 'nan', 'RPM8': 'nan', 'RPM9': 'nan', 'RPM10': 'nan', 'RPM11': 'nan', 'RPM12': 'nan', 'RPM13': 'nan', 'RPM14': 'nan', 'RPM15': 'nan', 'RPM16': 'nan', 'RPM17': 'nan', 'RPM18': 'nan', 'RPM19': 'nan', 'RPM20': 'nan', 'RPM21': 'nan', 'RPM22': 'nan', 'RPM23': 'nan', 'TEMP1': 'nan', 'TEMP2': 'nan', 'TEMP3': 'nan', 'TEMP4': 'nan', 'TEMP5': 'nan', 'TEMP6': 'nan', 'TEMP7': 'nan', 'TEMP8': 'nan', 'TEMP9': 'nan', 'TEMP10': 'nan', 'TEMP11': 'nan', 'TEMP12': 'nan', 'TEMP13': 'nan', 'TEMP14': 'nan', 'TEMP15': 'nan', 'TEMP16': 'nan', 'TEMP17': 'nan', 'TEMP18': 'nan', 'TEMP19': 'nan', 'TEMP20': 'nan', 'TEMP21': 'nan', 'TEMP22': 'nan', 'TEMP23': 'nan', 'TIME1': 'nan', 'TIME2': 'nan', 'TIME3': 'nan', 'TIME4': 'nan', 'TIME5': 'nan', 'TIME6': 'nan', 'TIME7': 'nan', 'TIME8': 'nan', 'TIME9': 'nan', 'TIME10': 'nan', 'TIME11': 'nan', 'TIME12': 'nan', 'TIME13': 'nan', 'TIME14': 'nan', 'TIME15': 'nan', 'TIME16': 'nan', 'TIME17': 'nan', 'TIME18': 'nan', 'TIME19': 'nan', 'TIME20': 'nan', 'TIME21': 'nan', 'TIME22': 'nan', 'TIME23': 'nan', 'JUK1': 'nan', 'JUK2': 'nan', 'JUK3': 'nan', 'JUK4': 'nan', 'JUK5': 'nan', 'JUK6': 'nan', 'JUK7': 'nan', 'JUK8': 'nan', 'JUK9': 'nan', 'JUK10': 'nan', 'JUK11': 'nan', 'JUK12': 'nan', 'JUK13': 'nan', 'JUK14': 'nan', 'JUK15': 'nan', 'JUK16': 'nan', 'JUK17': 'nan', 'JUK18': 'nan', 'JUK19': 'nan', 'JUK20': 'nan', 'JUK21': 'nan', 'JUK22': 'nan', 'JUK23': 'nan', 'JRCODE1': 'EE011A', 'JRCODE2': 'EY002B', 'JRCODE3': 'CB003A', 'JRCODE4': 'CB001C', 'JRCODE5': 'CB011B', 'JRCODE6': 'CT001A', 'JRCODE7': 'NN031A', 'JRCODE8': 'FF005A', 'JRCODE9': 'OP005A', 'JRCODE10': 'nan', 'JRCODE11': 'CA003C', 'JRCODE12': 'CA019C', 'JRCODE13': 'CA011D', 'JRCODE14': 'CV001D', 'JRCODE15': 'CA013A', 'JRCODE16': 'CE001D', 'JRCODE17': 'nan', 'JRCODE18': 'nan', 'JRCODE19': 'nan', 'JRCODE20': 'nan', 'JRCODE21': 'nan', 'JRCODE22': 'nan', 'JRCODE23': 'nan', 'JRCODE24': 'nan', 'JRCODE25': 'nan', 'PHR1': 11.673151750972762, 'PHR2': 27.237354085603112, 'PHR3': 1.9455252918287933, 'PHR4': 0.3891050583657587, 'PHR5': 0.3891050583657587, 'PHR6': 1.1673151750972763, 'PHR7': 31.1284046692607, 'PHR8': 9.72762645914397, 'PHR9': 16.342412451361866, 'PHR10': 'nan', 'PHR11': 0.3891050583657587, 'PHR12': 0.1945525291828793, 'PHR13': 0.1556420233463035, 'PHR14': 0.3891050583657587, 'PHR15': 0.2334630350194552, 'PHR16': 0.0778210116731517, 'PHR17': 'nan', 'PHR18': 'nan', 'PHR19': 'nan', 'PHR20': 'nan', 'PHR21': 'nan', 'PHR22': 'nan', 'PHR23': 'nan', 'PHR24': 'nan', 'PHR25': 'nan', 'PUTGB1': 'nan', 'PUTGB2': 'nan', 'PUTGB3': 'nan', 'PUTGB4': 'nan', 'PUTGB5': 'nan', 'PUTGB6': 'nan', 'PUTGB7': 'nan', 'PUTGB8': 'nan', 'PUTGB9': 'nan', 'PUTGB10': 'nan', 'PUTGB11': 'nan', 'PUTGB12': 'nan', 'PUTGB13': 'nan', 'PUTGB14': 'nan', 'PUTGB15': 'nan', 'PUTGB16': 'nan', 'PUTGB17': 'nan', 'PUTGB18': 'nan', 'PUTGB19': 'nan', 'PUTGB20': 'nan', 'PUTGB21': 'nan', 'PUTGB22': 'nan', 'PUTGB23': 'nan', 'PUTGB24': 'nan', 'PUTGB25': 'nan'}
                    ]
                }
            print("message : {}".format(message))
            value = json.dumps(message)

            print('test3')
            self.producer.send(self.topic_request, value=value.encode())
            # print(value.encode())
            print('test4')
            # print(message['DATA'])            

        except Exception as e:
            print(e)
            pass

def start_consumer_process():
    mlops_kafka_exam = MlopsKafkaExam()
    mlops_kafka_exam.run_consumer()

def start_producer_process():
    mlops_kafka_exam = MlopsKafkaExam()
    mlops_kafka_exam.run_producer()


if __name__ == "__main__":
    processes = list()

    # 컨슈머 프로세스 시작
    p = Process(target=start_consumer_process)
    processes.append(p)

    # 프로듀서 프로세스 시작
    p = Process(target=start_producer_process)
    processes.append(p)


    for p in processes:
        p.start()
        time.sleep(1)

    for p in processes:
        p.join()
