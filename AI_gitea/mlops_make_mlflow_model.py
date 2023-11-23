# 파일이름 : mlops_make_mlflow_model.py
# 코드설명 : MLOps 탑재를 위한 mlflow 최종 예측 모델 생성
# 입/출력 : 예측 대상별 학습된 모델들 / 테스트 데이터의 물성 예측 결과
# 유의 사항 : 
# 1. 예측 대상별로 최종 선정된 모델 기술명 매칭
# 2. 디버깅을 위한 에러 출력이 어려우므로 Output.txt 파일에 로그를 기록하는 방식으로 디버깅 진행
# 3. 모델 저장 경로 설정 : “mlflow_pyfunc_model_path”, 추후 테스트에서 사용
# 4. mlflow(2.0.1)와 autogluon(0.8.2), python(3.9.18) 버전을 맞추어야 함
# 최종수정 : 2023년 11월 24일
# 제 작 자 : 홍민성 (mshong@micube.co.kr) {전체 개발}
# Copyright : MICUBE Solution, Inc.

from sys import version_info
import mlflow.pyfunc
import autogluon.core as ag
import cloudpickle  # Create a Conda environment for the new MLflow Model that contains all necessary dependencies.
import pandas as pd
import numpy as np
import json

PYTHON_VERSION = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
# 테스트용
# artifacts = {'HS_RESULT': 'SFT_clas/combined_HS_RESULT_models',
#              'SG_RESULT': 'SFT_clas/combined_SG_RESULT_models',
#              'HS': 'SFT_regr/combined_HS_models',
#              }  # 학습 시킨 모델의 경로

artifacts = {'HS': 'SFT_regr/combined_HS_models',
             'SG': 'SFT_regr/combined_SG_models',
             'TS': 'SFT_regr/combined_TS_models',
             'EB': 'SFT_regr/combined_EB_models',
             'MNY' : 'SFT_regr/combined_MNY_models',
             'REHO_MIN' : 'SFT_regr/combined_REHO_MIN_models',
             'REHO_MAX' : 'SFT_regr/combined_REHO_MAX_models',
             'REHO_TS2' : 'SFT_regr/combined_REHO_TS2_models',
             'REHO_TC90' : 'SFT_regr/combined_REHO_TC90_models',
             'SCR' : 'SFT_regr/combined_SCR_models',

             'HS_RESULT': 'SFT_clas/combined_HS_RESULT_models',
             'SG_RESULT': 'SFT_clas/combined_SG_RESULT_models',
             'TS_RESULT': 'SFT_clas/combined_TS_RESULT_models',
             'EB_RESULT': 'SFT_clas/combined_EB_RESULT_models',
             'MNY_RESULT' : 'SFT_clas/combined_MNY_RESULT_models',
             'REHO_RESULT' : 'SFT_clas/combined_REHO_RESULT_models',
             'SCR_RESULT' : 'SFT_clas/combined_SCR_RESULT_models',
             }  # 학습 시킨 모델의 경로

class SFTprediction(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """
        from autogluon.tabular import TabularPredictor
        self.model_ls = []
        # 모델 로드 및 리스트 형태로 저장
        for yCol in list(context.artifacts.keys()):
            self.predictor = TabularPredictor.load(path=context.artifacts[yCol], verbosity=0)
            self.model_ls.append(self.predictor)

    def predict(self, context, model_input: list, params=None):
        """This is an abstract function. We customized it into a method to fetch the FastText model.
        Args:
            context ([type]): MLflow context where the model artifact is stored.
            model_input ([type]): the input data to fit into the model.
        Returns:
            [type]: the loaded model artifact.
        """
        model_input = model_input.to_dict('records')
        text_file = open("Output.txt", "w")
        try : 
            # 여기서 model_input의 타입은 반드시 List[Dict]여야함
            for i, row in enumerate(model_input):
                # text_file.write("\n" % row)
                model_input[i] = {k:(np.nan if v == 'nan' else v) for k,v in row.items()}

            # return model_input
            self.input = pd.DataFrame(model_input)
            self.input = self.input.iloc[1:]

            pred_dic = {}
            # 예측 대상별로 모델을 불러와서 예측한 결과를 dictionary에 저장
            for i in np.arange(len(context.artifacts)):
                yCol = list(context.artifacts.keys())[i]
                if yCol in ['HS', 'SG', 'TS', 'EB']:
                    model_name = 'LightGBM_BAG_L2'
                elif yCol in ['MNY', 'REHO_MIN', 'REHO_MAX', 'REHO_TS2', 'REHO_TC90', 'SCR']:
                    model_name = 'LightGBMXT_BAG_L1'
                else:
                    model_name = 'RandomForestEntr_BAG_L1'
                pred_dic[yCol] = list(self.model_ls[i].predict(self.input, model=model_name))
        except Exception as e:
            text_file.write("\error : %s" % str(e))
        else:
            text_file.write("\npred_dic : %s" % pred_dic)
            
        return pred_dic

model = SFTprediction()
conda_env = {
    "channels": ['conda-forge'],
    "dependencies": [
        f"python={PYTHON_VERSION}",
        "pip",
        {
            "pip": [
                f"mlflow=={mlflow.__version__}",
                f"autogluon=={ag.__version__}",
                # f"pandas=={np.__version__}",
                # f"numpy=={pd.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
            ],
        },
    ],
    "name": "sft_env",
}

# Save the MLflow Model
mlflow_pyfunc_model_path = "Final_SFT"
mlflow.pyfunc.save_model(
    path=mlflow_pyfunc_model_path,
    python_model=SFTprediction(),
    artifacts=artifacts,
    conda_env=conda_env,
)