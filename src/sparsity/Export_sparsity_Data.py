# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   tencent-advertise
@File       :   Export_tf_idf_Data.py
@Version    :   v0.1
@Time       :   2020-07-09 18:06
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   输出训练使用的数据集
@理解：
"""

from tools import beep_end, show_title


# ----------------------------------------------------------------------
def main():
    show_title("数据清洗开始...")
    from src.data.export_data import export_base_data
    x_data, y_data = export_base_data()
    # from src.data.load_data import load_base_data
    # x_data, y_data = load_base_data()

    from src.data.export_data import export_train_test_data
    x_train, y_train = export_train_test_data(x_data, y_data)
    # from src.data.load_data import load_train_data
    # x_train, y_train = load_train_data()
    # export_train_balance(x_train, y_train)

    from src.data.export_data import export_val_data
    x_train_val, y_train_val = export_val_data(x_train, y_train)
    # from src.data.load_data import load_val_data
    # x_train_val, y_train_val = load_val_data()
    # export_val_balance(x_train_val, y_train_val)

    show_title("数据清洗完成！")


# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()
    beep_end()  # 运行结束的提醒
