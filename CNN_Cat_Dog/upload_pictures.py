# coding:utf-8

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import modelling_and_classification as tf_model
from datetime import timedelta

# 设置允许的文件格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
        file_name = f.filename

        # 使用Opencv转换一下图片名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'rename.jpg'), img)

        tf_model.all_test_files_dir = "./static/images"
        my_tensor = tf_model.MyTensor()
        output = my_tensor.final_classify_single()

        #清空文件夹
        all_test_filenames = os.listdir(tf_model.all_test_files_dir)
        for each_filename in all_test_filenames:
            if each_filename == "rename.jpg":
                continue
            os.remove(os.path.join(tf_model.all_test_files_dir, each_filename))

        return render_template('upload_ok.html', userinput=file_name, result=output)

    return render_template('upload.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=5000, debug=True)
