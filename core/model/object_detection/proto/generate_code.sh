###
 # @Author: captainfffsama
 # @Date: 2023-03-28 12:04:59
 # @LastEditors: captainfffsama tuanzhangsama@outlook.com
 # @LastEditTime: 2023-03-28 12:05:01
 # @FilePath: /dataset_manager/core/model/object_detection/proto/generate_code.sh
 # @Description:
###
python -m grpc_tools.protoc -I ./ --proto_path=./dldetection.proto --python_out=. --pyi_out=. --grpc_python_out=./ dldetection.proto