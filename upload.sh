#!/bin/bash

# Kiểm tra xem có nhập được thông điệp commit không
if [ -z "$1" ]
then
   echo "Hãy nhập thông điệp commit như sau:"
   echo "./push_to_github.sh \"Thông điệp commit của bạn\""
   exit 1
fi
git checkout gg_colab_fix
# Thêm tất cả các thay đổi vào staging
git add .

# Commit các thay đổi với thông điệp
git commit -m "$1"



# Push code lên nhánh chính (main)
git push origin gg_colab_fix --force

echo "Đã push code lên GitHub thành công!"


# Cấp quyền
# chmod +x push_to_github.sh
# run
# ./push_to_github.sh "Thông điệp commit của bạn"
