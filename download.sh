
# Download book crossing dataset
if [ ! -d data ]
then
    if [ ! -f final_project.zip ]
    then
        wget 'https://learner.csie.ntu.edu.tw/~judge/ml18spring/final_project.zip'
    fi
    unzip final_project.zip
fi
