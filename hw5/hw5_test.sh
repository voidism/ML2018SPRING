if [ -f "./ensemble_82966.h5" ]; then
    echo "Model ensemble_82966.h5 exists!"
else
    echo "Download model ..."
    wget -O ensemble_82966.h5 https://www.dropbox.com/s/zxgdldl6aiwxekw/ensemble_82966.h5?dl=1
fi
python3 hw5_test.py $1 $2