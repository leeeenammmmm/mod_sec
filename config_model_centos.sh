
echo "### Install anomaly detection model keras-anomaly-detection-master"

rm -rf /opt/model/
cp -r model/ /opt/
cd /opt/model/keras-anomaly-detection-master/
python36 setup.py install

cp /opt/model/keras_anomaly_centos.service /etc/systemd/system/

systemctl daemon-reload
systemctl start redis.service
systemctl start keras_anomaly_centos.service
