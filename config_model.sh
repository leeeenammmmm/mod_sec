
echo "### Install anomaly detection model"

rm -rf /opt/model/
cp -r model/ /opt/
cd /opt/model/keras-anomaly-detection-master/
python3 setup.py install

cp /opt/model/keras_anomaly.service /etc/systemd/system/

echo "start service"
systemctl daemon-reload
systemctl enable keras_anomaly.service
systemctl start keras_anomaly.service
