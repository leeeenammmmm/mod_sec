from lib.http_detection import HttpPredict
import redis

r = redis.StrictRedis(host='localhost', port=6379, db=0)
myHttpPredict = HttpPredict()
myHttpPredict.loadModelInit()

data= "GET /index.php?dungyeuphuong"
predict = myHttpPredict.predict_binary(data)
print(predict)	
