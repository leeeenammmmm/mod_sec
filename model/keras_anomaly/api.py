from lib.http_detection import HttpPredict
import redis

r = redis.StrictRedis(host='localhost', port=6379, db=0)
myHttpPredict = HttpPredict()
myHttpPredict.loadModelInit()

# data= "GET /log"
# predict = myHttpPredict.predict_binary(data)
# print(predict)
if __name__ == '__main__':
	while 1 > 0:
		keys = r.lpop("myRequestQueue")
		# print(keys)
		if keys is None:
			continue
		data = str(r.get(keys))
		data = data.replace("'", "")
		# print(data)
		data = data.replace("$", "/")
		data = data.replace(">", " ")
		binary = myHttpPredict.predict_binary(data)
		r.set(keys.decode("utf-8"), binary)
