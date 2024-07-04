from lib.unknown_attack_classification import UnknownAttackClassificationModel
import redis
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

r = redis.StrictRedis(host='localhost', port=6379, db=0)
myHttpPredict = UnknownAttackClassificationModel()
myHttpPredict.loadModelInit()

data= "GET /log"
predict = myHttpPredict.predict(data)
print(predict)
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
		predict = myHttpPredict.predict(data)
		r.set(keys.decode("utf-8"), predict)
