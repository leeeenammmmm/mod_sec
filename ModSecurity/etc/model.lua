
function passModel(model_string)
   local sha1 = require 'sha1'
   local hash_as_hex   = sha1(model_string)

   local redis = require 'redis'
   local client = redis.connect('127.0.0.1', 6379)

   local response = client:ping()           -- true
   -- m.log(3, model_string)
   local value = client:get(hash_as_hex)
   if value == nil or (string.match(value, ".") ~= '0') then
   -- if 1 > 0 then		
      client:rpush("myRequestQueue",hash_as_hex)
      client:set(hash_as_hex, model_string)
      value = client:get(hash_as_hex)
      while(value == model_string)
      do
	      value = client:get(hash_as_hex)
      end
   end

   return value
end

function getModelString()
   local model_string;
   local method = m.getvar("REQUEST_METHOD");
   local req_uri = m.getvar("REQUEST_URI");
   model_string = method .. req_uri;
   
   local query_str = m.getvar("QUERY_STRING");
   if query_str ~= nil then
      model_string = model_string .. query_str;
   end

   if method == "POST" or method == "PUT" then
      local body = m.getvar("REQUEST_BODY");
      if body ~= nil then
	 model_string = model_string .. " " .. body;
      end
   end

   return model_string;
end

function getString()
   local model_string = m.getvar("REQUEST_URI");
   local query_str = m.getvar("QUERY_STRING");
   if query_str ~= nil then
      model_string = model_string .. query_str;
   end

   if method == "POST" or method == "PUT" then
      local body = m.getvar("REQUEST_BODY");
      if body ~= nil then
	 model_string = model_string .. " " .. body;
      end
   end

   return model_string;
end

function decide(anomaly_score, model_score)
   local crs_weight = 0.5;
   local model_weight = 0.5;
   local crs_score;

   if anomaly_score < 5 then
      crs_score = 0.3;
   elseif anomaly_score < 10 then
      crs_score = 0.5;
   else
      crs_score = 0.8;
   end
   
   local result = crs_score * crs_weight + model_score * model_weight;

   m.log(3, "Decided result: " .. result);

   if result > 0.5 then
      return true;
   end
   
   return false;
end

function main()

   local anomaly_score = m.getvar("TX.ANOMALY_SCORE");
   local model_string = getString();
   m.log(3, "Web Attack Detection System using Deep Learning and NLP Activated!!!");

   local model_result = passModel(model_string);

   if model_result ~= "000 - Normal" then
      return "Detected Web Attack Request!!! Web Attack Type: " .. model_result;
   end
   return nil

end




   

