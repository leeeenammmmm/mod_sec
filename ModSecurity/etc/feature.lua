LOG_FILE_PATH     = "/opt/modsecurity/var/log/feature.log";
LOG_REQ_PATH      = "/opt/modsecurity/var/log/request.log";

ENTROPY_FILE_PATH = "/opt/modsecurity/var/data/entropy.txt";


-- Count number of keywords in a string or a table contain string
function countKeywords(s)
   local keywords = {"for", "if", "while", "else", "func", "while", "break",
	    "continue", "function", "return", "switch", "var", "select",
	    "from", "union"}

   local count = 0;

   if type(s) == "string" then
      for key, keyword in ipairs(keywords) do
	 for eachMatch in s:gmatch(keyword) do
	    count = count + 1;
	 end
      end
   end

   if type(s) == "table" then
      for i = 1, #s do
	 arg = s[i].value;
	 for k, keyword in ipairs(keywords) do
	    for eachMatch in arg:gmatch(keyword) do
	       count = count + 1;
	    end
	 end
      end
   end

   return count;
end


-- Count number of letters in a string or a table contain string
function countLetters(s)
   local count = 0;

   if type(s) == "string" then
      for i = 1, #s do
	 local c = string.byte(s, i);
	 if (c > 47 and c < 58) or (c > 64 and c < 91) or (c > 96 and c < 123) then
	    count = count + 1;
	 end
      end
   end

   if type(s) == "table" then
      for i = 1, #s do
	 arg = s[i].value;
	 for i = 1, #arg do
	    local c = string.byte(arg, i);
	    if (c > 47 and c < 58) or (c > 64 and c < 91) or (c > 96 and c < 123) then
	       count = count + 1;
	    end
	 end
      end
   end

   return count;
end


-- Count number of cookies in cookie string
function countCookies(cookies)
   local count = 0;
   
   for eachMatch in cookies:gmatch(";") do
      count = count + 1;
   end

   if count == 0 then
      return 0;
   else
      return count + 1;
   end
end


-- Get statistic of all chars appears in request from file
function getStatisticChar()
   local data = {};

   local f = io.open(ENTROPY_FILE_PATH, "r");

   if f == nil then
      data['total'] = 0;
      for i = 32, 176 do
	 data[tostring(i)] = 0;
      end

      return data;
   end

   local total = f:read();
   data['total'] = total;

   for i = 32, 176 do
      data[tostring(i)] = f:read();
   end

   f:close();

   return data;
end


-- Write statistic of all chars to file
function setStatisticChar(data)
   local f = io.open(ENTROPY_FILE_PATH, "w");
   f:write(data['total'] .. '\n');
   
   for i = 32, 176 do
      f:write(data[tostring(i)] .. '\n');
   end

   f:close();
end


-- Count entropy of request
function countEntropy(requestData, totalData)
   local entropy = 0;
   for i = 32, 176 do
      if requestData[tostring(i)] ~= nil then
	 local frequent = totalData[tostring(i)]/totalData['total'];
	 entropy = entropy - frequent * math.log(frequent);
      end
   end

   return string.format("%.2f", entropy);
end


-- Analyze header of request
function analyzeHeader()
   local headerData     = {};
   headerData['reqLen'] = 0;
   local distinctBytes  = {};
   local maxByte = 32;
   local minByte = 127;
   local request = '';
   
   -- Get information of first line include length, distinct bytes,
   -- max byte value and min byte value
   local line    = m.getvar("REQUEST_LINE");
   request       = request .. line .. '\n'
   -- Length
   local lineLen = #line;
   headerData['reqLen'] = headerData['reqLen'] + lineLen;
   
   for i = 1, #line do
      local c = string.byte(line, i);
      -- Max byte
      if c > maxByte then maxByte = c end;
      -- Min byte
      if (c < minByte and c ~= 32) then minByte = c end;
      -- Distinct byte
      if distinctBytes[tostring(c)] == nil then
	 distinctBytes[tostring(c)] = 1;
      else
	 distinctBytes[tostring(c)] = distinctBytes[tostring(c)] + 1;
      end
   end
   
   -- Get information of header field include length, distinct bytes,
   -- max byte value and min byte value
   local header    = m.getvars("REQUEST_HEADERS");
   local headerLen = 0;
   
   for i = 1, #header do
      request = request .. string.match(header[i].name, "REQUEST_HEADERS:(.*)") .. ': ' .. header[i].value .. '\n';
      
      -- Length
      headerLen = headerLen + #header[i].value + #header[i].name - 14;
      
      for j = 1, #header[i].value do
	 local c = string.byte(header[i].value, j);
	 -- Max byte value
	 if c > maxByte then maxByte = c end;
	 -- Min byte value
	 if (c < minByte and c ~= 32) then minByte = c end;
	 -- Distinct byte
	 if distinctBytes[tostring(c)] == nil then
	    distinctBytes[tostring(c)] = 1;
	 else
	    distinctBytes[tostring(c)] = distinctBytes[tostring(c)] + 1;
	 end;
      end

      -- Count length of some header fields of request
      if string.find(header[i].name, "Accept[-]Encoding") then
	 headerData["Accept-Encoding"] = #header[i].value;
      end

      if string.find(header[i].name, "Accept[-]Language") then
	 headerData["Accept-Language"] = #header[i].value;
      end

      if string.find(header[i].name, "Content[-]Length") then
	 headerData["Content-Length"] = #header[i].value;
      end

      if string.find(header[i].name, "Host") then
	 headerData["Host"] = #header[i].value;
      end
      
      if string.find(header[i].name, "Accept$") then
	 headerData["Accept"] = #header[i].value;
      end

      if string.find(header[i].name, "User[-]Agent") then
	 headerData["User-Agent"] = #header[i].value;
      end

      if string.find(header[i].name, "Accept[-]Charset") then
	 headerData["Accept-Charset"] = #header[i].value;
      end

      if string.find(header[i].name, "Cookie") then
	 headerData["Cookie"]  = #header[i].value;
	 -- Count number of cookies 
	 headerData["cookies"] = countCookies(header[i].value);
      end

      if string.find(header[i].name, "Content[-]Type") then
	 headerData["Content-Type"] = #header[i].value;
      end

      if string.find(header[i].name, "Referer") then
	 headerData["Referer"] = #header[i].value;
      end
      
   end

   request = request .. '\n';
   
   -- Plus header length to request length
   headerData['reqLen'] = headerData['reqLen'] + headerLen;

   -- Get method of request
   local method = m.getvar("REQUEST_METHOD");
   headerData['method'] = method;
   
   -- Get body data about length, distinct bytes, max byte value and min byte
   -- value in body of request
   local body    = m.getvar("REQUEST_BODY");
   local bodyLen = 0;

   if body ~= nil then
      request = request .. body .. '\n\n';
      
      -- Length
      bodyLen = #body;
      for i = 1, #body do
	 local c = string.byte(body, i);
	 -- Max byte
	 if c > maxByte then maxByte = c end;
	 -- Min byte
	 if (c < minByte and c ~= 32) then minByte = c end;
	 -- Distinct byte 
	 if distinctBytes[tostring(c)] == nil then
	    distinctBytes[tostring(c)] = 1;
	 else
	    distinctBytes[tostring(c)] = distinctBytes[tostring(c)] + 1;
	 end;
      end
   end

   -- Count number of distinct bytes in request
   local numberDistinctBytes = 0;
   local statisticChar       = getStatisticChar();

   for i = 32, 176 do
      if distinctBytes[tostring(i)] ~= nil then
         numberDistinctBytes        = numberDistinctBytes + 1;
         statisticChar[tostring(i)] = statisticChar[tostring(i)] + distinctBytes[tostring(i)];
         statisticChar['total']     = statisticChar['total'] + distinctBytes[tostring(i)];
      end
   end
   headerData['distinctBytes'] = numberDistinctBytes;

   headerData['entropy'] = countEntropy(distinctBytes, statisticChar); 
   
   if statisticChar['total'] < 1000000 then
      setStatisticChar(statisticChar); 
   end
   
   -- Max and min byte value in request
   headerData['maxByte']       = maxByte;
   headerData['minByte']       = minByte;

   -- Total length of request
   headerData['reqLen'] = headerData['reqLen'] + bodyLen;

   -- Log request
   local f = io.open(LOG_REQ_PATH, "a");
   f:write(request);
   f:close();
   
   -- Return header data
   return headerData;
end


function analyzePath()
   local pathData = {};
   
   local reqURI = m.getvar("REQUEST_URI");
   
   -- Get length of file path
   pathData['length'] = #reqURI;
   
   -- Count number of digits in the path
   local numberOfDigits = 0;
   
   for i = 1, #reqURI do
      local c = string.byte(reqURI, i);
      if c > 47 and c < 58 then
	 numberOfDigits = numberOfDigits + 1;
      end
   end

   -- Number of digits and other char in the path
   pathData['digits'] = numberOfDigits;
   pathData['other']  = #reqURI - numberOfDigits;

   -- Count number of keywords in the path
   pathData['keywords'] = countKeywords(reqURI);
   -- Count number of letters char in the path 
   pathData['letters']  = countLetters(reqURI);
   -- Count number of special char in the path 
   pathData['special']  = pathData['length'] - pathData['letters'];
   
   -- Return path data
   return pathData;
end


-- Analyze arguments attribute including POST and GET
function analyzeArgs()
   local argsData = {};
   
   local args           = m.getvars("ARGS");
   local number         = #args;
   local totalLen       = 0;
   local numberOfDigits = 0;
   
   -- Number of arguments
   argsData['number'] = number;

   -- Get total length of all arguments and number of digits in those arguments
   for i = 1, number do
      arg = args[i].value;
      -- Total length
      totalLen = totalLen + #arg;

      -- Count number of digits
      for j = 1, #arg do
	 local c = string.byte(arg, j);
      	 if c > 47 and c < 58 then
      	    numberOfDigits = numberOfDigits + 1;
	 end
      end
   end

   -- Count number of keywords in the arguments
   argsData['keywords'] = countKeywords(args);
   -- Count number of letter char in the arguments
   argsData['letters']  = countLetters(args);
   -- Count number of special char in the arguments
   argsData['special']  = totalLen - argsData['letters'];

   -- Calculate average length of all arguments
   if totalLen == 0 then
      argsData['avg'] = 0;
   else
      argsData['avg'] = string.format("%.2f", totalLen/number);
   end

   -- Number of digits and other char in arguments
   argsData['digits'] = numberOfDigits;
   argsData['other']  = totalLen - numberOfDigits;
   
   -- Return arguments data 
   return argsData
end


function main()   
   
   -- Parse header and parameters data
   local headerData = analyzeHeader();
   local pathData   = analyzePath();
   local argsData   = analyzeArgs();

   -- Open log file
   f = io.open(LOG_FILE_PATH, "a");

   -- Start logging
   f.write(f, 
   
   -- Log length of the request
   headerData['reqLen'] .. ' '

   -- Log average length of arguments 
   .. argsData['avg'] .. ' '

   -- Log length of the header "Accept-Encoding"
   .. (headerData['Accept-Encoding'] or 0) .. ' '

   -- Log length of the header "Accept-Language"
   .. (headerData['Accept-Language'] or 0).. ' '

   -- Log length of the header "Content-Length"
   .. (headerData['Content-Length'] or 0) .. ' '

   -- Log length of the Host
   .. (headerData['Host'] or 0) .. ' '

   -- Log length of the header "User-Agent"
   .. (headerData['User-Agent'] or 0) .. ' '

   -- Log length of the path
   .. (pathData['length'] or 0) .. ' '

   -- Log length of the header "Accept"
   .. (headerData['Accept'] or 0) .. ' '

   -- Log length of the header "Accept-Charset"
   .. (headerData['Accept-Charset'] or 0) .. ' '

   -- Log length of the header "Cookie"
   .. (headerData['Cookie'] or 0) .. ' '

   -- Log length of the header "Content-Type"
   .. (headerData['Content-Type'] or 0) .. ' '

   -- Log length of the header "Referer"
   .. (headerData['Referer'] or 0) .. ' '

   -- Log method identifier
   .. (headerData['method'] or 'Unknown') .. ' '

   -- Log number of arguments
   .. (argsData['number'] or 0) .. ' '

   -- Log number of digits in the arguments
   .. (argsData['digits'] or 0) .. ' '

   -- Log number of other char in the arguments
   .. (argsData['other'] or 0) .. ' '

   -- Log number of digits in the path
   .. (pathData['digits'] or 0) .. ' '

   -- Log number of other char in the path
   .. (pathData['other'] or 0) .. ' '
	   
   -- Log number of distinct bytes in request
   .. (headerData['distinctBytes'] or 0) .. ' '

   -- Log number of keywords in the path
   .. (pathData['keywords'] or 0) .. ' '

   -- Log number of keywords in the arguments
   .. (argsData['keywords'] or 0) .. ' '

   -- Log number of letters char in the arguments
   .. (argsData['letters'] or 0) .. ' '

   -- Log number of special char in the arguments
   .. (argsData['special'] or 0) .. ' '

   -- Log number of letters char in the path
   .. (pathData['letters'] or 0) .. ' '

   -- Log number of special char in the path
   .. (argsData['special'] or 0) .. ' '

   -- Log number of cookies
   .. (headerData['cookies'] or 0) .. ' '

   -- Log entropy
   .. (headerData['entropy'] or 0) .. ' '
   
   -- Log max byte value
   .. (headerData['maxByte'] or 0) .. ' '

   -- Log min byte value
   .. (headerData['minByte'] or 0) .. ' '
   
   -- Terminate line
   .. "\n");
   f.close(f);
end
