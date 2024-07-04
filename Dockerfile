# set base image (host OS)
FROM ubuntu:18.04

RUN apt-get -y update
RUN apt-get -y install redis-server

RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install setuptools
RUN pip3 install redis
RUN pip3 install keras
RUN pip3 install tensorflow
RUN pip3 install pandas

RUN apt-get -y install lua5.1 liblua5.1-0
RUN apt-get -y install luarocks
RUN luarocks install redis-lua sha1

RUN apt-get -y install automake autoconf m4 libtool
RUN apt-get -y install apache2-dev libxml2-dev

COPY config_model.sh /opt/src/scripts/config_model.sh
COPY config_mod.sh /opt/src/scripts/config_mod.sh
