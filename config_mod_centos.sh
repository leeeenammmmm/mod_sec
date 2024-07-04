# Install modsecurity
echo "### Install Modsecurity"
echo

cd ModSecurity && ./autogen.sh && ./configure && make && make install
cp /usr/local/modsecurity/lib/mod_security2.so /etc/httpd/modules/
cd ..

# Install CRS
echo "### Install Core Rules Set"
echo 
rm -rf /usr/share/owasp-modsecurity-crs/
cp -rf owasp-modsecurity-crs/ /usr/share/

# Configure modsecurity
echo "### Configure Modsecurity"
echo
## Copy modsecurity folder to /opt
rm -r -f /opt/modsecurity/
cp -r modsecurity/ /opt/

## Set owner of files
chown root:apache -R /opt/modsecurity/
chown apache:root -R /opt/modsecurity/var/

## Set permission of files
chmod 750 -R /opt/modsecurity/
chmod 640 /opt/modsecurity/etc/*
chcon -R -t httpd_sys_rw_content_t /opt/modsecurity
/usr/sbin/setsebool httpd_can_network_connect=1

echo "LoadModule security2_module modules/mod_security2.so" > /etc/httpd/conf.modules.d/10-mod_security.conf
echo -e '<IfModule !mod_unique_id.c>\n\tLoadModule unique_id_module modules/mod_unique_id.so\n</IfModule>' >> /etc/httpd/conf.modules.d/10-mod_security.conf

echo "Include /opt/modsecurity/etc/modsecurity.conf" > /etc/httpd/conf.d/mod_security.conf
apachectl restart
