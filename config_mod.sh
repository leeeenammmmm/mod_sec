# Install modsecurity
echo "### Install Modsecurity"
echo

cd ModSecurity && ./autogen.sh && ./configure && make && make install
cp /usr/local/modsecurity/lib/mod_security2.so /usr/lib/apache2/modules/
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
chown root:www-data -R /opt/modsecurity/
chown www-data:root -R /opt/modsecurity/var/

## Set permission of files
chmod 750 -R /opt/modsecurity/
chmod 640 /opt/modsecurity/etc/*

echo "LoadFile libxml2.so.2" > /etc/apache2/mods-available/security2.load
echo "LoadModule security2_module /usr/lib/apache2/modules/mod_security2.so" >> /etc/apache2/mods-available/security2.load
echo "Include /opt/modsecurity/etc/modsecurity.conf" > /etc/apache2/mods-available/security2.conf

a2enmod security2
service apache2 restart