#!/usr/bin python
# -*- coding: utf-8 -*-

import smtplib
import netrc
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from os.path import basename
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s -'
                                                ' %(message)s')
# logging.disable()


def send_email(smtpserver, fromaddr, toaddrs, subject, body,
               att_file=None, tls=True):
    """
    :param smtpserver:
        Smtp server, e.g. `smtp.gmail.com`
    :param fromaddr:
        Address from.
    :param toaddrs:
        List of receipt addresses.
    Note that by default, this looks to check your netrc credentials
    to use this feature, create a .netrc file, so that only you can read and
    write it

        touch ~/.netrc
        chmod 600 ~/.netrc
    and then add the information for the gmail smtp server, i.e.
    ``machine smtp.gmail.com login yourusername@gmail.com password
    yourpassword``
    """
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddrs
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    if att_file is not None:
        with open(att_file) as fil:
            msg.attach(MIMEApplication(fil.read(),
                                       Content_Disposition='attachment;'
                                                           ' filename="%s"' %
                                                           basename(att_file),
                                       Name=basename(att_file)))
    s = smtplib.SMTP(smtpserver)
    secrets = netrc.netrc()
    netrclogin, netrcaccount, netrcpassword = secrets.authenticators(smtpserver)
    if tls:
        s.starttls()
        s.login(netrclogin, netrcpassword)
    s.sendmail(fromaddr, toaddrs,
               msg.as_string())
    s.quit()


if __name__ == '__main__':
    pass
