# -*- coding: utf-8 -*-

""" Global parameters ans settings """
import socket
if socket.gethostname() == "oshwork":
    path = "/home/osh/frb_test"
else:
    path = "/mnt/frb_data"  # Working  path

config = {
"raw_path"   : path + "/raw_data", # raw data path
"aspec_path" : path + "/aspec",    # path to autospectra
"cfx_path"   : path + "/cfx"       # path to cfx files
}

#stations = {
#"Ar" : ""
#}