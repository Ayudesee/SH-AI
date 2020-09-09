import win32api as wapi
import time

keyList = [40, 65, 68]
# keyList = ["\b"]
# for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890,.'APS$/\\":
#     keyList.append(char)
#
#
# def key_check():
#     keys = []
#     for key in keyList:
#         if wapi.GetAsyncKeyState(ord(key)):
#             keys.append(key)
#     return keys

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(key):
            keys.append(chr(key))
    return keys

