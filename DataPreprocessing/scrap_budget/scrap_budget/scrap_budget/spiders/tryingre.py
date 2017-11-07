from decimal import Decimal
from re import sub

mon = '''($70 million)

'''
oldmon = mon
mon = mon.replace('–','-')
dashloc = mon.find('-')
print(dashloc)
if dashloc > 0:
    mon = mon[0:dashloc]

value = Decimal(sub(r'[^-?\d.]', '', mon))
if 'million' in oldmon.lower():
    value = value * 10**6

print(value)