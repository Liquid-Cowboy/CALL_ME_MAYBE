#!/usr/bin/env python3

import re

sentence = 'Give "me" the \"sum\" of -34.567 and 4.'

numbers = re.compile(r'-?\d+(\.\d+)?')

integers = re.compile(r'-?(?:\d+\.\d+|\d+)')
matches = integers.finditer(sentence)

strs = re.findall(r'\b\w+\b', sentence)
strings = re.findall(r'([\'"])\s*(\w+)\s*\1', sentence)

for s in strs:
	print(s)

print()

for s in strings:
	print(s)

print(s)

for s in strs:
	if s not in [s[1] for s in strings]:
		print(s)