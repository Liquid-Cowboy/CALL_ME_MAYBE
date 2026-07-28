import re

sentence = 'Give me the sum of -34.567 and 4.'

numbers = re.compile(r'-?\d+(\.\d+)?')

integers = re.compile(r'-?(?:\d+\.\d+|\d+)')
matches = integers.finditer(sentence)


for match in matches:
	print(match)
