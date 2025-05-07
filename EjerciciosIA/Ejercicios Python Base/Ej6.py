
string1 = input("Introduce una cadena de caracteres: ")
string2 = input("Introduce la segunda cadena: ")

characters1 = set(string1)
characters2 = set(string2)

"""
characters1 = set()
characters2 = set()

for c in string1:
    characters1.add(c)
    
for c in string2:
    characters2.add(c)
"""

print (characters1.difference(characters2))