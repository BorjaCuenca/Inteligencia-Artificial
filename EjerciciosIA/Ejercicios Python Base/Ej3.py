
print("Introduce el intervalo de numeros (separados por ','): ")
valores = input().split(',')
a = int(valores[0])
b = int(valores[1])
esPrimo = True

for r in range(a,b+1):
    esPrimo = True
    for n in range (2,r):
        if r%n == 0:
            esPrimo = False
    if esPrimo:
        print(r)