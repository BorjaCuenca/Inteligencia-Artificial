print("Introduce el nombre del producto: ")
nombre = input()
print("Introduce el precio por unidad: ")
precio = float(input())
print("Introduce el numero de unidades: ")
numero = float(input())

print(nombre+ ": " + str(numero) + " x " + str(precio) + "€ = " + str(numero*precio))
