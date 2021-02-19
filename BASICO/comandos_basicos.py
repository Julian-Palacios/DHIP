import time
import sys,os,codecs
import matplotlib.pyplot as plt
from numpy import zeros, identity, array, save, savetxt
from numpy.linalg import inv

# ################       Variables       ###############
# a=3+5; A=2*6; b=2**3
# B=2.6/3.5
# α,β,Φ = 0.22,0.33,0.9
# η = 3.1416
# # print(a,A,b,B,Φ,α,β,η)
# print('B=%10.4f\n'%B)
# print("a=%i\nA=%s\nb=%.3d\nB=%10.5f\nΦ,α,β=%s,%s,%s"%(a,
# A,b,B,Φ,α,β))

# ###############    Condicionales   ###############
# A=5
# if A>2:
#   print("A es mayor que 2")
# elif A==2:
#   print("A es igual a 2")
# elif A<2 and A>=0:
#   print("A es menor que 2, pero positivo")
# else:
#   print("A es menor que 2 y negativo")

# ###############        Bucles       ###############
# B="PYTHON\n"
# i=0
# while i<6:
#   print(B[i])
#   i=i+1
# #
# print("\n")
# for e in B:
#   print(e)
# #
# [print(e) for e in B]
# #
# ############  Ejemplo con Try Except   ############
lista=[]
n=10
inicio = time.time()
while True and n>-10:
    # if n==0:
    #     print("Ojo n es igual a 0")
    #     break
    try:
        div=10/n
    except Exception as e:
        print(e)
        n=n-1
        continue
    # print("La división es: %7.5f"%div)
    lista.append([n,div])
    n=n-1
    # time.sleep(1)
print("Demoró: %6.3f segundos"%(time.time()-inicio))

############### Lectura de Archivos ###############
print(__file__)
os.chdir(os.path.dirname(__file__))
# # file1 = open('./data/test.txt',"r")
# file1 = codecs.open('./data/test.txt',"r","utf-8")
# texto = file1.readline()
# texto2 = file1.readline()
# file1.close()
# print(texto)
# print(texto.split())
# # #
# # file2 = open('./results/out.txt',"w")
# file2 = codecs.open('./results/out.txt',"w","utf-8")
# file2.write(texto)
# file2.write(texto2)
# file2.close()
#
################      FUNCIONES       #################
# def countSpaces(text='texto'):
#     '''
#     Esta es una función que cuenta espacios.
#     '''
#     j=0
#     for e in texto:
#         if e == ' ':
#             j=j+1
#     return j
# #
# texto='Hola JPI Ingeniería e Innovación'
# x=countSpaces(texto)
# print("Número de espacios en\n|%s|: %i"%(texto,x))
#
# #################      Clase      #####################
# class datos:
#     tiempo=60
#     amplitud=10
#     def countSpaces(text='texto'):
#         j=0
#         for i in text:
#             if i==' ':
#                 j=j+1
#         datos.espacios=j
  
# print(datos.tiempo)
# print(datos.amplitud)
# texto='Hola JPI Ingeniería e Innovación'
# datos.countSpaces(texto)
# print(datos.espacios)

# #################    Diccionario     #################
# Capital = dict(
#   Perú='Lima',
#   Bolivia='Sucre',
#   Ecuador='Quito'
# )
# print(Capital["Ecuador"])

##################     MATRICES    ####################
# Z = zeros((5,2))+1.0
# print(Z,'\n',Z*1.2,'\n',Z/0.5)

# I5 = identity(5)
# print(I5)
# print(I5@Z)
# print(inv(I5*2)@Z)
##
# from random import seed
# from random import random
# #
# N=500
# A=zeros((N,N),dtype='float32')
# seed(1)
# for i in range(N):
#   for j in range(N):
#     if i%5==0 and j%5==0:
#       A[i,j]=random()
# print(A[0,0])
# save('./results/Matriz_Numpy.npy',A)
# savetxt('./results/Matriz_Numpy.txt',A)
# print(A)

# ##################      PLOTEO       ####################
M=array(lista)
x=M[:,0]
y=M[:,1]
plt.figure(figsize=(8,4))
plt.plot(x,y,'r--',lw=2,label='Curva Roja')
plt.plot(x,y,'k',alpha=0.5,label='Curva Negra')
plt.axis([-10,10,-20,20])
plt.legend()
plt.show()

