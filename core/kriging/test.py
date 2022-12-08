class A:
    def __init__(self, a, c= "e"):
        self.a=a
        self.c=c
    
class B(A):
    def __init__(self, b="b",*argv,**kargv):
        super().__init__(*argv,**kargv)
        self.b=b

        

        
b=B('1','2', c = '3')
print(b.a)
print(b.b)
print(b.c)

t = {}
t[1] = 1
t[1] += 12
print(t)
