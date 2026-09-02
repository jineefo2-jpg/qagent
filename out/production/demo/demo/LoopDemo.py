from functools import reduce
from operator import truediv
from sqlalchemy import Column,String,create_engine
import sqlite3,os
res='B'
import math
match res:
    case 'B':
        print(res,1)
    case 'A':
        print(res+1231)
sum=0
for i in range(101):
    sum=sum+i
print(sum)
L = ['Bart', 'Lisa', 'Adam']
for l in L:
    print(l)
while sum>5040:
    if(sum<5040):
        break
    sum=sum-1
    print(sum)
dic={"1":2,2:"4"}
for k,v in dic.items():
    print(k,v)
print(bool())
print(str(100))
print(str(-100))
print(int(199))
print(float(199))
print(hex(199))
def custom_max(a,b,c,d):
    if a>b:
        return a
    else:
        return b
print(custom_max(1,2,3,4))
def quadratic(a,b,c):
    return (-b+math.sqrt(b*b-4*a*c))/2,(-b-math.sqrt(b*b-4*a*c))/2
x,y=quadratic(1,-3,2)
print(x,y)
nums=[1,2,3,4,5,6,7,8,9]
def custom(*nums):
    for n in nums:
        print(n)
custom(nums)
def custom1(**nums):
    for x,y in nums.items():
        print(x,y)
custom1(key=1,key1=2,key3=3,key4=4)
def mul(*z):
    res=1
    for z1 in z:
        res=z1*res
    return res
print(mul(1,2,3,4))
def trim(ss):
    start = 0
    end=len(ss)
    s=False
    e=False
    print(s,e,ss)
    for c in ss:
        if c==' ' and s==False:
            start=start+1
        elif c!=' ':
            s=True
        if ss[end-1]==' ' and e==False:
            end=end-1
        elif ss[end-1]!=' ':
            e=True
        if s and e:
            break
    print(start,end)
    return ss[start:end]
print(trim("  123"))
def maxormin(*nums):
    if not nums:
        return None,None
    min_ = max_ = nums[0]
    for n in nums[1:]:
        if n<min_:
            min_=n
        elif n>max_:
            max_=n
    return min_,max_
x,y=maxormin(1,3,2,412,41,14,12)
print(x,y)
b=[x*2 for x in range(1,10) if x%2==0]
print(b)
def fn(x):
    return x*2
print(list)
def fn2(a,b):
    return a+b
print(reduce(fn2,range(1,10)))
# conn=sqlite3.connect('my.db')
# cur=conn.cursor()
# # cur.execute('create table user(id varchar(20) primary key, name varchar(20))')
# # cur.execute('insert into user values(?,?)',('1','admin'))
# cur.execute('select * from user where id=?',(1,))
# values=cur.fetchall()
# print(values)
# print(f'数据库插入：{cur.rowcount}')
# conn.commit()
# cur.close()
# conn.close()
dbfile=os.path.join(os.path.expanduser('~'),'my.db')
if os.path.exists(dbfile):
    os.remove(dbfile)
oc=sqlite3.connect(dbfile)
occ=oc.cursor()
occ.execute('create table user(id varchar(20) primary key, name varchar(20), score int)')
occ.execute(r"insert into user values ('A-001', 'Adam', 95)")
occ.execute(r"insert into user values ('A-002', 'Bart', 62)")
occ.execute(r"insert into user values ('A-003', 'Lisa', 78)")
oc.commit()
occ.close()
# oc.close()
def get_score_in(low, high):
    occ_=oc.cursor()
    occ_.execute('select name from user where score>=? and score<=? order by score asc ', (str(low), str(high)))
    return [row[0] for row in occ_.fetchall()]
assert get_score_in(80, 95) == ['Adam'], get_score_in(80, 95)
assert get_score_in(60, 80) == ['Bart', 'Lisa'], get_score_in(60, 80)
assert get_score_in(60, 100) == ['Bart', 'Lisa', 'Adam'], get_score_in(60, 100)
oc.close()
print('pass')