import ssl
from functools import reduce
from operator import truediv

from charset_normalizer.cd import encoding_unicode_range
from sqlalchemy import Column, String, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sqlite3, os

from demo.memory.profile import ENUM_MARKET

res = 'B'
import math

match res:
    case 'B':
        print(res, 1)
    case 'A':
        print(res + 1231)
# sum = 0
# for i in range(101):
#     sum = sum + i
# print(sum)
L = ['Bart', 'Lisa', 'Adam']
for l in L:
    print(l)
# while sum > 5040:
#     if (sum < 5040):
#         break
#     sum = sum - 1
#     print(sum)
dic = {"1": 2, 2: "4"}
for k, v in dic.items():
    print(k, v)
print(bool())
print(str(100))
print(str(-100))
print(int(199))
print(float(199))
print(hex(199))


def custom_max(a, b, c, d):
    if a > b:
        return a
    else:
        return b


print(custom_max(1, 2, 3, 4))


def quadratic(a, b, c):
    return (-b + math.sqrt(b * b - 4 * a * c)) / 2, (-b - math.sqrt(b * b - 4 * a * c)) / 2


x, y = quadratic(1, -3, 2)
print(x, y)
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def custom(*nums):
    for n in nums:
        print(n)


custom(nums)


def custom1(**nums):
    for x, y in nums.items():
        print(x, y)


custom1(key=1, key1=2, key3=3, key4=4)


def mul(*z):
    res = 1
    for z1 in z:
        res = z1 * res
    return res


print(mul(1, 2, 3, 4))


def trim(ss):
    start = 0
    end = len(ss)
    s = False
    e = False
    print(s, e, ss)
    for c in ss:
        if c == ' ' and s == False:
            start = start + 1
        elif c != ' ':
            s = True
        if ss[end - 1] == ' ' and e == False:
            end = end - 1
        elif ss[end - 1] != ' ':
            e = True
        if s and e:
            break
    print(start, end)
    return ss[start:end]


print(trim("  123"))


def maxormin(*nums):
    if not nums:
        return None, None
    min_ = max_ = nums[0]
    for n in nums[1:]:
        if n < min_:
            min_ = n
        elif n > max_:
            max_ = n
    return min_, max_


x, y = maxormin(1, 3, 2, 412, 41, 14, 12)
print(x, y)
b = [x * 2 for x in range(1, 10) if x % 2 == 0]
print(b)


def fn(x):
    return x * 2


print(list)


def fn2(a, b):
    return a + b


print(reduce(fn2, range(1, 10)))
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
dbfile = os.path.join(os.path.expanduser('~'), 'my.db')
if os.path.exists(dbfile):
    os.remove(dbfile)
oc = sqlite3.connect(dbfile)
occ = oc.cursor()
occ.execute('create table user(id varchar(20) primary key, name varchar(20), score int)')
occ.execute(r"insert into user values ('A-001', 'Adam', 95)")
occ.execute(r"insert into user values ('A-002', 'Bart', 62)")
occ.execute(r"insert into user values ('A-003', 'Lisa', 78)")
oc.commit()
occ.close()


# oc.close()
def get_score_in(low, high):
    occ_ = oc.cursor()
    occ_.execute('select name from user where score>=? and score<=? order by score asc ', (str(low), str(high)))
    return [row[0] for row in occ_.fetchall()]


assert get_score_in(80, 95) == ['Adam'], get_score_in(80, 95)
assert get_score_in(60, 80) == ['Bart', 'Lisa'], get_score_in(60, 80)
assert get_score_in(60, 100) == ['Bart', 'Lisa', 'Adam'], get_score_in(60, 100)
oc.close()
print('pass')
# Base = declarative_base()
# class UserTest(Base):
#     __tablename__ = 'usertest'
#     id = Column(Integer, primary_key=True)
#     name = Column(String(20))
#     score = Column(Integer)
# engine = create_engine('sqlite:///' + dbfile)
# Base.metadata.create_all(engine)        # ← 建表 (没有这行就 no such table)
# DBSession = sessionmaker(bind=engine)
# session = DBSession()
# user = UserTest(id=1, name='Adam', score=100)
# session.add(user)
# session.commit()
# res = session.query(UserTest).filter(UserTest.score >90).all()
# print(res)
# print([(u.id, u.name, u.score) for u in res])   # 看清楚字段值
from email import encoders
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr


# import smtplib
# def _format_addr(s):
#     name, addr = parseaddr(s)
#     return formataddr((Header(name, 'utf-8').encode(), addr))
# from_addr = "jineefo@163.com"
# password = "TVSyLtZTQEvpPZk7"
# to_addr = "jineefo2@gmail.com"
# smtp_server = "smtp.163.com"
#
# msg = MIMEText('hello, send by Python...', 'plain', 'utf-8')
# msg['From'] = _format_addr('Python爱好者 <%s>' % from_addr)
# msg['To'] = _format_addr('管理员 <%s>' % to_addr)
# msg['Subject'] = Header('来自SMTP的问候……', 'utf-8').encode()
#
# server = smtplib.SMTP_SSL(smtp_server, 465)
# server.set_debuglevel(1)
# server.login(from_addr, password)
# server.sendmail(from_addr, [to_addr], msg.as_string())
# server.quit()
# import socket
# HOST = "www.sina.com.cn"
# PORT = 443
#
# raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# ctx = ssl.create_default_context()
# s = ctx.wrap_socket(raw, server_hostname=HOST)     # ← TLS 握手 SNI 必须填
# s.connect((HOST, PORT))
#
# request = (
#     "GET / HTTP/1.0\r\n"
#     f"Host: {HOST}\r\n"
#     "Connection: close\r\n"
#     "\r\n"
# ).encode()
# s.send(request)
# buffer=[]
# while True:
#     data=s.recv(1024)
#     if not data:
#         break
#     else:
#         buffer.append(data)
# data=b''.join(buffer)
# s.close()
# header,html=data.split(b'\r\n\r\n',1)
# print(header.decode('utf-8'))
# with open('163.html','wb') as f:
#     f.write(html)
class Student(object):
    _score = None
    item = None

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    def __getattr__(self, item):
        return item


stu = Student()
stu.score = 100
print(stu.score)
from enum import Enum, unique

Car = Enum('Car', ('auti', 'benci'))
print(Car.auti)
for x, y in Car.__members__.items():
    print(x, y, y.value)


@unique
class weekday(Enum):
    MON = 0
    TUE = 1


for x, y in weekday.__members__.items():
    print(x, y, y.value)


@unique
class Gender(Enum):
    MAN = 1
    WOMAN = 2


def hello(self):
    print(f'Hello')


Hello = type('Hello', (object,), {'hello1': hello})
#                                     ^^^^^ ^^^^^
#                                     属性名  绑定的函数对象(不加括号!)

h = Hello()
h.hello1()
# import unittest
#
# class Student(object):
#     def __init__(self, name, score):
#         self.name = name
#         self.score = score
#
#     def get_grade(self):
#         if self.score >= 80 and self.score <= 100:
#             return 'A'
#         if self.score >= 60 and self.score <80:
#             return 'B'
#         if self.score >= 0 and self.score < 60:
#             return 'C'
#         raise ValueError
#
# class TestStudent(unittest.TestCase):
#     def test_80_to_100(self):
#         s1 = Student('Bart', 80)
#         s2 = Student('Lisa', 100)
#         self.assertEqual(s1.get_grade(), 'A')
#         self.assertEqual(s2.get_grade(), 'A')
#
#     def test_60_to_80(self):
#         s1 = Student('Bart', 60)
#         s2 = Student('Lisa', 79)
#         self.assertEqual(s1.get_grade(), 'B')
#         self.assertEqual(s2.get_grade(), 'B')
#
#     def test_0_to_60(self):
#         s1 = Student('Bart', 0)
#         s2 = Student('Lisa', 59)
#         self.assertEqual(s1.get_grade(), 'C')
#         self.assertEqual(s2.get_grade(), 'C')
#
#     def test_invalid(self):
#         s1 = Student('Bart', -1)
#         s2 = Student('Lisa', 101)
#         with self.assertRaises(ValueError):
#             s1.get_grade()
#         with self.assertRaises(ValueError):
#             s2.get_grade()
# if __name__ == '__main__':
#     unittest.main()
# with open('./requirements-dev.txt', 'a',encoding='utf-8',errors='ignore') as f:
#     print(f.write("13"))
from io import BytesIO, StringIO
str=StringIO()
str.write('12312321')
print(str.getvalue())
dirs=[x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x)[1]=='.txt']
print(dirs)
# os.listdir(".").append()
stu=Student()
stu.score=11;
import json
print( json.dumps(stu, default=lambda obj: obj.__dict__))
import subprocess
r=subprocess.call(['nslookup', 'www.python.org'])
print(r)

from urllib import request
import json
def fetch_data(url):
    with request.urlopen(url) as response:
        return json.loads(response.read())

# 测试
URL = 'https://api.weatherapi.com/v1/current.json?key=b4e8f86b44654e6b86885330242207&q=Beijing&aqi=no'
data = fetch_data(URL)
print(data)
assert data['location']['name'] == 'Beijing'
print('ok')
from contextlib import contextmanager
@contextmanager
def heool():
    print('start')
    yield
    print('end')
with heool():
    print('ok')
from itertools import count, takewhile, islice


# def pi(n):
#     odds=takewhile(lambda x: x<2*n+1, count(1,2))
#     return sum((-1) ** i / d for i, d in enumerate(odds))
def pi(N):
    # 生成分母序列：1, 3, 5, 7, ... 取前 N 项
    odds = takewhile(lambda x: x < 2 * N, count(1, 2))
    # 交替正负求和
    total = sum((-1) ** i / d for i, d in enumerate(odds))
    return total * 4
def triangles():
    row = [1]
    while True:
        yield row
        # 由上一行生成下一行：两端补 0 后相邻相加
        row = [a + b for a, b in zip([0] + row, row + [0])]
for t in islice(triangles(), 10):
    print(t)
l1=[0,1,2,3]
l2=[4,5,6,7,9]
l3=[8,9,10,11,12]
print([a*b+c for a,b,c in zip(l1,l2,l3)])


