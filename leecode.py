# # 从用户输入中读取两个整数
# # input_string = input("请输入两个整数，以空格分隔: ")
# a, b = input().split(' ')
# a = int(a)
# b = int(b)
#
# # 打印读取的两个整数
# print(a+b)
##todo
# import sys
# N=int(input())
# for i in range (N):
#     for line in sys.stdin:
#         a,b=line.split(' ')
#         print(int(a)+int(b))
# ##TODO
# import sys
# import numpy as np
# # for line in sys.stdin:
# #     n,m=line.split(' ')
# # n, m = input().split(' ')
# # n=int(n)
# # m=int(m)
# n,m=3,4
# # n,m=input().split(' ')
# # nums = [[0] * n for _ in range(n)]
# nums=np.zeros((n ,m))
# print(n,m,nums)
# startx,starty=int((n-1)/2),int(m/2)
#
# loop, midx,midy = n // 2, n // 2,m//2
# count = 1
# for offset in range(1, loop + 1) :      # 每循环一层偏移量加1，偏移量从1开始
#     for i in range(starty, -1, -1) : # 从右至左
#                 nums[startx][i] = count
#                 count += 1
#
#     for i in range(startx+1, n, 1) : # 从上至下
#                 nums[i][starty-midy]= count
#                 count += 1
#
#
#
#     for i in range(1, m-offset) :    # 从左至右，左闭右开
#                 nums[startx+offset][i] = count
#                 count += 1
#     for i in range(startx, n - offset) :    # 从上至下
#                 nums[i][n - offset] = count
#                 count += 1
#     for i in range(n - offset, starty, -1) : # 从右至左
#                 nums[n - offset][i] = count
#                 count += 1
#     for i in range(n - offset, startx, -1) : # 从下至上
#                 nums[i][starty] = count
#                 count += 1
#     startx += 1         # 更新起始点
#     starty += 1
#
# if n % 2 != 0 :			# n为奇数时，填充中心点
#             nums[mid][mid] = count
# print(nums)
# ##TODO
# while 1:
#     try:
#         N = int(input())
#         for i in range(N):
#             l = list(map(int,input().split()))
#             print(sum(l))
#     except:
#         break
# # ##TODO
# while(1):
#     try:
#         L=list(map(int,input().split(' ')))
#         if L[0]==0 or L[0]<(len(L)-1):
#             break
#         else:
#             print(sum(L[1:(L[0]+1)]))
#     except:
#         break

##todo
# #TODO
# while(1):
#     try:
#         N=int(input())
#         for i in range(N):
#             L=list(map(int,input().split(' ')))
#             S=sum(L[1:(L[0]+1)])
#             print(S)
#             print()
#     except:
#         break

# ##TODO
# ''''
# 题目描述
# 每门课的成绩分为A、B、C、D、F五个等级，为了计算平均绩点，规定A、B、C、D、F分别代表4分、3分、2分、1分、0分。
# 输入
# 有多组测试样例。每组输入数据占一行，由一个或多个大写字母组成，字母之间由空格分隔。
# 输出
# 每组输出结果占一行。如果输入的大写字母都在集合｛A,B,C,D,F｝中，则输出对应的平均绩点，结果保留两位小数。否则，输出“Unknown”
# '''
# T = ['A', 'B', 'C', 'D', 'F']
# while (1):
#     try:
#
#         s = list(map(str, input().split(' ')))
#         Sum = 0
#         count = 0
#         for i in range(len(s)):
#
#                 if s[i] == 'A':
#                     v = 4
#                     count = count + 1
#                 elif s[i] == 'B':
#                     v = 3
#                     count = count + 1
#                 elif s[i] == 'C':
#                     v = 2
#                     count = count + 1
#                 elif s[i] == 'D':
#                     v = 1
#                     count = count + 1
#                 elif s[i] == 'F':
#                     v = 0
#                     count = count + 1
#                 else:
#                     v='Unknown'
#                     print('Unknown')
#                     break
#                 try:
#                     Sum = int(v + Sum)
#                 except:
#                     print('Unknown')
#                     break
#         else:####################这里很关键
#             print(f"{Sum / count:.2f}")
#     except:
#
#         break

# ##todo
# ''''
# 输入
# 输入包含多组测试样例。每组测试样例包含一个正整数n，表示小明已经堆好的积木堆的个数。
# 接着下一行是n个正整数，表示每一个积木堆的高度h，每块积木高度为1。其中1<=n<=50,1<=h<=100。
# 测试数据保证积木总数能被积木堆数整除。
# 当n=0时，输入结束。
# 输出
# 对于每一组数据，输出将积木堆变成相同高度需要移动的最少积木块的数量。
# 在每组输出结果的下面都输出一个空行。
# '''
# while (1):
#     try:
#         n = int(input())
#         if n != 0:
#             nums = list(map(int, input().split(' ')))
#             counter = 0
#             nums.sort()
#             ave = sum(nums) / (len(nums))
#             for num in nums:
#                 counter = counter + abs(num - ave)  # 按照排序大小依次与均值相减，然后再除2取整（因为遍历2次）
#             print(counter // 2)
#             print()
#         else:
#             break
#     except:
#         break

##todo
# ''''
# 题目描述
# 小明每天的话费是1元，运营商做活动，手机每充值K元就可以获赠1元，一开始小明充值M元，问最多可以用多少天？ 注意赠送的话费也可以参与到奖励规则中
# 输入
# 输入包括多个测试实例。每个测试实例包括2个整数M，K（2<=k<=M<=1000)。M=0，K=0代表输入结束。
# 输出
# 对于每个测试实例输出一个整数，表示M元可以用的天数。

# '''
# while (1):
#     try:
#         m, k = map(int, input().split(' '))
#         day = 0
#         sj = 10000
#         while (sj > k):
#             jl = m // k
#             sx = m % k
#             sj = jl + sx
#             day = day + m + jl + sx
#             m=sj
#         print(day)
#     except:
#         break
##TODO
'''



'''
# while(1):
#     try:
#         n=int(input())
#         for i in range(n):
#             string=str(input())
#             new_s=''
#             s=list(string)
#             for j in range(0,len(string),2):
#                 a=s[i]
#                 s[i]=s[i+1]
#                 new_s=new_s+s[i]
#                 s[i+1]=a
#                 new_s=new_s+s[i+1]
#             print(new_s)
#     except:
#         break
# ##todo
# while(1):
#     try:
#         n=int(input())
#         for _ in range(n):
#             s1=str(input())
#             s2=str(input())
#             mid=int(len(s1)//2)
#             # s11=list(s1)
#             new_s=s1[0:mid]+s2+s1[mid:(2*mid)]
#             print(new_s)
#     except:
#         break
# ##TODO
# '''
# 数字图形
# 样例输入
# 5
# 样例输出 #可以分成3部分输出：空格+正序数字+逆序数字；避免多个print出现空行，使用.join 作为连接；另外使用map(str,range(1,i+1))生成遍历
#     1
#    121
#   12321
#  1234321
# 123454321
#  1234321
#   12321
#    121
#     1
# '''
# while(1):
#     try:
#         n=int(input())
#         for i in range(1,n+1):
#             print(' '*(n-i),end='')
#             print(''.join(map(str,range(1,i+1)))+''.join(map(str,range(i-1,0,-1))))
#         for i in range (n-1,0,-1):
#             print(' '*(n-i),end='')
#             print(''.join(map(str, range(1, i +1))) + ''.join(map(str, range(i - 1, 0, -1))))
#     except:
#         break
# ##todo
# '''
# 题目描述
# 输出一个词组中每个单词的首字母的大写组合。
# 输入:
# 输入的第一行是一个整数n，表示一共有n组测试数据。（输入只有一个n，没有多组n的输入）
# 接下来有n行，每组测试数据占一行，每行有一个词组，每个词组由一个或多个单词组成；每组的单词个数不超过10个，每个单词有一个或多个大写或小写字母组成；
# 单词长度不超过10，由一个或多个空格分隔这些单词。
# 输出:
# 请为每组测试数据输出规定的缩写，每组输出占一行。
# 样例输入
# 1
# ad dfa     fgs
# 样例输出
# ADF
# '''
# # T = int(input())
# # for _ in range(T):
# #     words = input().split()
# #     abbr = ''.join(word[0].upper() for word in words)
# #     print(abbr)
# n=int(input())
# for _ in range(n):
#     sl=list(map(str,input().split(' ')))
# out=''
# for i in range(len(sl)):
#     output=sl[i]
#     if output!='':
#         output=sl[i][0].upper()
#         out=out+output
#     else:
#         continue
# print(out)
# ##todo
# '''
# 题目描述
# 把一个字符三角形掏空，就能节省材料成本，减轻重量，但关键是为了追求另一种视觉效果。在设计的过程中，需要给出各种花纹的材料和大小尺寸的三角形样板，通过电脑临时做出来，以便看看效果。
# 输入
# 每行包含一个字符和一个整数n(0<n<41)，不同的字符表示不同的花纹，整数n表示等腰三角形的高。显然其底边长为2n-1。如果遇到@字符，则表示所做出来的样板三角形已经够了。
# 输出
# 每个样板三角形之间应空上一行，三角形的中间为空。行末没有多余的空格。每条结果后需要再多输出一个空行。
# 样例输入
# X 2
# A 7
# @
# 样例输出
#  X
# XXX
#
#       A
#      A A
#     A   A
#    A     A
#   A       A
#  A         A
# AAAAAAAAAAAAA
# '''
# while (1):
#     try:
#
#         s, n = input().split()###在一行输入2个值，用空格隔开
#         n = int(n)
#         if s != '@':
#             for i in range(1, n, 1):
#                 if i == 1:
#                     # print()
#                     print(' ' * (n - i) + s + ' ' * (2 * (i - 1) - 1) + ' ' * (n - i) + '')
#                 else:
#
#                     print(' ' * (n - i) + s + ' ' * (2 * (i - 1) - 1) + s + ' ' * (n - i) + '')
#
#             print(s * (2 * n - 1), end='')
#             print()
#             print()
#         else:
#             break
#     except:
#         break
# ##todo
# '''
# 栈和队列
# '''
# class MyQueue:
#
#     def __init__(self):
#         self.stackin=[]
#         self.stackout=[]
#
#
#     def push(self, x: int) -> None:
#         self.stackin.append(x)
#
#     def pop(self) -> int:
#         if self.empty:
#             return None
#         if self.stackout:
#             return self.stack_out.pop()
#         else:
#             for i in range (len(self.stackin)):
#                 self.stack_out.append(self.stack_in.pop())
#             return self.stack_out.pop()
#
#     def peek(self) -> int:
#
#         ans=self.pop()
#         self.stack_out.append(ans)
#         return ans
#
#     def empty(self) -> bool:
#
#         return not self.stackin and not self.stackout
# Stack_=MyQueue()
##TODO 回溯算法
# void backtracking(参数) {
#     if (终止条件) {
#         存放结果;
#         return;
#     }
#
#     for (选择：本层集合中元素（树中节点孩子的数量就是集合的大小）) {
#         处理节点;
#         backtracking(路径，选择列表); // 递归
#         回溯，撤销处理结果
#     }
# }

class sum():
    def combine(self,n,k):
        # path=[]
        result=[]
        self.backtracking(n,k,0,1,[],result)
    def backtracking(self,targetsum,k,sum,startidex,path,result):
        if sum>targetsum:
            return
        if len(path)==k:
            if sum==targetsum:

                result.append(path[:])
            return
        for i in range(startidex,9-(k-len(path))+2,1):
            sum+=i
            path.append(i)
            self.backtracking(targetsum,k,sum,i+1,path,result)
            sum-=i
            path.pop()