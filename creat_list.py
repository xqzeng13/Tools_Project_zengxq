# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# #%%
# a=[9,6,3,8,5,2,7,4,1,0]
# for i in range(len(a)):
#     # ListNodeAA=ListNode.next(i)
#     ListNode.next=i
#     ListNode.val=a[i]
#     # print(i,a[i],ListNode.val)
#     print(ListNode.next)
# print(ListNode)
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        self.i=0
    def __str__(self):
        current = self      #指针
        values = []         #值
        while current:
            print(self.i)
            values.append(str(current.val))
            current = current.next
            # print(values,'------',current)
            self.i=self.i+1
        return "->******".join(values)

# 创建一个示例链表
head = ListNode(1)##括号里面是Val;意思是：头部指针值为1；
node2 = ListNode(2)#指针2值为2；
node3 = ListNode(3)#指针3值为3；
node4 = ListNode(4)#指针4值为4；

head.next = node2#头部指针指向下一个指针2；
node2.next = node3#指针2指向下一个指针3；
node3.next = node4#指针3指向下一个指针4；

# 打印链表内容
print(head) #输出头部指针