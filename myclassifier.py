# class Good:
#
#     def __init__(self,hello):
#         self.hello = hello
#
#     def goodmorning(self, X, y):
#         self.hello = 'hello'
#         model = self.hello + X + y
#         return self
#
#     def haha(self, X):
#         print(self.goodmorning())
#         # aha = self.goodmorning()
#         # print("Hello World")
#
# test = Good("good","morning")
# test.haha("555")
# a = [1,2,3]
# print(a*5)
list1 = [1,2,3,4]
list2 = [1,2,3,4]
a = zip(list1,list2)
print(list(a))